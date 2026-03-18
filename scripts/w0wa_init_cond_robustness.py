"""
Robustness to non-varying DE initial conditions.

Test: given a w0waCDM cosmology (at fixed sigma8), how well does the `ns no D`
model perform when the initial condition (non-linear P(k) at z=1.5) is computed with:
  1. The full w0waCDM cosmology (true IC)
  2. Same cosmology but with wa=0 (partial LCDM IC)
  3. Same cosmology but with w0=-1, wa=0 (full LCDM IC)
  4. LCDM IC rescaled by (D_w0waCDM / D_LCDM)^2 at z_IC

All four predictions are compared against the BACCO ground truth for the full w0waCDM cosmology.
Figures are saved to experiments/ns no D/<version>/plots/ic_robustness/.
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Override target_z to match the checkpoint (16-point grid) BEFORE importing models
import DUMP.data.constants as const
const.bacco_target_z = np.array([1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

from DUMP.models import NeuralODE
from DUMP.data.features_engineering import make_features, nonlin_pk
from DUMP.data.constants import bacco_k
import baccoemu
from colossus.cosmology import cosmology as colossus_cosmology

# ── Config ────────────────────────────────────────────────────────────────────
CKPT_PATH = Path("experiments/ns no D/version_xzh7pko1/weights/best_ns no D.ckpt")
SCALERS_PATH = Path("data/scalers.json")
SAVE_DIR = CKPT_PATH.parent.parent / "plots" / "ic_robustness"

FIDUCIAL = {
    "omega_cold": 0.3175, "omega_baryon": 0.049, "hubble": 0.6711,
    "sigma8_cold": 0.834, "ns": 0.9624,
}

# Only w0, wa vary across test cosmologies; all other params (incl. sigma8) are fiducial
TEST_COSMOLOGIES = [
    {**FIDUCIAL, "w0": -0.90, "wa": -0.25},
    {**FIDUCIAL, "w0": -1.10, "wa":  0.20},
    {**FIDUCIAL, "w0": -0.88, "wa": -0.15},
    {**FIDUCIAL, "w0": -1.12, "wa":  0.25},
]

Z_PLOT_INDICES = [0, 4, 9, 14]  # z = 1.4, 1.0, 0.5, 0.0
K_MIN, K_MAX = 0.1, 3.0  # h/Mpc range for error averaging


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_model(ckpt_path, scalers_path):
    with open(scalers_path) as f:
        raw = json.load(f)
    scalers = {"solver_z": raw["solver_z"], "target_z": raw["target_z"], "target": raw["target"]}
    for k, v in raw["features"].items():
        scalers[k] = v

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    model = NeuralODE(
        mlp_params=hp["mlp_params"],
        features_list=hp["features_list"],
        lr=hp["lr"],
        lr_factor=hp["lr_factor"],
        lr_scheduler_patience=hp["scheduler_patience"],
        scalers=scalers,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def inference_with_custom_ic(model, cosmo_features, cosmo_ic, bacco_emulator,
                             ic_rescale_factor=1.0):
    """
    Run model with features from `cosmo_features` but initial condition P(k, z_max)
    from `cosmo_ic`, optionally rescaled by `ic_rescale_factor` in log10 space.
    """
    solver_z = model.solver_z.cpu().numpy()
    target_z = model.target_z.cpu().numpy()

    features_dict = make_features(bacco_emulator, model.features_list, cosmo_features, solver_z)
    feat_tensors = []
    for f in model.features_list:
        ft = torch.tensor(features_dict[f], dtype=torch.float32)
        ft = (ft - getattr(model, f"{f}_mean")) / getattr(model, f"{f}_std")
        feat_tensors.append(ft.unsqueeze(-1) if ft.ndim == 1 else ft)
    features = torch.cat(feat_tensors, dim=-1).unsqueeze(0)

    ic_raw = torch.tensor(nonlin_pk(bacco_emulator, cosmo_ic, target_z)[0], dtype=torch.float32)
    if ic_rescale_factor != 1.0:
        ic_raw = ic_raw + np.log10(ic_rescale_factor)
    ic = (ic_raw - model.target_mean) / model.target_std

    with torch.no_grad():
        sol = model(features, ic.unsqueeze(0))
    return (sol * model.target_std + model.target_mean).squeeze(0)


def _make_colossus_cosmology(cosmo_params, name, de_model='lambda'):
    """Create (or retrieve) a colossus cosmology. Returns the cosmology object."""
    col_params = {
        'flat': True,
        'H0': cosmo_params['hubble'] * 100,
        'Om0': cosmo_params['omega_cold'],
        'Ob0': cosmo_params['omega_baryon'],
        'sigma8': cosmo_params['sigma8_cold'],
        'ns': cosmo_params['ns'],
        'de_model': de_model,
    }
    if de_model == 'w0wa':
        col_params['w0'] = cosmo_params['w0']
        col_params['wa'] = cosmo_params['wa']
    try:
        colossus_cosmology.addCosmology(name, col_params, persistence='')
    except ValueError:
        pass
    return colossus_cosmology.setCosmology(name)


def growth_ratio_squared_at_z(cosmo_params, z_eval):
    """
    Compute (D_w0waCDM(z_eval) / D_LCDM(z_eval))^2.
    Uses colossus growthFactor which is normalised so D(0) = 1.
    """
    # Unique names encoding all relevant parameters to avoid cache collisions
    base = (f"Om{cosmo_params['omega_cold']:.4f}_Ob{cosmo_params['omega_baryon']:.4f}"
            f"_h{cosmo_params['hubble']:.4f}")

    c_lcdm = _make_colossus_cosmology(cosmo_params, f'_gr_lcdm_{base}', de_model='lambda')
    D_lcdm = c_lcdm.growthFactor(z_eval)

    name_de = f'_gr_w0{cosmo_params["w0"]:.4f}_wa{cosmo_params["wa"]:.4f}_{base}'
    c_de = _make_colossus_cosmology(cosmo_params, name_de, de_model='w0wa')
    D_de = c_de.growthFactor(z_eval)

    return (D_de / D_lcdm) ** 2


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = load_model(CKPT_PATH, SCALERS_PATH)
    print(f"  features = {model.features_list}")
    print(f"  target_z = {model.target_z.numpy()}")

    print("Loading BACCO emulator...")
    emu = baccoemu.Matter_powerspectrum()

    target_z = model.target_z.cpu().numpy()
    z_pred = target_z[1:]  # predictions skip z_max (init cond)
    k_mask = (bacco_k > K_MIN) & (bacco_k < K_MAX)

    # ── Run inference ─────────────────────────────────────────────────────
    results = []
    for i, cosmo in enumerate(TEST_COSMOLOGIES):
        print(f"  Cosmology {i+1}/{len(TEST_COSMOLOGIES)}: "
              f"w0={cosmo['w0']:.2f}, wa={cosmo['wa']:.2f}")

        bacco_truth = nonlin_pk(emu, cosmo, target_z)[1:]
        pred_true = model.inference(cosmo, emu).cpu().numpy()

        # IC variants: same sigma8 as target, only (w0, wa) differ
        cosmo_wa0 = {**cosmo, "wa": 0.0}
        pred_wa0 = inference_with_custom_ic(model, cosmo, cosmo_wa0, emu).cpu().numpy()

        cosmo_lcdm = {**cosmo, "w0": -1.0, "wa": 0.0}
        pred_lcdm = inference_with_custom_ic(model, cosmo, cosmo_lcdm, emu).cpu().numpy()

        # Rescaled LCDM IC: multiply P_LCDM(k, z_IC) by (D_w0waCDM / D_LCDM)^2
        ic_scale = growth_ratio_squared_at_z(cosmo, z_eval=target_z[0])
        print(f"    IC rescale factor (D_de/D_lcdm)^2 at z={target_z[0]}: {ic_scale:.6f}")
        pred_lcdm_rescaled = inference_with_custom_ic(
            model, cosmo, cosmo_lcdm, emu, ic_rescale_factor=ic_scale).cpu().numpy()

        results.append(dict(cosmo=cosmo, bacco_truth=bacco_truth,
                            pred_true=pred_true, pred_wa0=pred_wa0,
                            pred_lcdm=pred_lcdm, pred_lcdm_rescaled=pred_lcdm_rescaled))

    # ── Plot 1: raw P(k) (top) + ratios (bottom) at selected redshifts ──
    print("Plotting P(k) + ratios...")
    z_plot_vals = z_pred[Z_PLOT_INDICES]
    n_z = len(Z_PLOT_INDICES)

    for ci, res in enumerate(results):
        cosmo = res["cosmo"]
        fig, axes = plt.subplots(2, n_z, figsize=(5 * n_z, 7),
                                 gridspec_kw={"height_ratios": [2, 1]})

        for ai, zi in enumerate(Z_PLOT_INDICES):
            ax_top = axes[0, ai]
            ax_bot = axes[1, ai]
            truth = 10 ** res["bacco_truth"][zi]
            pred_true = 10 ** res["pred_true"][zi]
            pred_wa0 = 10 ** res["pred_wa0"][zi]
            pred_lcdm = 10 ** res["pred_lcdm"][zi]
            pred_rescaled = 10 ** res["pred_lcdm_rescaled"][zi]

            # ── Top: raw P(k) ──
            ax_top.plot(bacco_k, truth, label="BACCO", color="k", lw=2.5, alpha=0.7)
            ax_top.plot(bacco_k, pred_true, label="True IC", color="C0", lw=1.5)
            ax_top.plot(bacco_k, pred_wa0, label="$w_a=0$ IC", color="C1", lw=1.5, ls="--")
            ax_top.plot(bacco_k, pred_lcdm, label=r"$\Lambda$CDM IC", color="C2", lw=1.5, ls=":")
            ax_top.plot(bacco_k, pred_rescaled, label=r"$\Lambda$CDM IC $\times D^2$", color="C3", lw=1.5, ls="-.")
            ax_top.set_xscale("log")
            ax_top.set_yscale("log")
            ax_top.set_title(f"$z = {z_plot_vals[ai]:.1f}$")
            ax_top.grid(alpha=0.2)
            ax_top.tick_params(labelbottom=False)
            if ai == 0:
                ax_top.set_ylabel(r"$P(k)$ [$h^{-3}$ Mpc$^3$]")

            # ── Bottom: ratios ──
            ax_bot.plot(bacco_k, pred_true / truth, color="C0", lw=2)
            ax_bot.plot(bacco_k, pred_wa0 / truth, color="C1", lw=2, ls="--")
            ax_bot.plot(bacco_k, pred_lcdm / truth, color="C2", lw=2, ls=":")
            ax_bot.plot(bacco_k, pred_rescaled / truth, color="C3", lw=2, ls="-.")
            ax_bot.axhline(1.0, color="gray", ls="-", lw=0.8)
            ax_bot.axhspan(0.99, 1.01, color="gray", alpha=0.15)
            ax_bot.set_xscale("log")
            ax_bot.set_xlabel(r"$k$ [$h$ Mpc$^{-1}$]")
            ax_bot.grid(alpha=0.2)
            if ai == 0:
                ax_bot.set_ylabel(r"$P_{\rm pred} / P_{\rm bacco}$")

        axes[0, -1].legend(fontsize=8, loc="best")
        fig.suptitle(
            f"$w_0={cosmo['w0']:.2f},\\; w_a={cosmo['wa']:.2f}$",
            fontsize=12)
        fig.tight_layout()
        fig.savefig(SAVE_DIR / f"pk_ratios_cosmo{ci}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ── Plot 2: mean relative error vs redshift ──────────────────────────
    print("Plotting error vs redshift...")
    for ci, res in enumerate(results):
        cosmo = res["cosmo"]
        truth = 10 ** res["bacco_truth"]
        fig, ax = plt.subplots(figsize=(8, 4))

        for label, key, style in [("True IC", "pred_true", "o-"),
                                   ("$w_a=0$ IC", "pred_wa0", "s--"),
                                   (r"$\Lambda$CDM IC", "pred_lcdm", "^:"),
                                   (r"$\Lambda$CDM IC $\times D^2$", "pred_lcdm_rescaled", "d-.")]:
            rel_err = 100 * np.abs((10 ** res[key] - truth) / truth)
            ax.plot(z_pred, np.mean(rel_err[:, k_mask], axis=1), style, label=label, markersize=4)

        ax.set_xlabel("Redshift $z$")
        ax.set_ylabel("Mean |relative error| (%)")
        ax.set_title(f"$w_0={cosmo['w0']:.2f},\\; w_a={cosmo['wa']:.2f}$"
                     f" ($k \\in [{K_MIN}, {K_MAX}]\\; h$/Mpc)")
        ax.legend()
        ax.grid(alpha=0.2)
        ax.invert_xaxis()
        fig.tight_layout()
        fig.savefig(SAVE_DIR / f"error_vs_z_cosmo{ci}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ── Plot 3: aggregate errors vs BACCO for all IC variants ──────────
    print("Plotting aggregate errors vs BACCO...")
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    for ci, res in enumerate(results):
        cosmo = res["cosmo"]
        truth = 10 ** res["bacco_truth"]
        lbl = f"$w_0={cosmo['w0']:.2f}, w_a={cosmo['wa']:.2f}$"

        err_true = np.mean(100 * np.abs((10 ** res["pred_true"] - truth) / truth)[:, k_mask], axis=1)
        err_wa0  = np.mean(100 * np.abs((10 ** res["pred_wa0"]  - truth) / truth)[:, k_mask], axis=1)
        err_lcdm = np.mean(100 * np.abs((10 ** res["pred_lcdm"] - truth) / truth)[:, k_mask], axis=1)
        err_rescaled = np.mean(100 * np.abs((10 ** res["pred_lcdm_rescaled"] - truth) / truth)[:, k_mask], axis=1)

        axes[0].plot(z_pred, err_true, "o-", label=lbl, markersize=4)
        axes[1].plot(z_pred, err_wa0, "o-", label=lbl, markersize=4)
        axes[2].plot(z_pred, err_lcdm, "o-", label=lbl, markersize=4)
        axes[3].plot(z_pred, err_rescaled, "o-", label=lbl, markersize=4)

    for ax, title in zip(axes, ["True IC (w0waCDM)", r"IC with $w_a=0$",
                                 r"IC with $w_0=-1, w_a=0$",
                                 r"$\Lambda$CDM IC $\times (D/D_{\Lambda\mathrm{CDM}})^2$"]):
        ax.set_xlabel("Redshift $z$")
        ax.set_ylabel("Mean |rel. error| vs BACCO (%)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        ax.invert_xaxis()

    fig.tight_layout()
    fig.savefig(SAVE_DIR / "error_vs_bacco_aggregate.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"All plots saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
