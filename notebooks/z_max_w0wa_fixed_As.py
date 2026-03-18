"""
Compute and plot z_max(w0, wa): the lowest redshift at which the power
spectrum ratio D²/D²_LCDM first deviates from unity by more than a given
threshold.

Growth factors are normalised at z_ref (deep in matter domination) so that
D(z_ref) = 1 for every cosmology — this enforces the same initial conditions.
"""
import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────
threshold = 0.01        # relative difference threshold (1%)
z_ref = 100.0           # normalisation redshift (matter domination)
z_upper = 10.0          # max redshift to scan
N_z = 1024              # redshift grid resolution
N_w0 = 100              # w0 grid resolution
N_wa = 100              # wa grid resolution
w0_range = (-1.3, 0.0)
wa_range = (-3.0, 0.5)

# ── Fiducial LCDM cosmology ───────────────────────────────────
fid_params = {
    'flat': True,
    'H0': 67.11,
    'Om0': 0.3175,
    'Ob0': 0.049,
    'ns': 0.9624,
    'sigma8': 0.834,  # required by colossus; irrelevant here (growthFactorUnnormalized is amplitude-independent)
}

# ── Redshift grid (exclude z=0 to avoid division issues) ──────
z_grid = np.linspace(1e-4, z_upper, N_z)

# ── Compute LCDM growth factor normalised at z_ref ────────────
cosmology.addCosmology('fid_lcdm', {**fid_params, 'de_model': 'lambda'}, persistence='')
cosmo_lcdm = cosmology.setCosmology('fid_lcdm')
D_lcdm_raw = cosmo_lcdm.growthFactorUnnormalized(z_grid)
D_lcdm_ref = cosmo_lcdm.growthFactorUnnormalized(z_ref)
D_lcdm = D_lcdm_raw / D_lcdm_ref

# ── Scan w0-wa grid ──────────────────────────────────────────
w0_grid = np.linspace(*w0_range, N_w0)
wa_grid = np.linspace(*wa_range, N_wa)
z_max_map = np.full((N_wa, N_w0), np.nan)

for i, wa in enumerate(tqdm(wa_grid, desc="Scanning wa")):
    for j, w0 in enumerate(w0_grid):
        # Skip exact LCDM point
        if w0 == -1.0 and wa == 0.0:
            continue

        name = f'w0{w0:.4f}_wa{wa:.4f}'
        params = {**fid_params, 'de_model': 'w0wa', 'w0': w0, 'wa': wa}
        try:
            cosmology.addCosmology(name, params, persistence='')
        except ValueError:
            pass  # already exists
        cosmo = cosmology.setCosmology(name)

        try:
            D_raw = cosmo.growthFactorUnnormalized(z_grid)
            D_ref = cosmo.growthFactorUnnormalized(z_ref)
        except Exception:
            continue  # unphysical cosmology (e.g. E(z)^2 < 0)

        D_normed = D_raw / D_ref

        ratio = (D_normed / D_lcdm) ** 2  # P(k,z) ratio
        outside = (ratio < 1.0 - threshold) | (ratio > 1.0 + threshold)

        # z_max = lowest z where the difference first reaches threshold
        # (scanning from high z to low z, i.e. last index in the array)
        if not np.any(outside):
            z_max_map[i, j] = 0.0
            continue
        if np.all(outside):
            z_max_map[i, j] = z_upper
            continue

        # Last index where outside is True = highest-z crossing
        idx_cross = np.where(outside)[0][-1]
        if idx_cross < N_z - 1:
            z_lo = z_grid[idx_cross]
            z_hi = z_grid[idx_cross + 1]
            r_lo = ratio[idx_cross]
            r_hi = ratio[idx_cross + 1]
            # Interpolate: find z where ratio crosses 1±threshold
            if r_hi > 1.0:
                boundary = 1.0 + threshold
            else:
                boundary = 1.0 - threshold
            frac = (boundary - r_lo) / (r_hi - r_lo) if r_hi != r_lo else 0.0
            z_max_map[i, j] = z_lo + frac * (z_hi - z_lo)
        else:
            z_max_map[i, j] = z_grid[idx_cross]


# ── Plot ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
pcm = ax.pcolormesh(w0_grid, wa_grid, z_max_map, shading='nearest', cmap='viridis')
cbar = fig.colorbar(pcm, ax=ax, label=r'$z_{\max}$')
ax.plot(-1.0, 0.0, '*', color='red', markersize=12, label=r'$\Lambda$CDM', zorder=5)
w0_line = np.array(w0_range)
ax.plot(w0_line, -1.0 - w0_line, '--', color='white', linewidth=1.5, label=r'$w_0 + w_a = -1$', zorder=4)
w0_line2 = np.array([max(w0_range[0], -0.5), w0_range[1]])
ax.plot(w0_line2, -w0_line2, '--', color='magenta', linewidth=1.5, label=r'$w_0 + w_a = 0$', zorder=4)
ax.set_xlabel(r'$w_0$')
ax.set_ylabel(r'$w_a$')
ax.set_title(r'Lowest $z$ where $D^2/D^2_{\Lambda \mathrm{CDM}} \notin$'
             + f' [{1-threshold:.2f}, {1+threshold:.2f}]'
             + r' (same IC, $z_{\rm ref}=' + f'{z_ref:.0f}' + r'$)')
ax.legend(loc='lower left')
plt.tight_layout()
plt.savefig('notebooks/z_max_w0wa_fixed_As.png', dpi=150)
plt.show()
print("Saved to notebooks/z_max_w0wa_fixed_As.png")

# ── Table along w0 + wa = -1 line ────────────────────────────
print(f"\n{'w0':>8s}  {'wa':>8s}  {'z_max':>8s}")
print("-" * 30)
for w0 in w0_grid:
    wa_target = -1.0 - w0
    if wa_target < wa_range[0] or wa_target > wa_range[1]:
        continue
    j = np.argmin(np.abs(w0_grid - w0))
    i = np.argmin(np.abs(wa_grid - wa_target))
    val = z_max_map[i, j]
    print(f"{w0:8.3f}  {wa_target:8.3f}  {val:8.3f}")
