"""
Constants for BACCO emulator and cosmology parameters.
"""
import numpy as np

# k-modes grid for power spectrum
bacco_k = np.logspace(np.log10(1e-2), np.log10(4.9), num=30)

# Training and test parameter ranges
bacco_train_ranges = {
    "omega_cold": [0.23, 0.4],
    "omega_baryon": [0.04, 0.06],
    "hubble": [0.6, 0.8],
    "w0": [-1.0, -1.0],
    "wa": [0.0, 0.0],
    "sigma8_cold": [0.73, 0.9],
    "ns": [0.92, 1.01]
}

bacco_test_ranges = {
    "omega_cold": [0.23, 0.4],
    "omega_baryon": [0.04, 0.06],
    "hubble": [0.6, 0.8],
    "w0": [-1.15, -0.85],
    "wa": [-0.3, 0.3],
    "sigma8_cold": [0.73, 0.9],
    "ns": [0.92, 1.01]
}

solver_dz = 1e-2
#bacco_target_z = np.array([1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
bacco_target_z = np.array([1.5, 1.0, 0.5, 0.0])
