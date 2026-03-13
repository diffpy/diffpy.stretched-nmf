from pathlib import Path

import numpy as np

from diffpy.stretched_nmf.snmf_class import SNMFOptimizer

# Input data
DATA_DIR = Path(__file__).resolve().parent / "data/XRD-MgMnO-YCl-real"
source_matrix_file = np.loadtxt(
    DATA_DIR / "source-matrix.txt", dtype=float, skiprows=4
)

# Optional starting initialization
# Without it, would need to provide n_components = 2 to get these results
init_components_file = np.loadtxt(
    DATA_DIR / "init-components.txt", dtype=float
)
init_stretch_file = np.loadtxt(DATA_DIR / "init-stretch.txt", dtype=float)
init_weights_file = np.loadtxt(DATA_DIR / "init-weights.txt", dtype=float)

my_model = SNMFOptimizer(
    show_plots=True,
    rho=1e12,
    eta=610,
)
# Experimentally found best fit parameters for this data
my_model.fit(
    source_matrix=source_matrix_file,
    init_weights=init_weights_file,
    init_components=init_components_file,
    init_stretch=init_stretch_file,
)

print("Done")
np.savetxt(
    "my_norm_components.txt", my_model.components_, fmt="%.6g", delimiter=" "
)
np.savetxt("my_norm_weights.txt", my_model.weights_, fmt="%.6g", delimiter=" ")
np.savetxt("my_norm_stretch.txt", my_model.stretch_, fmt="%.6g", delimiter=" ")
