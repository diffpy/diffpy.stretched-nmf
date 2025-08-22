import numpy as np

from diffpy.snmf.snmf_class import SNMFOptimizer

# Example input files (not provided)
init_components_file = np.loadtxt("inputs/init_components.txt", dtype=float)
source_matrix_file = np.loadtxt("inputs/source_matrix.txt", dtype=float)
init_stretch_file = np.loadtxt("inputs/init_stretch.txt", dtype=float)
init_weights_file = np.loadtxt("inputs/init_weights.txt", dtype=float)

my_model = SNMFOptimizer(
    source_matrix=source_matrix_file,
    init_weights=init_weights_file,
    init_components=init_components_file,
    init_stretch=init_stretch_file,
    show_plots=True,
)
my_model.fit(rho=1e12, eta=610)

print("Done")
np.savetxt("my_norm_components.txt", my_model.components_, fmt="%.6g", delimiter=" ")
np.savetxt("my_norm_weights.txt", my_model.weights_, fmt="%.6g", delimiter=" ")
np.savetxt("my_norm_stretch.txt", my_model.stretch_, fmt="%.6g", delimiter=" ")
