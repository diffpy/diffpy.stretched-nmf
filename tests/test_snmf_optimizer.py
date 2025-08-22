from pathlib import Path

import numpy as np
import pytest

from diffpy.snmf.snmf_class import SNMFOptimizer

DATA_DIR = Path(__file__).parent / "inputs/test_snmf_optimizer"

# Skip the test entirely if any inputs file is missing
_required = [
    "init_components.txt",
    "source_matrix.txt",
    "init_stretch.txt",
    "init_weights.txt",
]
_missing = [f for f in _required if not (DATA_DIR / f).exists()]
pytestmark = pytest.mark.skipif(_missing, reason=f"Missing test data files: {_missing}")


@pytest.fixture(scope="module")
def inputs():
    return {
        "components": np.loadtxt(DATA_DIR / "init_components.txt", dtype=float),
        "source": np.loadtxt(DATA_DIR / "source_matrix.txt", dtype=float),
        "stretch": np.loadtxt(DATA_DIR / "init_stretch.txt", dtype=float),
        "weights": np.loadtxt(DATA_DIR / "init_weights.txt", dtype=float),
    }


@pytest.mark.slow
def test_final_objective_below_threshold(inputs):
    model = SNMFOptimizer(
        source_matrix=inputs["source"],
        init_weights=inputs["weights"],
        init_components=inputs["components"],
        init_stretch=inputs["stretch"],
        show_plots=False,
        random_state=1,
        min_iter=5,
        max_iter=5,
    )
    model.fit(rho=1e12, eta=610)

    # Basic sanity check and the actual assertion
    assert np.isfinite(model.objective_function)
    assert model.objective_function < 5e6
