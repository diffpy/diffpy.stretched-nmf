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
pytestmark = pytest.mark.skipif(
    _missing, reason=f"Missing test data files: {_missing}"
)


@pytest.fixture(scope="module")
def inputs():
    return {
        "components": np.loadtxt(
            DATA_DIR / "init_components.txt", dtype=float
        ),
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


@pytest.mark.parametrize(
    "inputs, expected",
    # inputs tuple:
    # (components, residuals, stretch, rho, eta, spline smoothness operator)
    [
        # Case 0: No smoothness or sparsity penalty, reduces to NMF objective
        # residual Frobenius norm^2 = 3^2 + 4^2 = 25 -> 0.5 * 25 = 12.5
        (
            (
                np.array([[0.0, 0.0], [3.0, 4.0]]),
                np.array([[0.0, 0.0], [3.0, 4.0]]),
                np.ones((2, 2)),
                0.0,
                0.0,
                np.zeros((2, 2)),
            ),
            12.5,
        ),
        # Case 1: rho = 0, sparsity penalty only
        # sqrt components sum = 1 + 2 + 3 + 4 = 10 -> eta * 10 = 5
        # residual term remains 12.5 -> total = 17.5
        (
            (
                np.array([[1.0, 4.0], [9.0, 16.0]]),
                np.array([[3.0, 4.0], [0.0, 0.0]]),
                np.ones((2, 2)),
                0.0,
                0.5,
                np.zeros((2, 2)),
            ),
            17.5,
        ),
        # Case 2: eta = 0, smoothness penalty only
        # residual = 12.5, smoothing = 0.5 * 1 * 1 = 0.5 -> total = 13.0
        (
            (
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.array([[3.0, 4.0], [0.0, 0.0]]),
                np.array([[1.0, 2.0]]),
                1.0,
                0.0,
                np.array([[1.0, -1.0]]),
            ),
            13.0,
        ),
        # Case 3: penalty for smoothness and sparsity
        # residual = 2.5, sparsity = 1.5, smoothing = 9 -> total = 13.0
        (
            (
                np.array([[1.0, 4.0]]),
                np.array([[1.0, 2.0]]),
                np.array([[1.0, 4.0]]),
                2.0,
                0.5,
                np.array([[3.0, 0.0]]),
            ),
            13.0,
        ),
    ],
)
def test_compute_objective_function(inputs, expected):
    components, residuals, stretch, rho, eta, operator = inputs
    result = SNMFOptimizer._compute_objective_function(
        components=components,
        residuals=residuals,
        stretch=stretch,
        rho=rho,
        eta=eta,
        spline_smooth_operator=operator,
    )
    assert np.isclose(result, expected)
