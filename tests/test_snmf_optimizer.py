import numpy as np
import pytest

from diffpy.stretched_nmf.snmf_class import SNMFOptimizer


def test_fit_recovers_rank_one_factors():
    expected_components = np.array(
        [
            [0.20],
            [0.75],
            [1.20],
            [0.80],
            [0.30],
        ]
    )
    expected_weights = np.array(
        [
            [0.20, 0.60, 1.00, 0.40],
        ]
    )
    source = expected_components @ expected_weights

    model = SNMFOptimizer(
        n_components=1,
        show_plots=False,
        random_state=1,
        min_iter=0,
        max_iter=2,
        rho=0.0,
        eta=0.0,
    )
    model.fit(source_matrix=source)

    assert np.isfinite(model.objective_function_)
    assert np.allclose(
        model.components_, expected_components, rtol=0.2, atol=0.1
    )
    assert np.allclose(model.weights_, expected_weights, rtol=0.2, atol=0.1)


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
