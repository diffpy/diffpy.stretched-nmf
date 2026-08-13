import numpy as np
import pytest
from scipy.sparse import csr_matrix

from diffpy.stretched_nmf.snmf_class import (
    SNMFOptimizer,
    _cubic_largest_real_root,
)


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


def test_cubic_largest_real_root_preserves_tiny_zero_q_root():
    root = _cubic_largest_real_root(np.array([[-1e-300]]), np.zeros((1, 1)))

    np.testing.assert_allclose(root, [[1e-150]], rtol=1e-12, atol=0)


def test_failed_component_update_restores_previous_components():
    model = SNMFOptimizer(n_components=1, eta=0.0)
    model.signal_length_ = model.n_signals_ = model.n_components_ = 1
    model.components_ = np.array([[1.0]])
    max_float = np.finfo(float).max
    model.weights_ = np.array([[np.sqrt(max_float)]])
    model.stretch_ = np.ones((1, 1))
    model._source_matrix = np.zeros((1, 1))
    model._fill_tail_zero = True
    model._outer_iter = model._inner_iter = 0
    model.objective_function_ = 0.0
    model._compute_stretched_components = lambda: (
        np.zeros((1, 1)),
        None,
        None,
    )
    model._compute_component_gradient_zero_tail = lambda residuals: np.array(
        [[np.finfo(float).max]]
    )
    model._get_residual_matrix = lambda **kwargs: np.zeros((1, 1))
    model._get_objective_function = lambda **kwargs: 1.0

    with np.errstate(over="ignore"):
        model._update_components()

    np.testing.assert_array_equal(model.components_, [[1.0]])


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


def test_regularize_function_hessian_has_expected_structure():
    model = SNMFOptimizer(n_components=2, rho=0.5)
    model.n_components_ = 2
    model.n_signals_ = 3
    model._spline_smooth_operator = csr_matrix(
        [[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]]
    )

    residuals = np.array([[2.0, -1.0, 4.0], [1.0, 3.0, -2.0]])
    d_stretch_comps = np.array(
        [
            [1.0, 0.0, 1.0, 2.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0, 3.0, -1.0],
        ]
    )
    dd_stretch_comps = np.array(
        [
            [0.5, 1.0, 0.0, 1.0, 0.0, -0.5],
            [1.0, 0.0, 0.25, 0.0, 1.0, 0.5],
        ]
    )
    model._stretch_residual_and_derivatives = lambda stretch: (
        residuals,
        d_stretch_comps,
        dd_stretch_comps,
    )

    hessian = model._regularize_function_hessian(np.ones((2, 3)))

    expected = np.array(
        [
            [3.5, -0.5, 0.0, 2.0, 0.0, 0.0],
            [-0.5, 1.0, -0.5, 0.0, 3.0, 0.0],
            [0.0, -0.5, 2.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 6.5, -0.5, 0.0],
            [0.0, 3.0, 0.0, -0.5, 13.0, -0.5],
            [0.0, 0.0, 0.0, 0.0, -0.5, -0.5],
        ]
    )
    assert hessian.shape == (6, 6)
    assert np.allclose(hessian, hessian.T)
    assert np.allclose(hessian, expected)
