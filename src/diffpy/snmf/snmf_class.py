import cvxpy as cp
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, diags

from diffpy.snmf.plotter import SNMFPlotter


class SNMFOptimizer:
    """An implementation of stretched NMF (sNMF), including sparse
    stretched NMF.

    Instantiating the SNMFOptimizer class prepares initial guesses and sets up the
    optimization. It can then be run using fit().
    The results matrices can be accessed as instance attributes
    of the class (components_, weights_, and stretch_).

    For more information on sNMF, please reference:
    Gu, R., Rakita, Y., Lan, L. et al. Stretched non-negative matrix factorization.
    npj Comput Mater 10, 193 (2024). https://doi.org/10.1038/s41524-024-01377-5

    Attributes
    ----------
    source_matrix : ndarray
        The original, unmodified data to be decomposed and later, compared against.
        Shape is (length_of_signal, number_of_signals).
    stretch_ : ndarray
        The best guess (or while running, the current guess) for the stretching
        factor matrix.
    components_ : ndarray
        The best guess (or while running, the current guess) for the matrix of
        component intensities.
    weights_ : ndarray
        The best guess (or while running, the current guess) for the matrix of
        component weights.
    rho : float
        The stretching factor that influences the decomposition. Zero corresponds to no
        stretching present. Relatively insensitive and typically adjusted in powers of 10.
    eta : float
        The sparsity factor that influences the decomposition. Should be set to zero for
        non-sparse data such as PDF. Can be used to improve results for sparse data such
        as XRD, but due to instability, should be used only after first selecting the
        best value for rho. Suggested adjustment is by powers of 2.
    max_iter : int
        The maximum number of times to update each of stretch, components, and weights before stopping
        the optimization.
    min_iter : int
        The minimum number of times to update each of stretch, components, and weights before terminating
        the optimization due to low/no improvement.
    tol : float
        The convergence threshold. This is the minimum fractional improvement in the
        objective function to allow without terminating the optimization.
    n_components : int
        The number of components to extract from source_matrix. Must be provided when and only when
        init_weights is not provided.
    random_state : int
        The seed for the initial guesses at the matrices (stretch, components, and weights) created by
        the decomposition.
    num_updates : int
        The total number of times that any of (stretch, components, and weights) have had their values changed.
        If not terminated by other means, this value is used to stop when reaching max_iter.
    objective_difference : float
        The change in the objective function value since the last update. A negative value
        means that the result improved.
    """

    def __init__(
        self,
        source_matrix,
        init_weights=None,
        init_components=None,
        init_stretch=None,
        max_iter=500,
        min_iter=20,
        tol=5e-7,
        n_components=None,
        random_state=None,
        show_plots=False,
    ):
        """Initialize an instance of sNMF.

        Parameters
        ----------
        source_matrix : ndarray
            The data to be decomposed. Shape is (length_of_signal, number_of_conditions).
        init_weights : ndarray Optional  Default = rng.beta(a=2.0, b=2.0, size=(n_components, n_signals))
            The initial guesses for the component weights at each stretching condition.
            Shape is (number_of_components, number_of_signals) Must provide exactly one
            of this or n_components.
        init_components : ndarray Optional  Default = rng.random((self.signal_length, self.n_components))
            The initial guesses for the intensities of each component per
            row/sample/angle. Shape is (length_of_signal, number_of_components).
        init_stretch : ndarray Optional  Default = np.ones((self.n_components, self.n_signals)) + self._rng.normal(
                0, 1e-3, size=(self.n_components, self.n_signals)
            The initial guesses for the stretching factor for each component, at each
            condition (for each signal). Shape is (number_of_components, number_of_signals).
        max_iter : int Optional Default = 500
            The maximum number of times to update each of A, X, and Y before stopping
            the optimization.
        tol : float Optional  Default = 5e-7
            The convergence threshold. This is the minimum fractional improvement in the
            objective function to allow without terminating the optimization. Note that
            a minimum of 20 updates are run before this parameter is checked.
        n_components : int  Optional  Default = None
            The number of components to extract from source_matrix. Must be provided when and only when
            Y0 is not provided.
        random_state : int  Optional  Default = None
            The seed for the initial guesses at the matrices (A, X, and Y) created by
            the decomposition.
        show_plots : boolean Optional  Default = False
            Enables plotting at each step of the decomposition.
        """

        self.source_matrix = source_matrix
        self.tol = tol
        self.max_iter = max_iter
        self.min_iter = min_iter
        # Capture matrix dimensions
        self.signal_length, self.n_signals = source_matrix.shape
        self.num_updates = 0
        self._rng = np.random.default_rng(random_state)
        self.plotter = SNMFPlotter() if show_plots else None

        # Enforce exclusive specification of n_components or init_weights
        if (n_components is None and init_weights is None) or (
            n_components is not None and init_weights is not None
        ):
            raise ValueError(
                "Conflicting source for n_components. Must provide either init_weights or n_components "
                "directly, but not both."
            )

        # Initialize weights and determine number of components
        if init_weights is None:
            self.n_components = n_components
            self.weights_ = self._rng.beta(
                a=2.0, b=2.0, size=(self.n_components, self.n_signals)
            )
        else:
            self.n_components = init_weights.shape[0]
            self.weights_ = init_weights

        # Initialize stretching matrix if not provided
        if init_stretch is None:
            self.stretch_ = np.ones(
                (self.n_components, self.n_signals)
            ) + self._rng.normal(
                0, 1e-3, size=(self.n_components, self.n_signals)
            )
        else:
            self.stretch_ = init_stretch

        # Initialize component matrix if not provided
        if init_components is None:
            self.components_ = self._rng.random(
                (self.signal_length, self.n_components)
            )
        else:
            self.components_ = init_components

        # Enforce non-negativity in our initial guesses
        self.components_ = np.maximum(0, self.components_)
        self.weights_ = np.maximum(0, self.weights_)

        # Store the initial components, weights, and stretch
        self.init_components = self.components_.copy()
        self.init_weights = self.weights_.copy()
        self.init_stretch = self.stretch_.copy()

        # Second-order spline: Tridiagonal (-2 on diagonal, 1 on sub/superdiagonals)
        self._spline_smooth_operator = 0.25 * diags(
            [1, -2, 1],
            offsets=[0, 1, 2],
            shape=(self.n_signals - 2, self.n_signals),
        )

    def fit(self, rho=0, eta=0, reset=True):
        """Run the sNMF optimization with the given parameters, using
        the setup from __init__.

        Parameters
        ----------
        rho : float Optional  Default = 0
            The stretching factor that influences the decomposition. Zero corresponds to no
            stretching present. Relatively insensitive and typically adjusted in powers of 10.
        eta : int Optional  Default = 0
            The sparsity factor that influences the decomposition. Should be set to zero for
            non-sparse data such as PDF. Can be used to improve results for sparse data such
            as XRD, but due to instability, should be used only after first selecting the
            best value for rho. Suggested adjustment is by powers of 2.
        reset : boolean Optional  Default = True
            Whether to return to the initial set of components_, weights_, and stretch_ before
            running the optimization. When set to False, sequential calls to fit() will use the
            output of the previous fit() as their input.
        """

        if reset:
            self.components_ = self.init_components.copy()
            self.weights_ = self.init_weights.copy()
            self.stretch_ = self.init_stretch.copy()

        self.rho = rho
        self.eta = eta

        # Set up residual matrix, objective function, and history
        self.residuals = self.get_residual_matrix()
        self.objective_function = self.get_objective_function()
        self.best_objective = self.objective_function
        self.best_matrices = [
            self.components_.copy(),
            self.weights_.copy(),
            self.stretch_.copy(),
        ]
        self.objective_difference = None
        self._objective_history = [self.objective_function]

        # Set up tracking variables for update_components()
        self._prev_components = None
        self._grad_components = np.zeros_like(self.components_)
        self._prev_grad_components = np.zeros_like(self.components_)

        regularization_term = (
            0.5
            * self.rho
            * np.linalg.norm(
                self._spline_smooth_operator @ self.stretch_.T, "fro"
            )
            ** 2
        )
        sparsity_term = self.eta * np.sum(
            np.sqrt(self.components_)
        )  # Square root penalty
        print(
            f"Start, Objective function: {self.objective_function:.5e}"
            f", Obj - reg/sparse: {self.objective_function - regularization_term - sparsity_term:.5e}"
        )

        # Main optimization loop
        for outiter in range(self.max_iter):
            self.outiter = outiter
            self.outer_loop()
            # Print diagnostics
            regularization_term = (
                0.5
                * self.rho
                * np.linalg.norm(
                    self._spline_smooth_operator @ self.stretch_.T, "fro"
                )
                ** 2
            )
            sparsity_term = self.eta * np.sum(
                np.sqrt(self.components_)
            )  # Square root penalty
            print(
                f"Obj fun: {self.objective_function:.5e}, "
                f"Obj - reg/sparse: {self.objective_function - regularization_term - sparsity_term:.5e}, "
                f"Iter: {self.outiter}"
            )

            # Convergence check: Stop if diffun is small and at least min_iter iterations have passed
            print(
                "Checking if ",
                self.objective_difference,
                " < ",
                self.objective_function * self.tol,
            )
            if (
                self.objective_difference < self.objective_function * self.tol
                and outiter >= self.min_iter
            ):
                break

        self.normalize_results()

        return self

    def normalize_results(self):
        # Select our best results for normalization
        self.components_ = self.best_matrices[0]
        self.weights_ = self.best_matrices[1]
        self.stretch_ = self.best_matrices[2]

        # Normalize weights/stretch first
        weights_row_max = np.max(self.weights_, axis=1, keepdims=True)
        self.weights_ = self.weights_ / weights_row_max
        stretch_row_max = np.max(self.stretch_, axis=1, keepdims=True)
        self.stretch_ = self.stretch_ / stretch_row_max

        # effectively just re-running with component updates only vs normalized weights/stretch
        self._grad_components = np.zeros_like(
            self.components_
        )  # Gradient of X (zeros for now)
        self._prev_grad_components = np.zeros_like(
            self.components_
        )  # Previous gradient of X (zeros for now)
        self.residuals = self.get_residual_matrix()
        self.objective_function = self.get_objective_function()
        self.objective_difference = None
        self._objective_history = [self.objective_function]
        self.outiter = 0
        self.iter = 0
        for outiter in range(self.max_iter):
            if iter == 1:
                self.iter = 1  # So step size can adapt without an inner loop
            self.update_components()
            self.residuals = self.get_residual_matrix()
            self.objective_function = self.get_objective_function()
            print(
                f"Objective function after normalize_components: {self.objective_function:.5e}"
            )
            self._objective_history.append(self.objective_function)
            self.objective_difference = (
                self._objective_history[-2] - self._objective_history[-1]
            )
            if self.plotter is not None:
                self.plotter.update(
                    components=self.components_,
                    weights=self.weights_,
                    stretch=self.stretch_,
                    update_tag="normalize components",
                )
            if (
                self.objective_difference < self.objective_function * self.tol
                and outiter >= 7
            ):
                break

    def outer_loop(self):
        for iter in range(4):
            self.iter = iter
            self._prev_grad_components = self._grad_components.copy()
            self.update_components()
            self.residuals = self.get_residual_matrix()
            self.objective_function = self.get_objective_function()
            print(
                f"Objective function after update_components: {self.objective_function:.5e}"
            )
            self._objective_history.append(self.objective_function)
            self.objective_difference = (
                self._objective_history[-2] - self._objective_history[-1]
            )
            if self.objective_function < self.best_objective:
                self.best_objective = self.objective_function
                self.best_matrices = [
                    self.components_.copy(),
                    self.weights_.copy(),
                    self.stretch_.copy(),
                ]
            if self.plotter is not None:
                self.plotter.update(
                    components=self.components_,
                    weights=self.weights_,
                    stretch=self.stretch_,
                    update_tag="components",
                )

            self.update_weights()
            self.residuals = self.get_residual_matrix()
            self.objective_function = self.get_objective_function()
            print(
                f"Objective function after update_weights: {self.objective_function:.5e}"
            )
            self._objective_history.append(self.objective_function)
            self.objective_difference = (
                self._objective_history[-2] - self._objective_history[-1]
            )
            if self.objective_function < self.best_objective:
                self.best_objective = self.objective_function
                self.best_matrices = [
                    self.components_.copy(),
                    self.weights_.copy(),
                    self.stretch_.copy(),
                ]
            if self.plotter is not None:
                self.plotter.update(
                    components=self.components_,
                    weights=self.weights_,
                    stretch=self.stretch_,
                    update_tag="weights",
                )

            self.objective_difference = (
                self._objective_history[-2] - self._objective_history[-1]
            )
            if (
                self._objective_history[-3] - self.objective_function
                < self.objective_difference * 1e-3
            ):
                break

        self.update_stretch()
        self.residuals = self.get_residual_matrix()
        self.objective_function = self.get_objective_function()
        print(
            f"Objective function after update_stretch: {self.objective_function:.5e}"
        )
        self._objective_history.append(self.objective_function)
        self.objective_difference = (
            self._objective_history[-2] - self._objective_history[-1]
        )
        if self.objective_function < self.best_objective:
            self.best_objective = self.objective_function
            self.best_matrices = [
                self.components_.copy(),
                self.weights_.copy(),
                self.stretch_.copy(),
            ]
        if self.plotter is not None:
            self.plotter.update(
                components=self.components_,
                weights=self.weights_,
                stretch=self.stretch_,
                update_tag="stretch",
            )

    def get_residual_matrix(self, components=None, weights=None, stretch=None):
        """Return the residuals (difference) between the source matrix
        and its reconstruction.

        Parameters
        ----------
        components : (signal_len, n_components) array, optional
        weights    : (n_components, n_signals) array, optional
        stretch    : (n_components, n_signals) array, optional

        Returns
        -------
        residuals : (signal_len, n_signals) array
        """

        if components is None:
            components = self.components_
        if weights is None:
            weights = self.weights_
        if stretch is None:
            stretch = self.stretch_

        reconstructed_matrix = reconstruct_matrix(components, weights, stretch)
        residuals = reconstructed_matrix - self.source_matrix

        return residuals

    def get_objective_function(self, residuals=None, stretch=None):
        """Return the objective value, passing stored attributes or
        overrides to _compute_objective_function().

        Parameters
        ----------
        residuals : ndarray, optional
            Residual matrix to use instead of self.residuals.
        stretch : ndarray, optional
            Stretch matrix to use instead of self.stretch_.

        Returns
        -------
        float
            Current objective function value.
        """
        return SNMFOptimizer._compute_objective_function(
            components=self.components_,
            residuals=self.residuals if residuals is None else residuals,
            stretch=self.stretch_ if stretch is None else stretch,
            rho=self.rho,
            eta=self.eta,
            spline_smooth_operator=self._spline_smooth_operator,
        )

    def compute_stretched_components(
        self, components=None, weights=None, stretch=None
    ):
        """Interpolates each component along its sample axis according
        to per-(component, signal) stretch factors, then applies
        per-(component, signal) weights. Also computes the first and
        second derivatives with respect to stretch. Left and right,
        respectively, refer to the sample prior to and subsequent to the
        interpolated sample's position.

        Inputs
        ------
        components : array, shape (signal_len, n_components)
            Each column is a component with signal_len samples.
        weights : array, shape (n_components, n_signals)
            Per-(component, signal) weights.
        stretch : array, shape (n_components, n_signals)
            Per-(component, signal) stretch factors.

        Outputs
        -------
        stretched_components : array, shape (signal_len, n_components * n_signals)
            Interpolated and weighted components.
        d_stretched_components : array, shape (signal_len, n_components * n_signals)
            First derivatives with respect to stretch.
        dd_stretched_components : array, shape (signal_len, n_components * n_signals)
            Second derivatives with respect to stretch.
        """

        # --- Defaults ---
        if components is None:
            components = self.components_
        if weights is None:
            weights = self.weights_
        if stretch is None:
            stretch = self.stretch_

        # Dimensions
        signal_len = components.shape[0]  # number of samples
        n_components = components.shape[1]  # number of components
        n_signals = weights.shape[1]  # number of signals

        # Guard stretches
        eps = 1e-8
        stretch = np.clip(stretch, eps, None)
        stretch_inv = 1.0 / stretch

        # Apply stretching to the original sample indices, represented as a "time-stretch"
        t = (
            np.arange(signal_len, dtype=float)[:, None, None]
            * stretch_inv[None, :, :]
        )
        # has shape (signal_len, n_components, n_signals)

        # For each stretched coordinate, find its prior integer (original) index and their difference
        i0 = np.floor(t).astype(np.int64)  # prior original index
        alpha = t - i0.astype(float)  # fractional distance between left/right

        # Clip indices to valid range (0, signal_len - 1) to maintain original size
        max_idx = signal_len - 1
        i0 = np.clip(i0, 0, max_idx)
        i1 = np.clip(i0 + 1, 0, max_idx)

        # Gather sample values
        comps_3d = components[
            :, :, None
        ]  # expand components by a dimension for broadcasting across n_signals
        c0 = np.take_along_axis(comps_3d, i0, axis=0)  # left sample values
        c1 = np.take_along_axis(comps_3d, i1, axis=0)  # right sample values

        # Linear interpolation to determine stretched sample values
        interp = c0 * (1.0 - alpha) + c1 * alpha
        interp_weighted = interp * weights[None, :, :]

        # Derivatives
        di = -t * stretch_inv[None, :, :]  # first-derivative coefficient
        ddi = (
            -di * stretch_inv[None, :, :] * 2.0
        )  # second-derivative coefficient

        d_unweighted = c0 * (-di) + c1 * di
        dd_unweighted = c0 * (-ddi) + c1 * ddi

        d_weighted = d_unweighted * weights[None, :, :]
        dd_weighted = dd_unweighted * weights[None, :, :]

        # Flatten back to expected shape (signal_len, n_components * n_signals)
        return (
            interp_weighted.reshape(signal_len, n_components * n_signals),
            d_weighted.reshape(signal_len, n_components * n_signals),
            dd_weighted.reshape(signal_len, n_components * n_signals),
        )

    def apply_transformation_matrix(
        self, stretch=None, weights=None, residuals=None
    ):
        """Computes the transformation matrix `stretch_transformed` for
        residuals, using scaling matrix `stretch` and weight
        coefficients `weights`."""

        if stretch is None:
            stretch = self.stretch_
        if weights is None:
            weights = self.weights_
        if residuals is None:
            residuals = self.residuals

        # Compute scaling matrix
        stretch_tiled = np.tile(
            stretch.reshape(1, self.n_signals * self.n_components, order="F")
            ** -1,
            (self.signal_length, 1),
        )

        # Compute indices
        indices = np.arange(self.signal_length)[:, None] * stretch_tiled

        # Weighting coefficients
        weights_tiled = np.tile(
            weights.reshape(1, self.n_signals * self.n_components, order="F"),
            (self.signal_length, 1),
        )

        # Compute floor indices
        floor_indices = np.floor(indices).astype(int)
        floor_indices_1 = np.minimum(floor_indices + 1, self.signal_length)
        floor_indices_2 = np.minimum(floor_indices_1 + 1, self.signal_length)

        # Compute fractional part
        fractional_indices = indices - floor_indices

        # Expand row indices
        repm = np.tile(
            np.arange(self.n_components), (self.signal_length, self.n_signals)
        )

        # Compute transformations
        kron = np.kron(residuals, np.ones((1, self.n_components)))
        fractional_kron = kron * fractional_indices
        fractional_weights = (fractional_indices - 1) * weights_tiled

        # Construct sparse matrices
        x2 = coo_matrix(
            (
                (-kron * fractional_weights).flatten(),
                (floor_indices_1.flatten() - 1, repm.flatten()),
            ),
            shape=(self.signal_length + 1, self.n_components),
        ).tocsc()
        x3 = coo_matrix(
            (
                (fractional_kron * weights_tiled).flatten(),
                (floor_indices_2.flatten() - 1, repm.flatten()),
            ),
            shape=(self.signal_length + 1, self.n_components),
        ).tocsc()

        # Combine the last row into previous, then remove the last row
        x2[self.signal_length - 1, :] += x2[self.signal_length, :]
        x3[self.signal_length - 1, :] += x3[self.signal_length, :]
        x2 = x2[:-1, :]
        x3 = x3[:-1, :]

        stretch_transformed = x2 + x3

        return stretch_transformed

    def solve_quadratic_program(self, t, m):
        """
        Solves the quadratic program for updating y in stretched NMF:

            min J(y) = 0.5 * y^T q y + d^T y
            subject to: 0 ≤ y ≤ 1

        Parameters:
        - t: (N, k) ndarray
        - source_matrix_col: (N,) column of source_matrix for the corresponding m

        Returns:
        - y: (k,) optimal solution
        """

        source_matrix_col = self.source_matrix[:, m]

        # Compute q and d
        q = t.T @ t  # Gram matrix (k x k)
        d = -t.T @ source_matrix_col  # Linear term (k,)

        k = q.shape[0]  # Number of variables

        # Regularize q to ensure positive semi-definiteness
        reg_factor = 1e-8 * np.linalg.norm(
            q, ord="fro"
        )  # Adaptive regularization, original was fixed
        q += np.eye(k) * reg_factor

        # Define optimization variable
        y = cp.Variable(k)

        # Define quadratic objective
        objective = cp.Minimize(0.5 * cp.quad_form(y, q) + d.T @ y)

        # Define constraints (0 ≤ y ≤ 1)
        constraints = [y >= 0, y <= 1]

        # Solve using a QP solver
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        # Get the solution
        return np.maximum(
            y.value, 0
        )  # Ensure non-negative values in case of solver tolerance issues

    def update_components(self):
        """Updates `components` using gradient-based optimization with
        adaptive step size."""
        # Compute stretched components using the interpolation function
        stretched_components, _, _ = (
            self.compute_stretched_components()
        )  # Discard the derivatives
        # Compute reshaped_stretched_components and component_residuals
        intermediate_reshaped = stretched_components.flatten(
            order="F"
        ).reshape(
            (self.signal_length * self.n_signals, self.n_components), order="F"
        )
        reshaped_stretched_components = intermediate_reshaped.sum(
            axis=1
        ).reshape((self.signal_length, self.n_signals), order="F")
        component_residuals = (
            reshaped_stretched_components - self.source_matrix
        )
        # Compute gradient
        self._grad_components = self.apply_transformation_matrix(
            residuals=component_residuals
        ).toarray()  # toarray equivalent of full, make non-sparse

        # Compute initial step size `initial_step_size`
        initial_step_size = np.linalg.eigvalsh(
            self.weights_.T @ self.weights_
        ).max() * np.max([self.stretch_.max(), 1 / self.stretch_.min()])
        # Compute adaptive step size `step_size`
        if self.outiter == 0 and self.iter == 0:
            step_size = initial_step_size
        else:
            num = np.sum(
                (self._grad_components - self._prev_grad_components)
                * (self.components_ - self._prev_components)
            )  # Element-wise multiplication
            denom = (
                np.linalg.norm(self.components_ - self._prev_components, "fro")
                ** 2
            )  # Frobenius norm squared
            step_size = num / denom if denom > 0 else initial_step_size
            if step_size <= 0:
                step_size = initial_step_size

        # Store our old X before updating because it is used in step selection
        self._prev_components = self.components_.copy()

        while True:  # iterate updating components
            components_step = (
                self._prev_components - self._grad_components / step_size
            )
            # Solve x^3 + p*x + q = 0 for the largest real root
            self.components_ = np.square(
                cubic_largest_real_root(
                    -components_step, self.eta / (2 * step_size)
                )
            )
            # Mask values that should be set to zero
            mask = (
                self.components_**2 * step_size / 2
                - step_size * self.components_ * components_step
                + self.eta * np.sqrt(self.components_)
                < 0
            )
            self.components_ = mask * self.components_

            objective_improvement = self._objective_history[
                -1
            ] - self.get_objective_function(
                residuals=self.get_residual_matrix()
            )

            # Check if objective function improves
            if objective_improvement > 0:
                break
            # If not, increase step_size (step size)
            step_size *= 2
            if np.isinf(step_size):
                break

    def update_weights(self):
        """Updates weights by building the stretched component matrix
        `stretched_comps` with np.interp and solving a quadratic program
        for each signal."""

        sample_indices = np.arange(self.signal_length)
        for signal in range(self.n_signals):
            # Stretch factors for this signal across components:
            this_stretch = self.stretch_[:, signal]
            # Build stretched_comps[:, k] by interpolating component at frac. pos. index / this_stretch[comp]
            stretched_comps = np.empty(
                (self.signal_length, self.n_components),
                dtype=self.components_.dtype,
            )
            for comp in range(self.n_components):
                pos = sample_indices / this_stretch[comp]
                stretched_comps[:, comp] = np.interp(
                    pos,
                    sample_indices,
                    self.components_[:, comp],
                    left=self.components_[0, comp],
                    right=self.components_[-1, comp],
                )

            # Solve quadratic problem for a given signal and update its weight
            new_weight = self.solve_quadratic_program(
                t=stretched_comps, m=signal
            )
            self.weights_[:, signal] = new_weight

    def regularize_function(self, stretch=None):
        if stretch is None:
            stretch = self.stretch_

        stretched_components, d_stretch_comps, dd_stretch_comps = (
            self.compute_stretched_components(stretch=stretch)
        )
        intermediate = stretched_components.flatten(order="F").reshape(
            (self.signal_length * self.n_signals, self.n_components), order="F"
        )
        residuals = (
            intermediate.sum(axis=1).reshape(
                (self.signal_length, self.n_signals), order="F"
            )
            - self.source_matrix
        )

        fun = self.get_objective_function(residuals, stretch)

        tiled_res = np.tile(residuals, (1, self.n_components))
        grad_flat = np.sum(d_stretch_comps * tiled_res, axis=0)
        gra = grad_flat.reshape(
            (self.n_signals, self.n_components), order="F"
        ).T
        gra += (
            self.rho
            * stretch
            @ (self._spline_smooth_operator.T @ self._spline_smooth_operator)
        )

        # Hessian would go here

        return fun, gra

    def update_stretch(self):
        """Updates stretching matrix using constrained optimization
        (equivalent to fmincon in MATLAB)."""

        # Flatten stretch for compatibility with the optimizer (since SciPy expects 1D input)
        stretch_flat_initial = self.stretch_.flatten()

        # Define the optimization function
        def objective(stretch_vec):
            stretch_matrix = stretch_vec.reshape(
                self.stretch_.shape
            )  # Reshape back to matrix form
            fun, gra = self.regularize_function(stretch_matrix)
            gra = gra.flatten()
            return fun, gra

        # Optimization constraints: lower bound 0.1, no upper bound
        bounds = [
            (0.1, None)
        ] * stretch_flat_initial.size  # Equivalent to 0.1 * ones(K, M)

        # Solve optimization problem (equivalent to fmincon)
        result = minimize(
            fun=lambda stretch_vec: objective(stretch_vec)[0],
            x0=stretch_flat_initial,
            method="trust-constr",  # Substitute for 'trust-region-reflective'
            jac=lambda stretch_vec: objective(stretch_vec)[1],  # Gradient
            bounds=bounds,
        )

        # Update stretch with the optimized values
        self.stretch_ = result.x.reshape(self.stretch_.shape)

    @staticmethod
    def _compute_objective_function(
        components, residuals, stretch, rho, eta, spline_smooth_operator
    ):
        r"""Computes the objective function used in stretched non-
        negative matrix factorization.

        Parameters
        ----------
        components : ndarray
            Non-negative matrix of component signals :math:`X`.
        residuals : ndarray
            Difference between reconstructed and observed data.
        stretch : ndarray
            Stretching factors :math:`A` applied to each component across samples.
        rho : float
            Regularization parameter enforcing smooth variation in :math:`A`.
        eta : float
            Sparsity-promoting regularization parameter applied to :math:`X`.
        spline_smooth_operator : ndarray
            Linear operator :math:`L` penalizing non-smooth changes in :math:`A`.

        Returns
        -------
        float
            Value of the stretched-NMF objective function.

        Notes
        -----
        The stretched-NMF objective function :math:`J` is

        .. math::

           J(X, Y, A) =
              \tfrac{1}{2} \lVert Z - Y\,S(A)X \rVert_F^2
              + \tfrac{\rho}{2} \lVert L A \rVert_F^2
              + \eta \sum_{i,j} \sqrt{X_{ij}} \,,

        where :math:`Z` is the data matrix, :math:`Y` contains the non-negative
        weights, :math:`S(A)` denotes the spline-interpolated stretching operator,
        and :math:`\lVert \cdot \rVert_F` is the Frobenius norm.

        Special cases
        -------------
        - :math:`\rho = 0` — no smoothness regularization on stretching factors.
        - :math:`\eta = 0` — no sparsity promotion on components.
        - :math:`\rho = \eta = 0` — reduces to the classical NMF least-squares
          objective :math:`\tfrac{1}{2} \lVert Z - YX \rVert_F^2`.
        """
        residual_term = 0.5 * np.linalg.norm(residuals, "fro") ** 2
        regularization_term = (
            0.5
            * rho
            * np.linalg.norm(spline_smooth_operator @ stretch.T, "fro") ** 2
        )
        sparsity_term = eta * np.sum(np.sqrt(components))
        return residual_term + regularization_term + sparsity_term


def cubic_largest_real_root(p, q):
    """Solves x^3 + p*x + q = 0 element-wise for matrices, returning the
    largest real root."""
    # Handle special case where q == 0
    y = np.where(
        q == 0, np.maximum(0, -p) ** 0.5, np.zeros_like(p)
    )  # q=0 case

    # Compute discriminant
    delta = (q / 2) ** 2 + (p / 3) ** 3

    # Compute square root of delta safely
    d = np.where(delta >= 0, np.sqrt(delta), np.sqrt(np.abs(delta)) * 1j)
    # TODO: this line causes a warning but results seem correct

    # Compute cube roots safely
    a1 = (-q / 2 + d) ** (1 / 3)
    a2 = (-q / 2 - d) ** (1 / 3)

    # Compute cube roots of unity
    w = (np.sqrt(3) * 1j - 1) / 2

    # Compute the three possible roots (element-wise)
    y1 = a1 + a2
    y2 = w * a1 + w**2 * a2
    y3 = w**2 * a1 + w * a2

    # Take the largest real root element-wise when delta < 0
    real_roots = np.stack([np.real(y1), np.real(y2), np.real(y3)], axis=0)
    y = np.max(real_roots, axis=0) * (
        delta < 0
    )  # Keep only real roots when delta < 0

    return y


def reconstruct_matrix(components, weights, stretch):
    """Construct the approximation of the source matrix corresponding to
    the given components, weights, and stretch factors.

    Each component profile is stretched, interpolated to fractional positions,
    weighted per signal, and summed to form the reconstruction.

    Parameters
    ----------
    components : (signal_len, n_components) array
    weights    : (n_components, n_signals) array
    stretch    : (n_components, n_signals) array

    Returns
    -------
    reconstructed_matrix : (signal_len, n_signals) array
    """

    signal_len = components.shape[0]
    n_components = components.shape[1]
    n_signals = weights.shape[1]

    reconstructed_matrix = np.zeros((signal_len, n_signals))
    sample_indices = np.arange(signal_len)

    for comp in range(n_components):  # loop over components
        reconstructed_matrix += (
            np.interp(
                sample_indices[:, None]
                / stretch[comp][
                    None, :
                ],  # fractional positions (signal_len, n_signals)
                sample_indices,  # (signal_len,)
                components[:, comp],  # component profile (signal_len,)
                left=components[0, comp],
                right=components[-1, comp],
            )
            * weights[comp][None, :]  # broadcast (n_signals,) over rows
        )

    return reconstructed_matrix
