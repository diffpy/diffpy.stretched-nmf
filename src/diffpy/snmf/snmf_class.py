import cvxpy as cp
import numpy as np
from plotter import SNMFPlotter
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, diags


class SNMFOptimizer:
    """An implementation of stretched NMF (sNMF), including sparse stretched NMF.

    Instantiating the SNMFOptimizer class runs all the analysis immediately.
    The results matrices can then be accessed as instance attributes
    of the class (components, weights, and stretch).

    For more information on sNMF, please reference:
    Gu, R., Rakita, Y., Lan, L. et al. Stretched non-negative matrix factorization.
    npj Comput Mater 10, 193 (2024). https://doi.org/10.1038/s41524-024-01377-5

    Attributes
    ----------
    source_matrix : ndarray
        The original, unmodified data to be decomposed and later, compared against.
        Shape is (length_of_signal, number_of_signals).
    stretch : ndarray
        The best guess (or while running, the current guess) for the stretching
        factor matrix.
    components : ndarray
        The best guess (or while running, the current guess) for the matrix of
        component intensities.
    weights : ndarray
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
        rho=0,
        eta=0,
        max_iter=500,
        min_iter=20,
        tol=5e-7,
        n_components=None,
        random_state=None,
        show_plots=False,
    ):
        """Initialize an instance of SNMF and run the optimization.

        Parameters
        ----------
        source_matrix : ndarray
            The data to be decomposed. Shape is (length_of_signal, number_of_conditions).
        init_weights : ndarray Optional  Default = rng.beta(a=2.5, b=1.5, size=(n_components, n_signals))
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
        rho : float Optional  Default = 0
            The stretching factor that influences the decomposition. Zero corresponds to no
            stretching present. Relatively insensitive and typically adjusted in powers of 10.
        eta : int Optional  Default = 0
            The sparsity factor that influences the decomposition. Should be set to zero for
            non-sparse data such as PDF. Can be used to improve results for sparse data such
            as XRD, but due to instability, should be used only after first selecting the
            best value for rho. Suggested adjustment is by powers of 2.
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
        self.rho = rho
        self.eta = eta
        self.tol = tol
        self.max_iter = max_iter
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
            self.weights = self._rng.beta(a=2.0, b=2.0, size=(self.n_components, self.n_signals))
        else:
            self.n_components = init_weights.shape[0]
            self.weights = init_weights

        # Initialize stretching matrix if not provided
        if init_stretch is None:
            self.stretch = np.ones((self.n_components, self.n_signals)) + self._rng.normal(
                0, 1e-3, size=(self.n_components, self.n_signals)
            )
        else:
            self.stretch = init_stretch

        # Initialize component matrix if not provided
        if init_components is None:
            self.components = self._rng.random((self.signal_length, self.n_components))
        else:
            self.components = init_components

        # Enforce non-negativity in our initial guesses
        self.components = np.maximum(0, self.components)
        self.weights = np.maximum(0, self.weights)

        # Second-order spline: Tridiagonal (-2 on diagonal, 1 on sub/superdiagonals)
        self._spline_smooth_operator = 0.25 * diags(
            [1, -2, 1], offsets=[0, 1, 2], shape=(self.n_signals - 2, self.n_signals)
        )

        # Set up residual matrix, objective function, and history
        self.residuals = self.get_residual_matrix()
        self.objective_function = self.get_objective_function()
        self.best_objective = self.objective_function
        self.best_matrices = [self.components.copy(), self.weights.copy(), self.stretch.copy()]
        self.objective_difference = None
        self._objective_history = [self.objective_function]

        # Set up tracking variables for update_components()
        self._prev_components = None
        self._grad_components = np.zeros_like(self.components)
        self._prev_grad_components = np.zeros_like(self.components)

        regularization_term = 0.5 * rho * np.linalg.norm(self._spline_smooth_operator @ self.stretch.T, "fro") ** 2
        sparsity_term = eta * np.sum(np.sqrt(self.components))  # Square root penalty
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
                0.5 * rho * np.linalg.norm(self._spline_smooth_operator @ self.stretch.T, "fro") ** 2
            )
            sparsity_term = eta * np.sum(np.sqrt(self.components))  # Square root penalty
            print(
                f"Obj fun: {self.objective_function:.5e}, "
                f"Obj - reg/sparse: {self.objective_function - regularization_term - sparsity_term:.5e}, "
                f"Iter: {self.outiter}"
            )

            # Convergence check: Stop if diffun is small and at least min_iter iterations have passed
            print("Checking if ", self.objective_difference, " < ", self.objective_function * tol)
            if self.objective_difference < self.objective_function * tol and outiter >= min_iter:
                break

        self.normalize_results()

    def normalize_results(self):
        # Select our best results for normalization
        self.components = self.best_matrices[0]
        self.weights = self.best_matrices[1]
        self.stretch = self.best_matrices[2]

        # Normalize weights/stretch first
        weights_row_max = np.max(self.weights, axis=1, keepdims=True)
        self.weights = self.weights / weights_row_max
        stretch_row_max = np.max(self.stretch, axis=1, keepdims=True)
        self.stretch = self.stretch / stretch_row_max

        # effectively just re-running with component updates only vs normalized weights/stretch
        self._grad_components = np.zeros_like(self.components)  # Gradient of X (zeros for now)
        self._prev_grad_components = np.zeros_like(self.components)  # Previous gradient of X (zeros for now)
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
            print(f"Objective function after normalize_components: {self.objective_function:.5e}")
            self._objective_history.append(self.objective_function)
            self.objective_difference = self._objective_history[-2] - self._objective_history[-1]
            if self.plotter is not None:
                self.plotter.update(
                    components=self.components,
                    weights=self.weights,
                    stretch=self.stretch,
                    update_tag="normalize components",
                )
            if self.objective_difference < self.objective_function * self.tol and outiter >= 7:
                break

    def outer_loop(self):
        for iter in range(4):
            self.iter = iter
            self._prev_grad_components = self._grad_components.copy()
            self.update_components()
            self.residuals = self.get_residual_matrix()
            self.objective_function = self.get_objective_function()
            print(f"Objective function after update_components: {self.objective_function:.5e}")
            self._objective_history.append(self.objective_function)
            self.objective_difference = self._objective_history[-2] - self._objective_history[-1]
            if self.objective_function < self.best_objective:
                self.best_objective = self.objective_function
                self.best_matrices = [self.components.copy(), self.weights.copy(), self.stretch.copy()]
            if self.plotter is not None:
                self.plotter.update(
                    components=self.components, weights=self.weights, stretch=self.stretch, update_tag="components"
                )

            self.update_weights()
            self.residuals = self.get_residual_matrix()
            self.objective_function = self.get_objective_function()
            print(f"Objective function after update_weights: {self.objective_function:.5e}")
            self._objective_history.append(self.objective_function)
            self.objective_difference = self._objective_history[-2] - self._objective_history[-1]
            if self.objective_function < self.best_objective:
                self.best_objective = self.objective_function
                self.best_matrices = [self.components.copy(), self.weights.copy(), self.stretch.copy()]
            if self.plotter is not None:
                self.plotter.update(
                    components=self.components, weights=self.weights, stretch=self.stretch, update_tag="weights"
                )

            self.objective_difference = self._objective_history[-2] - self._objective_history[-1]
            if self._objective_history[-3] - self.objective_function < self.objective_difference * 1e-3:
                break

        self.update_stretch()
        self.residuals = self.get_residual_matrix()
        self.objective_function = self.get_objective_function()
        print(f"Objective function after update_stretch: {self.objective_function:.5e}")
        self._objective_history.append(self.objective_function)
        self.objective_difference = self._objective_history[-2] - self._objective_history[-1]
        if self.objective_function < self.best_objective:
            self.best_objective = self.objective_function
            self.best_matrices = [self.components.copy(), self.weights.copy(), self.stretch.copy()]
        if self.plotter is not None:
            self.plotter.update(
                components=self.components, weights=self.weights, stretch=self.stretch, update_tag="stretch"
            )

    def get_residual_matrix(self, components=None, weights=None, stretch=None):
        # Initialize residual matrix as negative of source_matrix
        if components is None:
            components = self.components
        if weights is None:
            weights = self.weights
        if stretch is None:
            stretch = self.stretch
        residuals = -self.source_matrix.copy()
        # Compute transformed components for all (k, m) pairs
        for k in range(weights.shape[0]):  # K
            stretched_components, _, _ = apply_interpolation(stretch[k, :], components[:, k])  # Only use Ax
            residuals += weights[k, :] * stretched_components  # Element-wise scaling and sum
        return residuals

    def get_objective_function(self, residuals=None, stretch=None):
        if residuals is None:
            residuals = self.residuals
        if stretch is None:
            stretch = self.stretch
        residual_term = 0.5 * np.linalg.norm(residuals, "fro") ** 2
        regularization_term = 0.5 * self.rho * np.linalg.norm(self._spline_smooth_operator @ stretch.T, "fro") ** 2
        sparsity_term = self.eta * np.sum(np.sqrt(self.components))  # Square root penalty
        # Final objective function value
        function = residual_term + regularization_term + sparsity_term
        return function

    def apply_interpolation_matrix(self, components=None, weights=None, stretch=None):
        """
        Applies an interpolation-based transformation to the 'components' using `stretch`,
        weighted by `weights`. Optionally computes first (`d_stretched_components`) and
        second (`dd_stretched_components`) derivatives.
        """

        if components is None:
            components = self.components
        if weights is None:
            weights = self.weights
        if stretch is None:
            stretch = self.stretch

        # Compute scaled indices
        stretch_flat = stretch.reshape(1, self.n_signals * self.n_components) ** -1
        stretch_tiled = np.tile(stretch_flat, (self.signal_length, 1))

        # Compute `fractional_indices`
        fractional_indices = (
            np.tile(np.arange(self.signal_length)[:, None], (1, self.n_signals * self.n_components))
            * stretch_tiled
        )

        # Weighting matrix
        weights_flat = weights.reshape(1, self.n_signals * self.n_components)
        weights_tiled = np.tile(weights_flat, (self.signal_length, 1))

        # Bias for indexing into reshaped components
        # TODO break this up or describe what it does better
        bias = np.kron(
            np.arange(self.n_components) * (self.signal_length + 1),
            np.ones((self.signal_length, self.n_signals), dtype=int),
        ).reshape(self.signal_length, self.n_components * self.n_signals)

        # Handle boundary conditions for interpolation
        components_bounded = np.vstack(
            [components, components[-1, :]]
        )  # Duplicate last row (like MATLAB, not sure why)

        # Compute floor indices
        floor_indices = np.floor(fractional_indices).astype(int)

        floor_indices_1 = np.minimum(floor_indices + 1, self.signal_length)
        floor_indices_2 = np.minimum(floor_indices_1 + 1, self.signal_length)

        # Compute fractional part
        fractional_floor_indices = fractional_indices - floor_indices

        # Compute offset indices
        offset_indices_1 = floor_indices_1 + bias
        offset_indices_2 = floor_indices_2 + bias

        # Extract values
        # Note: this "-1" corrects an off-by-one error that may have originated in an earlier line
        comp_values_1 = components_bounded.flatten(order="F")[(offset_indices_1 - 1).ravel(order="F")].reshape(
            self.signal_length, self.n_components * self.n_signals, order="F"
        )  # order = F uses FORTRAN, column major order
        comp_values_2 = components_bounded.flatten(order="F")[(offset_indices_2 - 1).ravel(order="F")].reshape(
            self.signal_length, self.n_components * self.n_signals, order="F"
        )

        # Interpolation
        unweighted_stretched_comps = (
            comp_values_1 * (1 - fractional_floor_indices) + comp_values_2 * fractional_floor_indices
        )
        stretched_components = unweighted_stretched_comps * weights_tiled  # Apply weighting

        # Compute first derivative
        di = -fractional_indices * stretch_tiled
        d_comps_unweighted = comp_values_1 * (-di) + comp_values_2 * di
        d_stretched_components = d_comps_unweighted * weights_tiled

        # Compute second derivative
        ddi = -di * stretch_tiled * 2
        dd_comps_unweighted = comp_values_1 * (-ddi) + comp_values_2 * ddi
        dd_stretched_components = dd_comps_unweighted * weights_tiled

        return stretched_components, d_stretched_components, dd_stretched_components

    def apply_transformation_matrix(self, stretch=None, weights=None, residuals=None):
        """
        Computes the transformation matrix `stretch_transformed` for residuals,
        using scaling matrix `stretch` and weight coefficients `weights`.
        """

        if stretch is None:
            stretch = self.stretch
        if weights is None:
            weights = self.weights
        if residuals is None:
            residuals = self.residuals

        # Compute scaling matrix
        stretch_tiled = np.tile(
            stretch.reshape(1, self.n_signals * self.n_components, order="F") ** -1, (self.signal_length, 1)
        )

        # Compute indices
        indices = np.arange(self.signal_length)[:, None] * stretch_tiled

        # Weighting coefficients
        weights_tiled = np.tile(
            weights.reshape(1, self.n_signals * self.n_components, order="F"), (self.signal_length, 1)
        )

        # Compute floor indices
        floor_indices = np.floor(indices).astype(int)
        floor_indices_1 = np.minimum(floor_indices + 1, self.signal_length)
        floor_indices_2 = np.minimum(floor_indices_1 + 1, self.signal_length)

        # Compute fractional part
        fractional_indices = indices - floor_indices

        # Expand row indices
        repm = np.tile(np.arange(self.n_components), (self.signal_length, self.n_signals))

        # Compute transformations
        kron = np.kron(residuals, np.ones((1, self.n_components)))
        fractional_kron = kron * fractional_indices
        fractional_weights = (fractional_indices - 1) * weights_tiled

        # Construct sparse matrices
        x2 = coo_matrix(
            ((-kron * fractional_weights).flatten(), (floor_indices_1.flatten() - 1, repm.flatten())),
            shape=(self.signal_length + 1, self.n_components),
        ).tocsc()
        x3 = coo_matrix(
            ((fractional_kron * weights_tiled).flatten(), (floor_indices_2.flatten() - 1, repm.flatten())),
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
        reg_factor = 1e-8 * np.linalg.norm(q, ord="fro")  # Adaptive regularization, original was fixed
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
        return np.maximum(y.value, 0)  # Ensure non-negative values in case of solver tolerance issues

    def update_components(self):
        """
        Updates `components` using gradient-based optimization with adaptive step size.
        """
        # Compute stretched components using the interpolation function
        stretched_components, _, _ = self.apply_interpolation_matrix()  # Discard the derivatives
        # Compute reshaped_stretched_components and component_residuals
        intermediate_reshaped = stretched_components.flatten(order="F").reshape(
            (self.signal_length * self.n_signals, self.n_components), order="F"
        )
        reshaped_stretched_components = intermediate_reshaped.sum(axis=1).reshape(
            (self.signal_length, self.n_signals), order="F"
        )
        component_residuals = reshaped_stretched_components - self.source_matrix
        # Compute gradient
        self._grad_components = self.apply_transformation_matrix(
            residuals=component_residuals
        ).toarray()  # toarray equivalent of full, make non-sparse

        # Compute initial step size `initial_step_size`
        initial_step_size = np.linalg.eigvalsh(self.weights.T @ self.weights).max() * np.max(
            [self.stretch.max(), 1 / self.stretch.min()]
        )
        # Compute adaptive step size `step_size`
        if self.outiter == 0 and self.iter == 0:
            step_size = initial_step_size
        else:
            num = np.sum(
                (self._grad_components - self._prev_grad_components) * (self.components - self._prev_components)
            )  # Element-wise multiplication
            denom = np.linalg.norm(self.components - self._prev_components, "fro") ** 2  # Frobenius norm squared
            step_size = num / denom if denom > 0 else initial_step_size
            if step_size <= 0:
                step_size = initial_step_size

        # Store our old X before updating because it is used in step selection
        self._prev_components = self.components.copy()

        while True:  # iterate updating components
            components_step = self._prev_components - self._grad_components / step_size
            # Solve x^3 + p*x + q = 0 for the largest real root
            self.components = np.square(cubic_largest_real_root(-components_step, self.eta / (2 * step_size)))
            # Mask values that should be set to zero
            mask = (
                self.components**2 * step_size / 2
                - step_size * self.components * components_step
                + self.eta * np.sqrt(self.components)
                < 0
            )
            self.components = mask * self.components

            objective_improvement = self._objective_history[-1] - self.get_objective_function(
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
        """
        Updates weights using matrix operations, solving a quadratic program to do so.
        """

        signal_length = self.signal_length
        n_signals = self.n_signals

        for m in range(n_signals):
            t = np.zeros((signal_length, self.n_components))

            # Populate t using apply_interpolation
            for k in range(self.n_components):
                t[:, k] = apply_interpolation(self.stretch[k, m], self.components[:, k])[0].squeeze()

            # Solve quadratic problem for y
            y = self.solve_quadratic_program(t=t, m=m)

            # Update Y
            self.weights[:, m] = y

    def regularize_function(self, stretch=None):
        if stretch is None:
            stretch = self.stretch

        K = self.n_components
        M = self.n_signals
        N = self.signal_length

        stretched_components, d_stretch_comps, dd_stretch_comps = self.apply_interpolation_matrix(stretch=stretch)
        intermediate = stretched_components.flatten(order="F").reshape((N * M, K), order="F")
        residuals = intermediate.sum(axis=1).reshape((N, M), order="F") - self.source_matrix

        fun = self.get_objective_function(residuals, stretch)

        tiled_res = np.tile(residuals, (1, K))
        grad_flat = np.sum(d_stretch_comps * tiled_res, axis=0)
        gra = grad_flat.reshape((M, K), order="F").T
        gra += self.rho * stretch @ (self._spline_smooth_operator.T @ self._spline_smooth_operator)

        # Hessian would go here

        return fun, gra

    def update_stretch(self):
        """
        Updates matrix A using constrained optimization (equivalent to fmincon in MATLAB).
        """

        # Flatten A for compatibility with the optimizer (since SciPy expects 1D input)
        stretch_flat_initial = self.stretch.flatten()

        # Define the optimization function
        def objective(stretch_vec):
            stretch_matrix = stretch_vec.reshape(self.stretch.shape)  # Reshape back to matrix form
            fun, gra = self.regularize_function(stretch_matrix)
            gra = gra.flatten()
            return fun, gra

        # Optimization constraints: lower bound 0.1, no upper bound
        bounds = [(0.1, None)] * stretch_flat_initial.size  # Equivalent to 0.1 * ones(K, M)

        # Solve optimization problem (equivalent to fmincon)
        result = minimize(
            fun=lambda stretch_vec: objective(stretch_vec)[0],
            x0=stretch_flat_initial,
            method="trust-constr",  # Substitute for 'trust-region-reflective'
            jac=lambda stretch_vec: objective(stretch_vec)[1],  # Gradient
            bounds=bounds,
        )

        # Update A with the optimized values
        self.stretch = result.x.reshape(self.stretch.shape)


def cubic_largest_real_root(p, q):
    """
    Solves x^3 + p*x + q = 0 element-wise for matrices, returning the largest real root.
    """
    # Handle special case where q == 0
    y = np.where(q == 0, np.maximum(0, -p) ** 0.5, np.zeros_like(p))  # q=0 case

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
    y = np.max(real_roots, axis=0) * (delta < 0)  # Keep only real roots when delta < 0

    return y


def apply_interpolation(a, x):
    """
    Applies an interpolation-based transformation to `x` based on scaling `a`.
    Also computes first (`d_intr_x`) and second (`dd_intr_x`) derivatives.
    """
    x_len = len(x)

    # Ensure `a` is an array and reshape for broadcasting
    a = np.atleast_1d(np.asarray(a))  # Ensures a is at least 1D

    # Compute fractional indices, broadcasting over `a`
    fractional_indices = np.arange(x_len)[:, None] / a  # Shape (N, M)

    integer_indices = np.floor(fractional_indices).astype(int)  # Integer part (still (N, M))
    valid_mask = integer_indices < (x_len - 1)  # Ensure indices are within bounds

    # Apply valid_mask to keep correct indices
    idx_int = np.where(valid_mask, integer_indices, x_len - 2)  # Prevent out-of-bounds indexing (previously "I")
    idx_frac = np.where(valid_mask, fractional_indices, integer_indices)  # Keep aligned (previously "i")

    # Ensure x is a 1D array
    x = np.asarray(x).ravel()

    # Compute interpolated_x (linear interpolation)
    interpolated_x = x[idx_int] * (1 - idx_frac + idx_int) + x[np.minimum(idx_int + 1, x_len - 1)] * (
        idx_frac - idx_int
    )

    # Fill the tail with the last valid value
    intr_x_tail = np.full((x_len - len(idx_int), interpolated_x.shape[1]), interpolated_x[-1, :])
    interpolated_x = np.vstack([interpolated_x, intr_x_tail])

    # Compute first derivative (d_intr_x)
    di = -idx_frac / a
    d_intr_x = x[idx_int] * (-di) + x[np.minimum(idx_int + 1, x_len - 1)] * di
    d_intr_x = np.vstack([d_intr_x, np.zeros((x_len - len(idx_int), d_intr_x.shape[1]))])

    # Compute second derivative (dd_intr_x)
    ddi = -di / a + idx_frac * a**-2
    dd_intr_x = x[idx_int] * (-ddi) + x[np.minimum(idx_int + 1, x_len - 1)] * ddi
    dd_intr_x = np.vstack([dd_intr_x, np.zeros((x_len - len(idx_int), dd_intr_x.shape[1]))])

    return interpolated_x, d_intr_x, dd_intr_x
