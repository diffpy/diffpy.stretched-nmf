import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, csc_matrix, diags


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
    tol : float
        The convergence threshold. This is the minimum fractional improvement in the
        objective function to allow without terminating the optimization. Note that
        a minimum of 20 updates are run before this parameter is checked.
    n_components : int
        The number of components to extract from source_matrix. Must be provided when and only when
        Y0 is not provided.
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
        tol=5e-7,
        n_components=None,
        random_state=None,
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
        """

        self.source_matrix = source_matrix
        self.rho = rho
        self.eta = eta
        # Capture matrix dimensions
        self.signal_length, self.n_signals = source_matrix.shape
        self.num_updates = 0
        self._rng = np.random.default_rng(random_state)

        # Enforce exclusive specification of n_components or Y0
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
        self._spline_smooth_penalty = self._spline_smooth_operator.T @ self._spline_smooth_operator

        # Set up residual matrix, objective function, and history
        self.residuals = self.get_residual_matrix()
        self._objective_history = []
        self.update_objective()
        self.objective_difference = None

        # Set up tracking variables for update_components()
        self._prev_components = None
        self.grad_components = np.zeros_like(self.components)  # Gradient of X (zeros for now)
        self._prev_grad_components = np.zeros_like(self.components)  # Previous gradient of X (zeros for now)

        regularization_term = 0.5 * rho * np.linalg.norm(self._spline_smooth_operator @ self.stretch.T, "fro") ** 2
        sparsity_term = eta * np.sum(np.sqrt(self.components))  # Square root penalty
        print(
            f"Start, Objective function: {self._objective_history[-1]:.5e}"
            f", Obj - reg/sparse: {self._objective_history[-1] - regularization_term - sparsity_term:.5e}"
        )

        # Main optimization loop
        for iter in range(max_iter):
            self.optimize_loop()
            # Print diagnostics
            regularization_term = (
                0.5 * rho * np.linalg.norm(self._spline_smooth_operator @ self.stretch.T, "fro") ** 2
            )
            sparsity_term = eta * np.sum(np.sqrt(self.components))  # Square root penalty
            print(
                f"Num_updates: {self.num_updates}, "
                f"Obj fun: {self._objective_history[-1]:.5e}, "
                f"Obj - reg/sparse: {self._objective_history[-1] - regularization_term - sparsity_term:.5e}, "
                f"Iter: {iter}"
            )

            # Convergence check: decide when to terminate for small/no improvement
            if self.objective_difference < self._objective_history[-1] * tol and iter >= 20:
                break
            print(self.objective_difference, " < ", self._objective_history[-1] * tol)

        # Normalize our results
        weights_row_max = np.max(self.weights, axis=1, keepdims=True)
        stretch_row_max = np.max(self.stretch, axis=1, keepdims=True)
        self.weights = self.weights / weights_row_max
        self.stretch = self.stretch / stretch_row_max

        # loop to normalize components
        # effectively just re-running class with non-normalized components, normalized wts/stretch as inputs,
        # then only update components
        self._prev_components = None
        self.grad_components = np.zeros_like(self.components)
        self._prev_grad_components = np.zeros_like(self.components)
        self.residuals = self.get_residual_matrix()
        self.objective_difference = None
        self._objective_history = []
        self.update_objective()
        for norm_iter in range(100):
            self.update_components()
            self.residuals = self.get_residual_matrix()
            self.update_objective()
            print(f"Objective function after normalize_components: {self._objective_history[-1]:.5e}")
            self._objective_history.append(self._objective_history[-1])
            self.objective_difference = self._objective_history[-2] - self._objective_history[-1]
            if self.objective_difference < self._objective_history[-1] * tol and norm_iter >= 20:
                break
        # end of normalization (and program)
        # note that objective function may not fully recover after normalization, this is okay
        print("Finished optimization.")

    def optimize_loop(self):
        # Update components first
        self._prev_grad_components = self.grad_components.copy()

        self.update_components()

        self.num_updates += 1
        self.residuals = self.get_residual_matrix()
        self.update_objective()
        print(f"Objective function after update_components: {self._objective_history[-1]:.5e}")

        if self.objective_difference is None:
            self.objective_difference = self._objective_history[-2] - self._objective_history[-1]

        # Now we update weights
        self.update_weights()

        self.num_updates += 1
        self.residuals = self.get_residual_matrix()
        self.update_objective()
        print(f"Objective function after update_weights: {self._objective_history[-1]:.5e}")

        # Now we update stretch
        self.update_stretch()

        self.num_updates += 1
        self.residuals = self.get_residual_matrix()
        self.update_objective()
        print(f"Objective function after update_stretch: {self._objective_history[-1]:.5e}")

        self.objective_difference = self._objective_history[-2] - self._objective_history[-1]

    def apply_interpolation(self, a, x, return_derivatives=False):
        """
        Applies an interpolation-based transformation to `x` based on scaling `a`.
        Also can compute first (`d_intr_x`) and second (`dd_intr_x`) derivatives.
        """
        x_len = len(x)

        # Ensure `a` is an array and reshape for broadcasting
        a = np.atleast_1d(np.asarray(a))  # Ensures a is at least 1D

        # Compute fractional indices, broadcasting over `a`
        fractional_indices = np.arange(x_len)[:, None] / a  # Shape (N, M)

        integer_indices = np.floor(fractional_indices).astype(int)  # Integer part (still (N, M))
        valid_mask = integer_indices < (x_len - 1)  # Ensure indices are within bounds

        # Apply valid_mask to keep correct indices
        idx_int = np.where(
            valid_mask, integer_indices, x_len - 2
        )  # Prevent out-of-bounds indexing (previously "I")
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

        if return_derivatives:
            # Compute first derivative (d_intr_x)
            di = -idx_frac / a
            d_intr_x = x[idx_int] * (-di) + x[np.minimum(idx_int + 1, x_len - 1)] * di
            d_intr_x = np.vstack([d_intr_x, np.zeros((x_len - len(idx_int), d_intr_x.shape[1]))])

            # Compute second derivative (dd_intr_x)
            ddi = -di / a + idx_frac * a**-2
            dd_intr_x = x[idx_int] * (-ddi) + x[np.minimum(idx_int + 1, x_len - 1)] * ddi
            dd_intr_x = np.vstack([dd_intr_x, np.zeros((x_len - len(idx_int), dd_intr_x.shape[1]))])
        else:
            # Make placeholders
            d_intr_x = np.empty(interpolated_x.shape)
            dd_intr_x = np.empty(interpolated_x.shape)

        return interpolated_x, d_intr_x, dd_intr_x

    def get_residual_matrix(self, components=None, weights=None, stretch=None):
        # Initialize residual matrix as negative of source_matrix
        # In MATLAB this is getR
        if components is None:
            components = self.components
        if weights is None:
            weights = self.weights
        if stretch is None:
            stretch = self.stretch
        residuals = -self.source_matrix.copy()
        # Compute transformed components for all (k, m) pairs
        for k in range(weights.shape[0]):
            stretched_components, _, _ = self.apply_interpolation(
                stretch[k, :], components[:, k]
            )  # Only calculate Ax
            residuals += weights[k, :] * stretched_components  # Element-wise scaling and sum
        return residuals

    def update_objective(self, residuals=None, stretch=None):
        to_return = not (residuals is None and stretch is None)
        if residuals is None:
            residuals = self.residuals
        if stretch is None:
            stretch = self.stretch
        residual_term = 0.5 * np.linalg.norm(residuals, "fro") ** 2
        regularization_term = 0.5 * self.rho * np.linalg.norm(self._spline_smooth_operator @ stretch.T, "fro") ** 2
        sparsity_term = self.eta * np.sum(np.sqrt(self.components))  # Square root penalty
        # Final objective function value
        function = residual_term + regularization_term + sparsity_term

        if to_return:
            return function  # Get value directly for use
        else:
            self._objective_history.append(function)  # Store value

    def apply_interpolation_matrix(self, components=None, weights=None, stretch=None, return_derivatives=False):
        """
        Applies an interpolation-based transformation to the matrix `components` using `stretch`,
        weighted by `weights`. Optionally computes first and second derivatives.
        Equivalent to getAfun_matrix in MATLAB.
        """

        if components is None:
            components = self.components
        if weights is None:
            weights = self.weights
        if stretch is None:
            stretch = self.stretch

        # Compute scaled indices (MATLAB: AA = repmat(reshape(A',1,M*K).^-1, N,1))
        stretch_flat = stretch.reshape(1, self.n_signals * self.n_components) ** -1
        stretch_tiled = np.tile(stretch_flat, (self.signal_length, 1))

        # Compute `ii` (MATLAB: ii = repmat((0:N-1)',1,K*M).*tiled_stretch)
        fractional_indices = (
            np.tile(np.arange(self.signal_length)[:, None], (1, self.n_signals * self.n_components))
            * stretch_tiled
        )

        # Weighting matrix (MATLAB: YY = repmat(reshape(Y',1,M*K), N,1))
        weights_flat = weights.reshape(1, self.n_signals * self.n_components)
        weights_tiled = np.tile(weights_flat, (self.signal_length, 1))

        # Bias for indexing into reshaped X (MATLAB: bias = kron((0:K-1)*(N+1),ones(N,M)))
        # TODO break this up or describe what it does better
        bias = np.kron(
            np.arange(self.n_components) * (self.signal_length + 1),
            np.ones((self.signal_length, self.n_signals), dtype=int),
        ).reshape(self.signal_length, self.n_components * self.n_signals)

        # Handle boundary conditions for interpolation (MATLAB: X1=[X;X(end,:)])
        components_bounded = np.vstack([components, components[-1, :]])  # Duplicate last row (like MATLAB)

        # Compute floor indices (MATLAB: II = floor(ii); II1=min(II+1,N+1); II2=min(II1+1,N+1))
        floor_indices = np.floor(fractional_indices).astype(int)

        floor_ind_1 = np.minimum(floor_indices + 1, self.signal_length)
        floor_ind_2 = np.minimum(floor_ind_1 + 1, self.signal_length)

        # Compute fractional part (MATLAB: iI = ii - II)
        fractional_floor_indices = fractional_indices - floor_indices

        # Compute offset indices (MATLAB: II1_ = II1 + bias; II2_ = II2 + bias)
        offset_floor_ind_1 = floor_ind_1 + bias
        offset_floor_ind_2 = floor_ind_2 + bias

        # Extract values (MATLAB: XI1 = reshape(X1(II1_), N, K*M); XI2 = reshape(X1(II2_), N, K*M))
        # Note: this "-1" corrects an off-by-one error that may have originated in an earlier line
        # order = F uses FORTRAN, column major order
        components_val_1 = components_bounded.flatten(order="F")[(offset_floor_ind_1 - 1).ravel()].reshape(
            self.signal_length, self.n_components * self.n_signals
        )
        components_val_2 = components_bounded.flatten(order="F")[(offset_floor_ind_2 - 1).ravel()].reshape(
            self.signal_length, self.n_components * self.n_signals
        )

        # Interpolation (MATLAB: Ax2=XI1.*(1-iI)+XI2.*(iI); stretched_components=Ax2.*YY)
        stretch_components2 = (
            components_val_1 * (1 - fractional_floor_indices) + components_val_2 * fractional_floor_indices
        )
        stretched_components = stretch_components2 * weights_tiled  # Apply weighting

        if return_derivatives:
            # Compute first derivative (MATLAB: Tx2=XI1.*(-di)+XI2.*di; d_str_cmps=Tx2.*YY)
            di = -fractional_indices * stretch_tiled
            d_components2 = components_val_1 * (-di) + components_val_2 * di
            d_stretch_components = d_components2 * weights_tiled

            # Compute second derivative (MATLAB: Hx2=XI1.*(-ddi)+XI2.*ddi; dd_str_components=Hx2.*YY)
            ddi = -di * stretch_tiled * 2
            dd_components2 = components_val_1 * (-ddi) + components_val_2 * ddi
            dd_stretch_components = dd_components2 * weights_tiled
        else:
            shape = stretched_components.shape
            d_stretch_components = np.empty(shape)
            dd_stretch_components = np.empty(shape)

        return stretched_components, d_stretch_components, dd_stretch_components

    def apply_transformation_matrix(self, stretch=None, weights=None, residuals=None):
        """
        Computes the transformation matrix `stretch_transformed` for `residuals`,
        using scaling matrix `stretch` and coefficients `weights`.
        """

        if stretch is None:
            stretch = self.stretch
        if weights is None:
            weights = self.weights
        if residuals is None:
            residuals = self.residuals

        # Compute scaling matrix (MATLAB: AA = repmat(reshape(A,1,M*K).^-1,Nindex,1))
        stretch_tiled = np.tile(
            stretch.reshape(1, self.n_signals * self.n_components, order="F") ** -1, (self.signal_length, 1)
        )

        # Compute indices (MATLAB: ii = repmat((index-1)',1,K*M).*AA)
        indices = np.arange(self.signal_length)[:, None] * stretch_tiled  # Shape (N, M*K), replacing `index`

        # Weighting coefficients (MATLAB: YY = repmat(reshape(Y,1,M*K),Nindex,1))
        weights_tiled = np.tile(
            weights.reshape(1, self.n_signals * self.n_components, order="F"), (self.signal_length, 1)
        )

        # Compute floor indices (MATLAB: II = floor(ii); II1 = min(II+1,N+1); II2 = min(II1+1,N+1))
        floor_indices = np.floor(indices).astype(int)
        floor_indices_1 = np.minimum(floor_indices + 1, self.signal_length)
        floor_indices_2 = np.minimum(floor_indices_1 + 1, self.signal_length)

        # Compute fractional part (MATLAB: iI = ii - II)
        fractional_indices = indices - floor_indices

        # Expand row indices (MATLAB: repm = repmat(1:K, Nindex, M))
        repm = np.tile(np.arange(self.n_components), (self.signal_length, self.n_signals))

        # Compute transformations (MATLAB: kro = kron(R(index,:), ones(1, K)))
        kron = np.kron(residuals, np.ones((1, self.n_components)))

        # (MATLAB: kroiI = kro .* (iI); iIYY = (iI-1) .* YY)
        fractional_kron = kron * fractional_indices
        fractional_weights = (fractional_indices - 1) * weights_tiled

        # Construct sparse matrices (MATLAB: sparse(II1_,repm,kro.*-iIYY,(N+1),K))
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

    def solve_quadratic_program(self, t, m, alg="trust-constr"):
        """
        Solves the quadratic program for updating y in stretched NMF using scipy.optimize:

            min J(y) = 0.5 * y^T Q y + d^T y
            subject to: 0 ≤ y ≤ 1

        Uses the 'trust-constr' solver with the analytical gradient and Hessian.
        Alternatively, can use scipy's L-BFGS-B algorithm, which supports bound
        constraints.

        Parameters:
        - t: (N, K) ndarray
            Matrix computed from getAfun(A(k, m), X[:, k]).
        - m: int
            Index of the current column in source_matrix.

        Returns:
        - y: (k,) ndarray
            Optimal solution for y, clipped to ensure non-negativity.
        """
        source_matrix_col = self.source_matrix[:, m]
        q = t.T @ t
        d = -t.T @ source_matrix_col
        k = q.shape[0]
        reg_factor = 1e-8 * np.linalg.norm(q, ord="fro")
        q += np.eye(k) * reg_factor

        def objective(y):
            return 0.5 * y @ q @ y + d @ y

        def grad(y):
            return q @ y + d

        if alg == "trust-constr":

            def hess(y):
                return csc_matrix(q)  # sparse format for efficiency

            bounds = [(0, 1)] * k
            y0 = np.clip(-np.linalg.solve(q + np.eye(k) * 1e-5, d), 0, 1)
            result = minimize(
                objective, y0, method="trust-constr", jac=grad, hess=hess, bounds=bounds, options={"verbose": 0}
            )
        elif alg == "L-BFGS-B":
            bounds = [(0, 1) for _ in range(k)]  # per-variable bounds
            y0 = np.clip(-np.linalg.solve(q + np.eye(k) * 1e-5, d), 0, 1)  # Initial guess
            result = minimize(objective, y0, method="L-BFGS-B", jac=grad, bounds=bounds)

        return np.maximum(result.x, 0)

    def update_components(self):
        """
        Updates `components` using gradient-based optimization with adaptive step size step_size.
        """
        # Compute `stretched_components` using the interpolation function
        stretched_components, _, _ = self.apply_interpolation_matrix()  # Skip the other two outputs (derivatives)
        # Compute RA and RR
        intermediate_reshaped = stretched_components.flatten(order="F").reshape(
            (self.signal_length * self.n_signals, self.n_components), order="F"
        )
        reshaped_stretched_components = intermediate_reshaped.sum(axis=1).reshape(
            (self.signal_length, self.n_signals), order="F"
        )
        component_residuals = reshaped_stretched_components - self.source_matrix
        # Compute gradient `GraX`
        self.grad_components = self.apply_transformation_matrix(
            residuals=component_residuals
        ).toarray()  # toarray equivalent of MATLAB "full", makes non-sparse

        # Compute initial step size `initial_step_size`
        initial_step_size = np.linalg.eigvalsh(self.weights.T @ self.weights).max() * np.max(
            [self.stretch.max(), 1 / self.stretch.min()]
        )
        # Compute adaptive step size `step_size`
        if self._prev_components is None:
            step_size = initial_step_size
        else:
            num = np.sum(
                (self.grad_components - self._prev_grad_components) * (self.components - self._prev_components)
            )  # Elem-wise multiply
            denom = np.linalg.norm(self.components - self._prev_components, "fro") ** 2  # Frobenius norm squared
            step_size = num / denom if denom > 0 else initial_step_size
            if step_size <= 0:
                step_size = initial_step_size

        # Store our old component matrix before updating because it is used in step selection
        self._prev_components = self.components.copy()

        while True:  # iterate updating components
            components_step = self._prev_components - self.grad_components / step_size
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

            objective_improvement = self._objective_history[-1] - self.update_objective(
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
        Updates weights using matrix operations, solving a quadratic program via to do so.
        """

        for m in range(self.n_signals):
            t = np.zeros((self.signal_length, self.n_components))

            # Populate T using apply_interpolation
            for k in range(self.n_components):
                t[:, k] = self.apply_interpolation(
                    self.stretch[k, m], self.components[:, k], return_derivatives=True
                )[0].squeeze()

            # Solve quadratic problem for y
            y = self.solve_quadratic_program(t=t, m=m)

            # Update Y
            self.weights[:, m] = y

    def regularize_function(self, stretch=None):
        """
        Computes the regularization function, gradient, and Hessian for optimization.
        Returns:
        - fun: Objective function value (scalar)
        - gra: Gradient (same shape as stretch)
        """
        if stretch is None:
            stretch = self.stretch

        # Compute interpolated matrices
        stretched_components, d_stretch_components, dd_stretch_components = self.apply_interpolation_matrix(
            stretch=stretch, return_derivatives=True
        )

        # Compute residual
        intermediate_diff = stretched_components.flatten(order="F").reshape(
            (self.signal_length * self.n_signals, self.n_components), order="F"
        )
        stretch_difference = intermediate_diff.sum(axis=1).reshape((self.signal_length, self.n_signals), order="F")
        stretch_difference = stretch_difference - self.source_matrix

        # Compute objective function
        reg_func = self.update_objective(stretch_difference, stretch)

        # Compute gradient
        tiled_derivative = np.sum(
            d_stretch_components * np.tile(stretch_difference, (1, self.n_components)), axis=0
        )
        der_reshaped = np.asarray(tiled_derivative).reshape((self.n_signals, self.n_components), order="F")
        func_grad = (
            der_reshaped.T + self.rho * stretch @ self._spline_smooth_operator.T @ self._spline_smooth_operator
        )

        return reg_func, func_grad

    def update_stretch(self):
        """
        Updates stretching matrix using constrained optimization (equivalent to fmincon in MATLAB).
        """

        # Flatten stretch for compatibility with the optimizer (since SciPy expects 1D input)
        stretch_init_vec = self.stretch.flatten()

        # Define the optimization function
        def objective(stretch_vec):
            stretch_matrix = stretch_vec.reshape(self.stretch.shape)  # Reshape back to matrix form
            func, grad = self.regularize_function(stretch_matrix)
            grad = grad.flatten()
            return func, grad

        # Optimization constraints: lower bound 0.1, no upper bound
        bounds = [(0.1, None)] * stretch_init_vec.size  # Equivalent to 0.1 * ones(K, M)

        # Solve optimization problem (equivalent to fmincon)
        result = minimize(
            fun=lambda stretch_vec: objective(stretch_vec)[0],  # Objective function
            x0=stretch_init_vec,  # Initial guess
            method="trust-constr",  # Equivalent to 'trust-region-reflective'
            jac=lambda stretch_vec: objective(stretch_vec)[1],  # Gradient
            bounds=bounds,  # Lower bounds on stretch
            # TODO: A Hessian can be incorporated for better convergence.
        )

        # Update stretch with the optimized values
        self.stretch = result.x.reshape(self.stretch.shape)


def cubic_largest_real_root(p, q):
    """
    Vectorized solver for x^3 + p*x + q = 0.
    Returns the largest real root element-wise.
    """
    # calculate the discriminant
    delta = (q / 2) ** 2 + (p / 3) ** 3
    sqrt_delta = np.sqrt(np.abs(delta))

    # When delta >= 0: one real root
    a = np.cbrt(-q / 2 + sqrt_delta)
    b = np.cbrt(-q / 2 - sqrt_delta)
    root1 = a + b

    # When delta < 0: three real roots, use trigonometric method
    phi = np.arccos(-q / (2 * np.sqrt(-((p / 3) ** 3) + 1e-12)))
    r = 2 * np.sqrt(-p / 3)
    root2 = r * np.cos(phi / 3)

    # Choose correct root depending on sign of delta
    return np.where(delta >= 0, root1, root2)
