import numpy as np
import torch
import abc

from .utils.decorators import enforce_observed


def identitiy_link(x):
    return x


class Fitter(metaclass=abc.ABCMeta):
    """
    Abstract fitter object for fitting algorithms with a `fit` function
    """

    def _check_convergence(self, tol):
        """Check convergence of `fit`"""
        # Keep this feature in `numpy`, even for `torch` backends
        loss = float(self.loss)
        if hasattr(self, 'loss_prev'):
            rel_loss = abs(loss - self.loss_prev) / abs(self.loss_prev)
            if rel_loss < tol:
                return True
        self.loss_prev = float(loss)
        return False

    @abc.abstractmethod
    @enforce_observed
    def fit(self, *args, **kwargs):
        pass


class FitterIterativeNumpy(Fitter):
    """
    Basic fitter for iterative algorithms (no gradient descent) using `numpy`
    """

    @enforce_observed
    def fit(self, *, step_function, tol, max_iter, seed=None, callback=None):
        """
        Fit the model.

        Arguments:
        step_function : callable
            Function to evaluate at each iteration
        tol : float
            Tolerence for convergence
        max_iter : int
            Maximum number of iterations
        callback : callable
            Callback function that takes as input `self`

        Returns:
        --------
        converged : bool
            Indicator of convergence
        """
        # Set random seed
        if seed:
            np.random.seed(seed)
        # Set callable if None
        if callback is None:
            def callback(arg, end=''): pass
        for t in range(max_iter):
            self._n_iter_done = t
            # Run iteration
            step_function()

            # Sanity check that the optimization did not fail
            if np.isnan(self.coeffs).any():
                raise ValueError('NaNs in coeffs! Stop optimization...')

            if (t+1) % 10 == 0:
                # Check convergence in callback (if available)
                if hasattr(callback, 'has_converged'):
                    if callback.has_converged():
                        callback(self, end='\n')  # Callback before the end
                        return True
                # Or, check convergence in fitter, and then callback
                if self._check_convergence(tol):
                    callback(self, end='\n')  # Callback before the end
                    return True

            callback(self)  # Callback at each iteration
        return False


class FitterSGD(Fitter):
    """
    Simple SGD Fitter projected on positive hyperplane based on `torch`.

    Methods:
    --------
    fit : Fit the model
    """

    _allowed_penalties = ['l1', 'l2', 'elasticnet', 'none']

    def __init__(self, **kwargs):
        self._n_iter_done = 0
        self.coeffs = None
        self.loss = np.inf
        self.elastic_net_ratio = None
        self.penalty_link = identitiy_link
        self.penalty_C = None
        super().__init__(**kwargs)

    def _init_penalty(self, penalty_name, penalty_link, elastic_net_ratio, penalty_C):
        # Set name of penalty
        self.penalty_name = penalty_name
        # Check penalty link type
        if penalty_link is not None:
            self.penalty_link = penalty_link
        # Set config
        if self.penalty_name == 'elasticnet':
            self.elastic_net_ratio = elastic_net_ratio
            self.penalty_C = penalty_C
        elif self.penalty_name == 'l1':
            self.elastic_net_ratio = 1.0
            self.penalty_C = penalty_C
        elif self.penalty_name == 'l2':
            self.elastic_net_ratio = 0.0
            self.penalty_C = penalty_C
        elif self.penalty_name == 'none':
            self.elastic_net_ratio = 1.0
            self.penalty_C = np.inf
        else:
            raise ValueError(f'Invalid penalty name, must be: {self._allowed_penalties}')

    @property
    def strength_ridge(self):
        return (1 - self.elastic_net_ratio) / self.penalty_C

    @property
    def strength_lasso(self):
        return self.elastic_net_ratio / self.penalty_C

    def penalty(self, coeffs):
        if self.penalty_name == 'none':
            # If no penalty, return 0.0 penalty
            return 0.0
        # L2 Penalty
        l2_reg = torch.sum(self.penalty_link(coeffs) ** 2)
        # L1 Penalty
        l1_reg = torch.abs(self.penalty_link(coeffs)).sum()
        return self.strength_lasso * l1_reg + self.strength_ridge * l2_reg

    def _take_gradient_step(self):
        # Gradient update
        self.optimizer.zero_grad()
        self._loss = self._objective_func(self.coeffs) + self.penalty(self.coeffs)
        self._loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.loss = self._loss.detach()
        if self.positive_constraint:  # Project to positive
            with torch.no_grad():
                self.coeffs[self.coeffs < 1e-20] = 1e-20

    @property
    def coeffs_copy(self):
        return self.coeffs.detach().clone()

    @property
    def number_of_iterations(self):
        return self._n_iter_done

    def fit(self, *, objective_func, x0, optimizer, lr, lr_sched, tol, max_iter,
            penalty_name='none', penalty_C=1e3, penalty_link=None, elastic_net_ratio=0.95,
            seed=None, positive_constraint=False, callback=None):
        """
        Fit the model.

        Arguments:
        ----------
        objective_func : callable
            Objective function to minimize
        x0 : torch.tensor
            Initial estimate
        optimizer : torch.optim
            Optimizer object
        lr : float
            Learning rate of optimizer
        lr_sched : float
            Exponential decay of learning rate scheduler
        tol : float
            Tolerence for convergence
        max_iter : int
            Maximum number of iterations
        penalty_name : str
            Type of penalty
        penalty_C : float
            Inverse weight of penalty
        penalty_link : callable (optional, default: None)
            Link function to apply to coefficients to for the penalty term.
            Namely, let $x$ be the coefficients, $g(\cdot)$ be the penalty and
            $f(\cdot)$ be the link function, then the penalty term is $g(f(x))$
        elastic_net_ratio : float
            Ratio of elasticnet penalty
            (Set to 1 for L1 only, set to 0 for L2 only)
        seed : int
            Random seed (for both `numpy` and `torch`)
        positive_constraint : bool (optional, default: True)
            Indicate whether to project the gradient steps onto the positive
            plane.
        callback : callable
            Callback function that takes as input `self`

        Returns:
        --------
        converged : bool
            Indicator of convergence
        """
        # Set random seed
        if seed:
            np.random.seed(seed)
            torch.manual_seed(seed)
        # Initialize estimate
        self.coeffs = x0.clone().detach().to(self.device).requires_grad_(True)
        # Set callable if None, and call it
        if callback is None:
            def callback(*arg, **kwargs): pass
        callback(self)
        # Set alias for objective function
        self._objective_func = objective_func
        # Set penalty term attributes
        self._init_penalty(penalty_name=penalty_name,
                           penalty_link=penalty_link,
                           elastic_net_ratio=elastic_net_ratio,
                           penalty_C=penalty_C)
        # Set positive constraint attribute (used in `_take_gradient_step`)
        self.positive_constraint = positive_constraint
        # Reset optimizer & scheduler
        self.optimizer = optimizer([self.coeffs], lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                gamma=lr_sched)
        # Iteration main loop
        for t in range(int(max_iter)):
            self._n_iter_done = t
            self._take_gradient_step()

            # Check that the optimization did not fail
            if torch.isnan(self.coeffs).any():
                raise ValueError('NaNs in coeffs! Stop optimization...')

            # Check convergence in callback (if available)
            if hasattr(callback, 'has_converged'):
                if callback.has_converged():
                    callback(self, end='\n', force=True)  # Callback before the end
                    return True

            # Or, check convergence in fitter, and then callback
            if self._check_convergence(tol):
                callback(self, end='\n', force=True)  # Callback before the end
                return True

            if t < int(max_iter) - 1:
                callback(self)  # Callback at each iteration

        callback(self, end='\n', force=True)  # Callback before the end
        return False


class FitterVariationalEM(Fitter):
    """
    Fitter object implementing the Variational EM algorithm.

    """

    def _e_step(self):
        """"
        Perform a signle gradient updates of posterior coefficients `coeffs`
        """
        # Gradient update
        self.optimizer.zero_grad()
        self._loss = self._objective_func(self.coeffs)
        self._loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def _m_step(self):
        """"Update the parameters of the prior `C`"""
        # Update hyper-parameters
        condition = ((self._n_iter_done + 1) % self.mstep_interval == 0
                     and self._n_iter_done > self.mstep_offset)
        if condition:
            self.hyper_parameter_learn(self.coeffs.detach().to(self.device),
                                       momentum=self.mstep_momentum)

    @enforce_observed
    def fit(self, *, objective_func, x0=None, optimizer=torch.optim.Adam, lr=0.1,
            lr_sched=0.999, tol=1e-6, max_iter=10000, mstep_interval=100,
            mstep_offset=0, mstep_momentum=0.5, seed=None, callback=None):
        """
        Fit the model.

        Arguments:
        ----------
        objective_func : callable
            Objective function to minimize
        x0 : torch.tensor
            Initial estimate
        optimizer : torch.optim
            Optimizer object
        lr : float
            Learning rate of optimizer
        lr_sched : float
            Exponential decay of learning rate scheduler
        tol : float
            Tolerence for convergence
        max_iter : int
            Maximum number of iterations
        mstep_interval : int
            Number of iterations between M-Step updates
        mstep_offset : int
            Number of iterations before first M-Step
        mstep_momentum : float
            Momentum of M-step
        seed : int
            Random seed (for both `numpy` and `torch`)
        callback : callable
            Callback function that takes as input `self`

        Returns:
        --------
        converged : bool
            Indicator of convergence
        """
        # Set random seed
        if seed:
            np.random.seed(seed)
            torch.manual_seed(seed)
        # Set callable if None
        if callback is None:
            def callback(*args, **kwargs): pass
        # Set initial estimate if None
        if x0 is None:
            x0 = 0.1 * torch.ones(self.n_var_params)
        # Set alias for objective function
        self._objective_func = objective_func
        # Set the attributes for the E and M steps
        self.mstep_interval = mstep_interval
        self.mstep_offset = mstep_offset
        self.mstep_momentum = mstep_momentum
        # Initialize estimate
        self.coeffs = x0.clone().detach().to(self.device).requires_grad_(True)
        # Reset optimizer & scheduler
        self.optimizer = optimizer([self.coeffs], lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                gamma=lr_sched)
        for t in range(max_iter):
            self._n_iter_done = t
            # E step
            self._e_step()
            # M step
            self._m_step()
            # Check that the optimization did not fail
            if torch.isnan(self.coeffs).any():
                raise ValueError('NaNs in coeffs! Stop optimization...')

            if (t+1) % 100 == 0:
                # Check convergence in callback (if available)
                if hasattr(callback, 'has_converged'):
                    if callback.has_converged(n=10):
                        callback(self, end='\n')  # Callback before the end
                        return True
                # Or, check convergence in fitter, and then callback
                if self._check_convergence(tol):
                    callback(self, end='\n')  # Callback before the end
                    return True

            callback(self)  # Callback at each iteration
        return False
