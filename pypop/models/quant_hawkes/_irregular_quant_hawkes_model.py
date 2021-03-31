import torch
import numpy as np
import numba
import warnings

from .._models import Model, enforce_observed
from ...fitter import FitterSGD


@numba.jit(nopython=True, fastmath=True)
def _quant_hawkes_model_init_cache(times, counts, M, excit_func_jit):
    dim = len(counts)
    n_jumps = [len(times[i]) for i in range(dim)]
    cache = [np.zeros((dim, M, n_jumps[i])) for i in range(dim)]
    for i in range(dim):
        for j in range(dim):
            for n in range(1, len(times[i])):
                # Time difference from current t_{i,n} all events in j
                t_ij = times[i][n] - times[j]
                # Mask for valid events: those prior to t_{i,n-1}
                valid_ij = (times[i][n-1] - times[j]) >= 0
                # Compute cache value
                kappas = excit_func_jit(t_ij[valid_ij]) * counts[j][valid_ij]
                cache[i][j, :, n] = np.sum(kappas, axis=-1)  # sum over M bases
    return cache


class IrregQuantHawkesModel(Model):
    """
    Irreguarly Quantized Multivariate Hawkes Process
    """

    def __init__(self, excitation, verbose=False, device='cpu', **kwargs):
        """
        Initialize the model

        Arguments:
        ----------
        prior : Prior
            Prior object
        excitation: excitation
            Excitation object
        """
        self.excitation = excitation
        self.M = self.excitation.M or 1
        self.n_jumps = None
        self.dim = None
        self.n_params = None
        self.n_var_params = None
        self._observed = False
        self.verbose = verbose
        if torch.cuda.is_available() and device == 'cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        super().__init__(**kwargs)

    def observe(self, times, counts):
        """
        Set the data for the model
        """
        assert isinstance(counts[0], torch.Tensor)
        assert isinstance(times[0], torch.Tensor)

        # Set the data attributes
        self.counts = counts
        self.times = times
        self.deltas = [torch.cat(
            (ts[:1], ts[1:] - ts[:-1])) for ts in self.times]

        # Set various util attributes
        self.dim = len(counts)
        self.n_params = self.dim * (self.dim * self.excitation.M + 1)
        self.n_jumps = sum(map(sum, counts))

        # Init the pre-computed cache
        if not self._observed:
            self._init_cache()
        self._observed = True

    def _init_cache_python(self):
        """
        caching the required computations

        cache[i][j,0,k]: float
            sum_{t^j < t^i_k} phi(t^i_k - t^j)
            This is used in k^th timestamp of node i, i.e., lambda_i(t^i_k)
        cache_integral: float
            used in the integral of intensity
        """
        self._cache = [torch.zeros(
            (self.dim, self.excitation.M, len(counts_i)),
            dtype=torch.float64, device=self.device)
            for counts_i in self.counts]
        for i in range(self.dim):
            for j in range(self.dim):
                if self.verbose:
                    print((f"\rInitialize cache {i*self.dim+j+1}/{self.dim**2}"
                           "     "), end='')
                for n in range(1, len(self.times[i])):
                    # Time difference from currtent t_{i,n} all events in j
                    t_ij = self.times[i][n] - self.times[j]
                    # Mask for valid events: those prior to t_{i,n-1}
                    valid_ij = (self.times[i][n-1] - self.times[j]) >= 0
                    # Compute cache value
                    kappas = (self.excitation.call(t_ij[valid_ij]) *
                              self.counts[j][valid_ij])
                    kappas = kappas.sum(-1)  # sum over M bases
                    self._cache[i][j, :, n] = kappas
        if self.verbose:
            print()

    def _init_cache(self):
        try:
            times_arr = [np.array(ev, dtype=float) for ev in self.times]
            counts_arr = [np.array(ev, dtype=float) for ev in self.counts]
            # Catch annoying NumbaPendingDeprecationWarning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cache = _quant_hawkes_model_init_cache(
                    times_arr, counts_arr, M=self.excitation.M,
                    excit_func_jit=self.excitation.call_jit())
            self._cache = [torch.tensor(
                ci, dtype=torch.float64, device=self.device) for ci in cache]
        except NotImplementedError:
            print(('Notice: Fast caching not implemented for this excitation '
                   'kernel. Falling back to pure python implementation.'))
            self._init_cache_python()

    @enforce_observed
    def log_likelihood(self, mu, W):
        """
        Log likelihood of an irregularly quantized Hawkes process for the given
        parameters mu and W.

        Arguments:
        ----------
        mu : torch.Tensor
            (dim x 1)
            Base intensities
        W : torch.Tensor
            (dim x dim x M) --> M is for the number of different excitation
            functions
            The weight matrix.
        """
        log_like = 0
        for i in range(self.dim):
            # _cache[i] shape: (dim, M, len(events[i]))
            # W[i] shape: (dim, M) --> need unsqueeze dimension 2
            # mu[i] shape: () --> ok
            lamb_i = mu[i] + (W[i].unsqueeze(2) * self._cache[i]).sum(0).sum(0)
            intens_i = lamb_i * self.deltas[i]
            log_like += torch.sum(self.counts[i] * intens_i.log() - intens_i)
        return log_like

    @enforce_observed
    def mean_squared_loss(self, mu, W):
        """
        Mean-square loss of an irregularly quantized Hawkes process for the
        given parameters mu and W.

        Arguments:
        ----------
        mu : torch.Tensor
            (dim x 1)
            Base intensities
        W : torch.Tensor
            (dim x dim x M) --> M is for the number of different excitation
            functions
            The weight matrix.
        """
        loss = 0.0
        num = 0.0
        for i in range(self.dim):
            lamb_i = mu[i] + (W[i].unsqueeze(2) * self._cache[i]).sum(0).sum(0)
            intens_i = lamb_i * self.deltas[i]
            loss += torch.sum((intens_i - self.counts[i]) ** 2, axis=0)
            num += len(intens_i)
        loss /= num
        return loss

    def lambda_in(self, i, n, mu, W):
        """Compute the intensity in dimension `i` at the `n`-th observation"""
        lamb_in = mu[i] + (W[i] * self._cache[i][:, :, n]).sum(0).sum(0)
        return lamb_in


class IrregQuantHawkesModelMLE(IrregQuantHawkesModel, FitterSGD):
    """Irreguarly Quantized Multivariate Hawkes Process with Maximum Likelihood
    Estimation fitter"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @enforce_observed
    def mle_objective(self, coeffs):
        """Objectvie function for MLE: Averaged negative log-likelihood"""
        mu = self.coeffs[:self.dim]
        W = self.coeffs[self.dim:].reshape(self.dim, self.dim, self.excitation.M)
        return -1.0 * self.log_likelihood(mu, W) / self.n_jumps

    @enforce_observed
    def mle_objective_log_input(self, coeffs):
        """Objectvie function for MLE: Averaged negative log-likelihood"""
        log_mu = self.coeffs[:self.dim]
        log_W = self.coeffs[self.dim:].reshape(self.dim, self.dim, self.excitation.M)
        return -1.0 * self.log_likelihood(torch.exp(log_mu), torch.exp(log_W)) / self.n_jumps

    def fit(self, *args, **kwargs):
        return super().fit(objective_func=self.mle_objective, *args, **kwargs)

    def fit_log_input(self, *args, **kwargs):
        """Fit log-likelihood with log-input variables"""
        return super().fit(objective_func=self.mle_objective_log_input, *args, **kwargs)

    def adjacency(self, exp_link=False):
        W = self.coeffs[self.dim:].reshape(self.dim, self.dim, self.excitation.M).detach()
        if exp_link:
            W = torch.exp(W)
        return W

    def baseline(self, exp_link=False):
        mu = self.coeffs[:self.dim].detach()
        if exp_link:
            mu = torch.exp(mu)
        return mu
