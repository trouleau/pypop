import torch
import numpy as np

from .hawkes_model import HawkesModel
from ...fitter import FitterSGD
from ...utils.decorators import enforce_observed


class NoisyHawkesModel(HawkesModel):
    """
    Same as `HawkesModel` but `_init_cache` is slightly different
    """

    def __init__(self, excitation, min_cutoff, verbose=False, device='cpu'):
        """
        Initialize the model

        Arguments:
        ----------
        prior : Prior
            Prior object
        excitation: excitation
            Excitation object
        """
        super().__init__(excitation=excitation, verbose=verbose, device=device)
        # Minimum time for which `excitation` is non-zero.
        self.min_cutoff = min_cutoff

    def _init_cache(self):
        """
        caching the required computations

        cache[i][j,0,k]: float
            sum_{t^j < t^i_k + M} phi(t^i_k - t^j)
            This is used in k^th timestamp of node i, i.e., lambda_i(t^i_k)
        cache_integral: float
            used in the integral of intensity
        """
        self._cache = [torch.zeros(
            (self.dim, self.excitation.M, len(events_i)), dtype=torch.float64, device=self.device)
            for events_i in self.events]
        for i in range(self.dim):
            for j in range(self.dim):
                if self.verbose:
                    print(f"\rInitialize cache {i*self.dim+j+1}/{self.dim**2}     ", end='')
                id_end = np.searchsorted(
                    self.events[j].cpu().numpy() - self.min_cutoff,
                    self.events[i].cpu().numpy())
                id_start = np.searchsorted(
                    self.events[j].cpu().numpy(),
                    self.events[i].cpu().numpy() - self.excitation.cut_off)
                for k, time_i in enumerate(self.events[i]):
                    t_ij = time_i - self.events[j][id_start[k]:id_end[k]]
                    kappas = self.excitation.call(t_ij).sum(-1)  # (M)
                    self._cache[i][j, :, k] = kappas
        if self.verbose:
            print()

        self._cache_integral = torch.zeros((self.dim, self.excitation.M),
                                           dtype=torch.float64, device=self.device)
        for j in range(self.dim):
            t_diff = self.end_time - self.events[j]
            integ_excit = self.excitation.callIntegral(t_diff).sum(-1)  # (M)
            self._cache_integral[j, :] = integ_excit


class NoisyHawkesModelMLE(NoisyHawkesModel, FitterSGD):
    """Noisy Hawkes Model with Maximum Likelihoof Estimation fitter"""

    @enforce_observed
    def mle_objective(self, coeffs):
        """Objectvie function for MLE: Averaged negative log-likelihood"""
        return -1.0 * self.log_likelihood(coeffs) / self.n_jumps

    def fit(self, *args, **kwargs):
        return super().fit(objective_func=self.mle_objective, *args, **kwargs)
