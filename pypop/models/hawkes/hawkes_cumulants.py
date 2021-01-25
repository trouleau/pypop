from multiprocessing import Pool
from functools import partial
import numpy as np
import numpy.linalg as al
from numpy.linalg import inv, norm
import scipy.linalg

import torch

from ._cumulant_computer import cumulant_computer
from ...fitter import FitterSGD

from tick.hawkes import HawkesCumulantMatching as _HawkesCumulantMatching


def compute_R(G):
    """Compute the matrix R from a matrix G"""
    dim = G.shape[0]
    if isinstance(G, torch.Tensor):
        return torch.inverse(torch.eye(dim) - G)
    return inv(np.eye(dim) - G)


def compute_G(R):
    """Compute the matrix G from a matrix R"""
    dim = R.shape[0]
    if isinstance(R, torch.Tensor):
        return torch.eye(dim) - torch.inverse(R)
    return np.eye(dim) - inv(R)


def compute_cumulants(G, mus, R=None, return_R=False):
    """
    Compute the cumulants of a Hawkes process given the integrated kernel
    matrix `G` and the baseline rate vector `mus`

    Arguments
    ---------
    G : np.ndarray
        The integrated kernel matrix of shape shape dim x dim
    mus : np.ndarray
        The baseline rate vector of shape dim
    R : np.ndarray (optional)
        Precomputed matrix R
    return_R : bool (optional)
        Return the matrix R if set to `True`

    Return
    ------
    L : np.ndarray
        Mean intensity matrix
    C : np.ndarray
        Covariance matrix
    Kc : np.ndarray
        Skewness matrix
    R : np.ndarray (returned only if `return_R` is True)
        Internal matrix to compute the cumulants
    """
    if not len(G.shape) == 2:
        raise ValueError("Matrix `G` should be 2-dimensional")
    if not len(mus.shape) == 1:
        raise ValueError("Vector `mus` should be 1-dimensional")
    if not G.shape[0] == G.shape[1]:
        raise ValueError("Matrix `G` should be a squared matrix")
    if not G.shape[0] == mus.shape[0]:
        raise ValueError("Vector `mus` should have the same dinension as `G`")
    R = compute_R(G)
    L = np.diag(R @ mus)
    C = R @ L @ R.T
    Kc = (R**2) @ C.T + 2 * R * (C - R @ L) @ R.T
    if return_R:
        return L, C, Kc, R
    return L, C, Kc


class HawkesCumulantLearner(FitterSGD):

    def __init__(self, integration_support, cs_ratio=None):
        super().__init__()
        self._cumulants_ready = False
        self.integration_support = integration_support
        assert isinstance(integration_support, float) and (integration_support > 0)
        self.cs_ratio = cs_ratio
        self.device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'

    def observe(self, events):
        if not isinstance(events[0], list):
            events = [events]
        self.events = events
        self.end_times = [max(map(max, ev)) for ev in self.events]
        self.num_real = len(self.events)
        self.dim = len(self.events[0])
        # Reset cumulants ready flag is new events are prodived
        self._cumulants_ready = False

    def set_cumulants(self, L_vec, C, Kc):
        self.dim = len(L_vec)
        if not L_vec.shape == (self.dim,):
            raise ValueError(f"`L_vec` should be 1-dimensional")
        self.L_vec = torch.tensor(L_vec)
        if not C.shape == (self.dim, self.dim):
            raise ValueError(f"Invalid shape for `C`")
        self.C = torch.tensor(C)
        self.F = torch.tensor(scipy.linalg.sqrtm(self.C).astype(np.float))
        if not np.allclose(self.F @ self.F.T, self.C):
            print('WARNING: F @ F.T not close to C')
        if not Kc.shape == (self.dim, self.dim):
            raise ValueError(f"Invalid shape for `Kc`")
        self.Kc = torch.tensor(Kc)
        if self.cs_ratio is None:
            self.cs_ratio = self._estimate_cs_ratio()
        self._cumulants_ready = True

    def _estimate_cs_ratio(self):
        norm_skewness = norm(self.Kc.numpy()) ** 2
        norm_covariance = norm(self.C.numpy()) ** 2
        return float(norm_skewness / (norm_skewness + norm_covariance))

    def _estimate_mean(self):
        mean_intensity_all_real = np.zeros((self.num_real, self.dim))
        for r in range(self.num_real):
            for i in range(self.dim):
                mean_intensity_all_real[r, i] = len(self.events[r][i]) / self.end_times[r]
        mean_intensity = np.array(mean_intensity_all_real.mean(axis=0))
        return mean_intensity

    def _estimate_covariance(self, mean_intensity):
        C = cumulant_computer.compute_covariance(
            integration_support=self.integration_support,
            end_times=self.end_times,
            mean_intensity=mean_intensity,
            multi_events=self.events)
        C = np.array(C)
        C = 0.5 * (C + C.T)
        return C

    def _estimate_skewness(self, mean_intensity):
        Kc = cumulant_computer.compute_skewness(
            integration_support=self.integration_support,
            end_times=self.end_times,
            mean_intensity=mean_intensity,
            multi_events=self.events)
        return Kc

    def _estimate_cumulants(self):
        mean_intensity = self._estimate_mean()
        covariance = self._estimate_covariance(mean_intensity)
        skewness = self._estimate_skewness(mean_intensity)
        # Vector of mean intensity
        self.L_vec = torch.tensor(mean_intensity)
        # Covariance matrix
        self.C = torch.tensor(covariance)
        # Square-root of matrix C, s.t. C = F * F.T
        self.F = torch.tensor(scipy.linalg.sqrtm(covariance))
        # Skewness matrix
        self.Kc = torch.tensor(skewness)
        if self.cs_ratio is None:
            self.cs_ratio = self._estimate_cs_ratio()
        self._cumulants_ready = True

    def _estimate_cumulants_nphc(self):
        nphc = _HawkesCumulantMatching(self.integration_support,
                                       cs_ratio=self.cs_ratio,
                                       C=1e3)
        nphc.fit(self.events)
        self.nphc = nphc
        self.L_vec = torch.tensor(nphc.mean_intensity)
        self.C = torch.tensor(nphc.covariance)
        self.F = torch.tensor(scipy.linalg.sqrtm(nphc.covariance).astype(float))
        self.Kc = torch.tensor(nphc.skewness.T)
        if self.cs_ratio is None:
            self.cs_ratio = self._estimate_cs_ratio()
        self._cumulants_ready = True

    def _compute_initial_guess(self, seed):
        # Sample random orthogonal matrix
        X_start = torch.tensor(scipy.stats.ortho_group.rvs(dim=self.dim,
                                                           random_state=seed))
        # Use decomposition of R
        R_start = self.F @ X_start @ torch.diag(1 / self.L_vec.sqrt())
        return R_start.detach()

    def objective_R(self, R, return_parts=False):
        L = torch.diag(self.L_vec)
        C = self.C
        C_part = R.mm(L).mm(R.T)
        Kc_part = (R ** 2).mm(C.T) + 2 * (R * (C - R.mm(L))).mm(R.T)
        if return_parts:
            return C_part, Kc_part
        return ((1 - self.cs_ratio) * torch.mean((Kc_part - self.Kc) ** 2)
                + self.cs_ratio * torch.mean((C_part - self.C) ** 2))

    def objective_G(self, G):
        R = torch.inverse(torch.eye(self.dim) - G)
        return self.objective_R(R)

    def fit_R(self, R_start=None, seed=None, **kwargs):
        if R_start is None:  # Sample random initial guess
            print('- Sample random initial guess R')
            R_start = self._compute_initial_guess(seed=seed)
        self.fit(objective_func=self.objective_R, x0=R_start, **kwargs)
        self.solution = self.coeffs.detach()
        self.adjacency = haw

    def fit_G(self, G_start=None, seed=None, **kwargs):
        if G_start is None:  # Sample random initial guess
            print('- Sample random initial guess G')
            R_start = self._compute_initial_guess(seed=seed)
            G_start = torch.eye(self.dim) - torch.inverse(R_start)
            G_start = G_start.detach()
        return self.fit(objective_func=self.objective_G, x0=G_start, **kwargs)
