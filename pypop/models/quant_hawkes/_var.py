"""
Variational Inference algorithm for multivariate Hawkes processes, according to

    Linderman, Scott W., and Adams, Ryan P. Scalable Bayesian Inference for
    Excitatory Point Process Networks. arXiv preprint arXiv:1507.03228, 2015.

NOTE: Implementation not finished. Running into numerical issues and/or bugs.

FIXME: update functions are separated from the model object to enable `numba`
jit compilation. It might not be needed though... Updates should be made part of
the object.

FIXME: Model object `VariationalQuantizedHawkesModel` is not a `Model` because
it does not implement the log-likelihood function. Should merge with the basic
discrete-time MHP object `RegularlyQuantizedHawkesModel`.
"""
import torch
import numpy as np
# import numba
from scipy.special import gamma, digamma

from .._models import enforce_observed
from ...fitter import FitterIterativeNumpy

def _expect_log_gamma(shape_params, rate_params):
    """Expected value ``E_q[ln q]` of log of a Gamma distribution `q`"""
    return digamma(shape_params) - np.log(rate_params)

def _expect_log_mu(mu_s_po, mu_r_po):
    return _expect_log_gamma(mu_s_po, mu_r_po)

def _expect_log_w(a_po, w_pos_s_po, w_pos_r_po, w_neg_s_po, w_neg_r_po):
    pos_edges_part = _expect_log_gamma(w_pos_s_po, w_pos_r_po)
    neg_edges_part = _expect_log_gamma(w_neg_s_po, w_neg_r_po)
    return a_po * pos_edges_part + (1 - a_po) * neg_edges_part

def _expect_log_g(g_po):
    return digamma(g_po) - digamma(g_po.sum(axis=0))[np.newaxis,:,:]

def _expect_z(z_exo_po, z_end_po, event_count):
    # shape of `event_count`: (n_bins, dim)
    expect_z_exo = z_exo_po * event_count.T  # shape: (n_bins, dim)
    expect_z_end = z_end_po * event_count.T[:, np.newaxis, np.newaxis,:]  # shape: (n_bins, M, dim, dim)
    return expect_z_exo, expect_z_end

def _update_mu(mu_s_pr, mu_r_pr, z_exo_po, n_bins, delta, dim):
    mu_s_po = mu_s_pr + z_exo_po.sum(axis=0)
    mu_r_po = (mu_r_pr + n_bins * delta) * np.ones(dim)
    return mu_s_po, mu_r_po

def _update_g(g_pr, expect_z_end):
    g_po = g_pr[:,np.newaxis, np.newaxis] * expect_z_end.sum(axis=0)
    return g_po

def _update_w(w_pos_s_pr, w_pos_r_pr, w_neg_s_pr, w_neg_r_pr, expect_z_end, num_events_per_dim, dim):
    expect_z_end_summed_bins_bases = expect_z_end.sum(axis=0).sum(axis=0)
    w_pos_s_po = w_pos_s_pr + expect_z_end_summed_bins_bases
    w_pos_r_po = (w_pos_r_pr + num_events_per_dim[:,np.newaxis]) * np.ones((dim, dim))
    w_neg_s_po = w_neg_s_pr + expect_z_end_summed_bins_bases
    w_neg_r_po = (w_neg_r_pr + num_events_per_dim[:,np.newaxis]) * np.ones((dim, dim))
    return w_pos_s_po, w_pos_r_po, w_neg_s_po, w_neg_r_po

def _update_a(net_edge_pr_ratio, w_pos_s_pr, w_pos_r_pr, w_neg_s_pr, w_neg_r_pr, w_pos_s_po, w_pos_r_po, w_neg_s_po, w_neg_r_po):
    p_ratio = (
        net_edge_pr_ratio
        * (w_pos_r_pr ** w_pos_s_pr) / gamma(w_pos_s_pr)
        * np.clip(gamma(w_pos_s_po) / np.clip(w_pos_r_po ** w_pos_s_po, 0.0, 1e4), 0.0, 1e4)
        * gamma(w_neg_s_pr) / (w_neg_r_pr ** w_neg_s_pr)
        * np.clip((w_neg_r_po ** w_neg_s_po) / np.clip(gamma(w_neg_s_po), 0.0, 1e4), 0.0, 1e4)
    )
    print('gamma(w_pos_s_po) / np.clip(w_pos_r_po ** w_pos_s_po, 0.0, 1e4) =')
    print(np.clip(gamma(w_pos_s_po) / np.clip(w_pos_r_po ** w_pos_s_po, 0.0, 1e4), 0.0, 1e4))
    print('(w_neg_r_po ** w_neg_s_po) / np.clip(gamma(w_neg_s_po), 0.0, 1e4) =')
    print(np.clip((w_neg_r_po ** w_neg_s_po) / np.clip(gamma(w_neg_s_po), 0.0, 1e4), 0.0, 1e4))
    a_po = p_ratio / (p_ratio + 1)
    return a_po

def _update_z(shape_end, shape_exo, event_count_per_basis, mu_s_po, mu_r_po, g_po, a_po, w_pos_s_po, w_pos_r_po, w_neg_s_po, w_neg_r_po):
    """Update the parent variables, denoted by `z`"""
    # For background (exogenous) parents
    # Array of probabilities
    z_exo_po = np.zeros(shape_exo)  # `z[t, k]`, shape: (n_bins, dim)
    # For other (endogenous) parents
    z_end_po = np.zeros(shape_end)  # `z[t, b, k, k']`, shape: (n_bins, M, dim, dim)

    # Background parents
    expect_log_mu = _expect_log_gamma(mu_s_po, mu_r_po)
    z_exo_po = np.exp(expect_log_mu) * np.ones(shape_exo)

    # Other parents
    # NOTE: All `np.newaxis` are to correct shapes for `numpy` broadcasting
    expect_log_g = _expect_log_g(g_po) # shape: (M, dim, dim)
    expect_log_w = _expect_log_w(a_po, w_pos_s_po, w_pos_r_po, w_neg_s_po, w_neg_r_po)[np.newaxis,:,:]
    exp_expect_g_w = np.exp(expect_log_g + expect_log_w)[np.newaxis,:,:,:]
    z_end_po = event_count_per_basis[:,:,:,np.newaxis] * exp_expect_g_w

    return z_exo_po, z_end_po


class VariationalQuantizedHawkesModel(FitterIterativeNumpy):
    """
    Bayesian Quantized Hawkes Model with Spike-and-Slab prior, as in

    ```
    Linderman, Scott W., and Adams, Ryan P. Scalable Bayesian Inference for
    Excitatory Point Process Networks. arXiv preprint arXiv:1507.03228, 2015.
    ```
    """

    def __init__(self, excitation, verbose=False):
        raise NotImplementedError("VI algorithm not implemented")
        self.excitation = excitation
        self.M = self.excitation.M
        self.n_jumps = None
        self.dim = None
        self.n_params = None
        self.n_var_params = None
        self._fitted = False
        self.verbose = verbose

    def observe(self, delta, counts):
        # Set the event data attributes
        assert counts.shape[0] < counts.shape[1], "Number of bins must be larger than the number of dimensions"
        self.counts = counts.astype(float)  # shape: (dim, n_bins)
        self.delta = delta  # Bin size
        # Self misc attributes
        assert self.counts.ndim == 2, "Invalid shape for `counts`. Must be of shape 'num. bins' x 'num. dim'."
        self.dim, self.n_bins = self.counts.shape
        # Preprocess events for fitting
        self.num_events_per_dim = self.counts.sum(axis=1)
        self.event_count_per_basis = np.zeros((self.n_bins, self.M, self.dim))
        # Basis cut-off (in number of bins)
        self.basis_cut_off = int(np.floor(self.excitation.cut_off / self.delta))
        # Discretize excitation into a discrete basis function
        vals = delta * torch.arange(self.basis_cut_off)
        self.phi_dts = self.excitation.call(vals).numpy()
        print(self.phi_dts.shape)
        self.phi_dts /= self.phi_dts.sum()  # Basis should sum to one
        for m in range(self.M):
            for i in range(self.dim):
                for t in range(2,self.n_bins):
                    t_min = max(0, t-self.basis_cut_off-1)
                    t_max = t - 1
                    # print(f"m={m}, i={i}, t={t}, t_min={t_min}, t_max={t_max}, t_max-t_min={t_max-t_min}")  # DEBUG
                    self.event_count_per_basis[t,m,i] = np.sum(self.counts[i, t_min:t_max] * self.phi_dts[:t_max-t_min][::-1])

    def _init_fit(self, mu_s_pr, mu_r_pr, g_pr, w_pos_s_pr, w_pos_r_pr, w_neg_s_pr, w_neg_r_pr, net_edge_pr):

        # Recall notation:
        # ----------------
        #    dim: number of dimension
        # n_bins: number of time bins
        #      M: number of basis for excitation functions
        #  delta: bin size

        # Set prior hyper-parameters
        # Mu (background) parameters
        self._mu_s_pr = mu_s_pr  # Shape of Gamma prior, shape: (1,)
        self._mu_r_pr = mu_r_pr  #  Rate of Gamma prior, shape: (1,)
        #  Excitation parameters prior
        self._g_pr = g_pr  # Shape of Dirichlet prior, shape: (M,)
        # Edge weights
        self._w_pos_s_pr = w_pos_s_pr  # For positive edges, Shape of Gamma prior, shape: (1,)
        self._w_pos_r_pr = w_pos_r_pr  # For positive edges,  Rate of Gamma prior, shape: (1,)
        self._w_neg_s_pr = w_neg_s_pr  # For non-edges, Shape of Gamma prior, shape: (1,), should be small
        self._w_neg_r_pr = w_neg_r_pr  # For non-edges,  Rate of Gamma prior, shape: (1,), should be large
        # Network binary edge prior
        self._net_edge_pr = net_edge_pr  # Prob. of Bernoulli prior, shape: (dim, dim) or (1,)
        self.net_edge_pr_ratio = net_edge_pr / (1 - net_edge_pr) # Ratio p/(1-p) of edge probabilities
        # self._net_weight_rate  # NOTE:  No Bayesian weighted network model in this implemention. Fixed `self._w_pos_r_pr` instead

        # Init posterior hyper-parameters
        self._z_exo_po = None  # Background (exogenous) parent variable, probability of Multiomial posterior, shape: (n_bins, dim)
        self._z_end_po = None  # Other (endogenous) parent variable, probability of Multiomial posterior, shape: (n_bins, M, dim, dim)
        self._mu_s_po = mu_s_pr * np.ones(self.dim)  # Mu (background), Shape of Gamma posterior, shape: (dim,)
        self._mu_r_po = mu_r_pr * np.ones(self.dim)  # Mu (background), Rate of Gamma posterior, shape: (dim,)
        self._g_po = g_pr[:, np.newaxis, np.newaxis] * np.ones((self.M, self.dim, self.dim))  # Excitation parameters, Dirichlet posterior, shape: (M, dim, dim)
        self._w_pos_s_po = w_pos_s_pr * np.ones((self.dim, self.dim))  # Edge weights (for positive edges), Shape of Gamma posterior, shape: (dim, dim)
        self._w_pos_r_po = w_pos_r_pr * np.ones((self.dim, self.dim))  # Edge weights (for positive edges),  Rate of Gamma posterior, shape: (dim, dim)
        self._w_neg_s_po = w_neg_s_pr * np.ones((self.dim, self.dim))  # Edge weights (for non-edges), Shape of Gamma posterior, shape: (dim, dim)
        self._w_neg_r_po = w_neg_r_pr * np.ones((self.dim, self.dim))  # Edge weights (for non-edges),  Rate of Gamma posterior, shape: (dim, dim)
        self._a_po = net_edge_pr  # Edge, probability of Bernoulli posterior, shape: (dim, dim)

    def _iteration(self):
        # Update parent variables
        self._z_exo_po, self._z_end_po = _update_z(
            shape_end=(self.n_bins, self.M, self.dim, self.dim),
            shape_exo=(self.n_bins, self.dim),
            event_count_per_basis=self.event_count_per_basis,
            mu_s_po=self._mu_s_po,
            mu_r_po=self._mu_r_po,
            g_po=self._g_po,
            a_po=self._a_po,
            w_pos_s_po=self._w_pos_s_po,
            w_pos_r_po=self._w_pos_r_po,
            w_neg_s_po=self._w_neg_s_po,
            w_neg_r_po=self._w_neg_r_po)
        _, expect_z_end = _expect_z(
            z_exo_po=self._z_exo_po,
            z_end_po=self._z_end_po,
            event_count=self.counts)
        # Update background rates
        self._mu_s_po, self._mu_r_po = _update_mu(mu_s_pr=self._mu_s_pr,
                                      mu_r_pr=self._mu_r_pr,
                                      z_exo_po=self._z_exo_po,
                                      n_bins=self.n_bins,
                                      delta=self.delta,
                                      dim=self.dim)
        # Update basis weights
        self._g_po = _update_g(g_pr=self._g_pr, expect_z_end=expect_z_end)
        # Update edge weights
        self._w_pos_s_po, self._w_pos_r_po, self._w_neg_s_po, self._w_neg_r_po = _update_w(
            w_pos_s_pr=self._w_pos_s_pr,
            w_pos_r_pr=self._w_pos_r_pr,
            w_neg_s_pr=self._w_neg_s_pr,
            w_neg_r_pr=self._w_neg_r_pr,
            expect_z_end=expect_z_end,
            num_events_per_dim=self.num_events_per_dim,
            dim=self.dim)
        # Update edge probs
        self._a_po = _update_a(self.net_edge_pr_ratio,
                               w_pos_s_pr = self._w_pos_s_pr,
                               w_pos_r_pr = self._w_pos_r_pr,
                               w_neg_s_pr = self._w_neg_s_pr,
                               w_neg_r_pr = self._w_neg_r_pr,
                               w_pos_s_po = self._w_pos_s_po,
                               w_pos_r_po = self._w_pos_r_po,
                               w_neg_s_po = self._w_neg_s_po,
                               w_neg_r_po = self._w_neg_r_po)

    @enforce_observed
    def fit(self, mu_s_pr, mu_r_pr, g_pr, w_pos_s_pr, w_pos_r_pr, w_neg_s_pr, w_neg_r_pr, net_edge_pr, *args, **kwargs):
        self._init_fit(mu_s_pr, mu_r_pr, g_pr, w_pos_s_pr, w_pos_r_pr, w_neg_s_pr, w_neg_r_pr, net_edge_pr)
        return super().fit(step_function=self._iteration, *args, **kwargs)
