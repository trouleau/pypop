"""
Regulary-quantized Hawkes processes

FIXME: Implement log-likelihood from `IrregQuantHawkesModel` for the simpler
regularly-quantized case.
"""
import itertools
import torch
import numpy as np
import numba
import warnings

from scipy.special import gamma, digamma

from .._models import Model, enforce_observed
from ...fitter import FitterIterativeNumpy


class RegularlyQuantizedHawkesModel:
    """
    Reguarly Quantized Multivariate Hawkes Process
    """

    def __init__(self, p, verbose=False):
        """
        Initialize the model

        Arguments:
        ----------
        ...

        """
        # Set model parameters
        self.p = p
        # Init general attributes
        self.n_jumps = None
        self.n_bins = None
        self.dim = None
        self.n_params = None
        self.n_var_params = None
        self._fitted = False
        self.verbose = verbose

    def observe(self, delta, counts):
        """
        Set the data for the model
        """
        assert isinstance(counts[0], np.ndarray)

        # Set the data attributes
        self.counts = counts
        self.delta = delta

        # Set various util attributes
        self.dim = len(counts)
        self.n_params = self.dim * (self.dim * self.p + 1)
        self.n_jumps = sum(map(sum, counts))
        self.n_bins = self.counts.shape[1]

        if self.p >= self.n_bins:
            raise RuntimeError("Number should be larger than `p`.")

        self._fitted = True

    def log_likelihood(self, coeffs):
        raise NotImplementedError((
            'Not implemented for regularly quantized model. '
            'Use `IrregQuantHawkesModel` instead.'))


class RegularlyQuantizedHawkesModelCLS(RegularlyQuantizedHawkesModel):
    """
    Reguarly Quantized Multivariate Hawkes Process with Conditional Least Square
    (CLS) estimation
    """

    def _build_Z(self):
        """Produces Z matrix of estimator from dicrete bins

        Arguments:
            bins {np.ndarray} -- array of bins, per process event counts,
            p {int} -- lag of INAR process

        Returns:
            ray -- Z matrix of estimator
        """
        n_cols = self.n_bins - self.p
        res = np.zeros((self.dim * self.p + 1, n_cols))
        for i, shift_i in enumerate(range(self.p - 1, -1, -1)):
            for k, bin_k in enumerate(self.counts):
                res[i * self.dim + k] = bin_k[shift_i: shift_i + n_cols]
        res[-1] = np.ones(n_cols)
        return res

    def _build_Y(self):
        """Produce Y matrix of estimator

        Arguments:
            bins {np.ndparray} -- array of bins, per process event counts
            p {int} -- lag of INAR process

        Returns:
            np.ndarray -- Y matrix of estimator
        """
        return self.counts[:, self.p:]


    def fit(self):
        """Compute Kirchner's estimator

        Arguments:
            bins {np.ndarray} -- array of bins, per process event counts
            p {int} -- lag of INAR process

        Returns:
            np.ndarray -- d x (dp + 1) estimator
        """
        self.y_matrix_ = self._build_Y()
        self.z_matrix_ = self._build_Z()
        z_matrix_t = self.z_matrix_.T
        # Compute estimator
        z_zt_inv = np.linalg.inv(np.dot(self.z_matrix_, z_matrix_t))  # Could raise np.linalg.LinAlgError
        theta = self.y_matrix_.dot(z_matrix_t).dot(z_zt_inv) #/ self.delta
        # Format estimator in desired shape
        #  matrix of estimates of the form (from, to, at)
        #  where from/to relates to origin/dest. of influence
        #  and at relates to sample.
        res = np.zeros((self.dim, self.dim, self.p))
        for col_start in range(self.dim):
            indexes = np.ix_(range(self.dim), [self.dim * k + col_start for k in range(self.p)])
            res[:, col_start, :] = theta[indexes]
        return res
