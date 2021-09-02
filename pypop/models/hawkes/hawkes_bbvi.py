from .hawkes_model import HawkesModel
from .. import ModelBlackBoxVariational
from ...fitter import FitterVariationalEM
from ...posteriors import LogNormalPosterior
from ...priors import LaplacianPrior
from .excitation_kernels import ExponentialKernel


class HawkesModelVariationalEM(ModelBlackBoxVariational, HawkesModel, FitterVariationalEM):
    """
    Multivariate Hawkes Process Model learned with the Variational EM algorithm,
    based on
        https://arxiv.org/abs/1911.00292
    """

    def __init__(self):
            super().__init__(posterior=LogNormalPosterior, prior=LaplacianPrior,
                             C=1e3, n_samples=1, model_kwargs={'excitation': ExponentialKernel(decay=1.0)})

    def fit(self, *args, **kwargs):
        return super().fit(objective_func=self.bbvi_objective, *args, **kwargs)
