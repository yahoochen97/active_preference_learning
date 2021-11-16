import numpy as np
import torch
import gpytorch
from utils.model import IndicatorKernel
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.models import ExactGP

class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel())
        # self.i_covar_module = ScaleKernel(IndicatorKernel())

    def forward(self, x):
        # indices = x[:,0]
        # weights = x[:,1:]
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # i_covar_x = self.i_covar_module(indices)
        # covar_x = w_covar_x + i_covar_x
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def generate_data(num_points=1000, num_features=1, SEED=1):
    '''
    Generate features and latent scores.

    Parameters
    ----------
    num_points : int
        number of points
    num_features : int
        number of features
    SEED: int
        random seed
    '''
    np.random.seed(SEED)
    torch.random.manual_seed(SEED)

    # sample features
    # idx = np.arange(num_points).reshape((-1,1))
    xs = np.random.normal(loc=0.0, scale=1.0, size=(num_points, num_features))
    # xs = np.hstack((idx, xs))
    
    # define data generating function as a GP
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(torch.Tensor(xs), torch.Tensor(xs), likelihood)

    # specify hyperparameters in a cell
    hypers = {
    'likelihood.noise_covar.noise': torch.tensor(0.1**2),
    'covar_module.base_kernel.lengthscale': torch.tensor(1.0),
    'covar_module.outputscale': torch.tensor(1.0)
    # 'i_covar_module.outputscale': torch.tensor(0.1**2),
    }

    model.initialize(**hypers)
    
    # sample from prior
    with torch.no_grad(), gpytorch.settings.fast_computations(
        covar_root_decomposition=False, log_prob=False, solves=False):
        scores = likelihood(model(torch.Tensor(xs))).sample()

    return xs, scores.detach().numpy()
    