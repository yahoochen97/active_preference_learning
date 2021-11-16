import torch
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ZeroMean
from gpytorch.kernels import Kernel, ScaleKernel, RBFKernel


class IndicatorKernel(Kernel):
    '''
    Indicator covariance function, K(x,z)=1 iff x==z else 0.
    '''
    is_stationary = True
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # the indicator kernel is stationary

    def forward(self, x1, x2, **params):
        # calculate the distance between inputs
        diff = self.covar_dist(x1, x2, **params)
        return (diff<=1e-8)


class PreferenceKernel(Kernel):
    '''
    Preference covariance function:
    k[(x_1,x_2),(x_3,x_4)]=k[x_1,x_3]+k[x_2,x_4]-k[x_1,x_4]-k[x_2,x_3]
    '''
    is_stationary = False
    def __init__(self, base_kernel, **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel

    def forward(self, pair1, pair2, last_dim_is_batch=False, diag=False, **params):
        ndim = int(pair1.shape[-1]/2)
        x1 = pair1[:,0:ndim]
        x2 = pair1[:,ndim:]
        x3 = pair2[:,0:ndim]
        x4 = pair2[:,ndim:]
        # x1, x2 = pair1
        # print(x1)
        # x3, x4 = pair2
        K13 = self.base_kernel.forward(x1, x3, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
        K24 = self.base_kernel.forward(x2, x4, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
        K14 = self.base_kernel.forward(x1, x4, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
        K23 = self.base_kernel.forward(x2, x3, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
        
        return (K13 + K24 - K14 - K23)


class PreferenceModel(ApproximateGP):
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x,
                    variational_distribution, learn_inducing_locations=True)
        super(PreferenceModel, self).__init__(variational_strategy)
        self.mean_module = ZeroMean()
        self.covar_module = PreferenceKernel(ScaleKernel(RBFKernel()))
        # self.i_covar_module = ScaleKernel(IndicatorKernel())
        # self.covar_module = PreferenceKernel(self.w_covar_module+self.i_covar_module)

    def forward(self, train_x):
        mean_x = self.mean_module(train_x)
        covar_x = self.covar_module(train_x)
        latent_pred = MultivariateNormal(mean_x, covar_x)
        return latent_pred

