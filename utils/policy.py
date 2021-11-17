import torch
from utils.model import PreferenceModel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO
import numpy as np
from scipy.stats import norm

training_iterations = 100

def h(t):
    return -t*np.log2(t) - (1-t)*np.log2(1-t)

def BALD(mus, s2s):
    C = np.sqrt(np.pi*np.log(2)/2)
    phi = norm.cdf(mus/np.sqrt(s2s+1))
    return h(phi) - C/np.sqrt(s2s+C**2)*np.exp(-mus**2/2/(s2s+C**2))


def select_next_batch(train_x, train_y, 
            data, BATCH_SIZE,  query_space, anchor_data):
    '''
    Select query idx for next batch from data.
    '''
    torch.set_default_dtype(torch.float64)
    train_x = torch.from_numpy(train_x)
    test_x = torch.from_numpy(data)
    anchor_data = torch.from_numpy(anchor_data)
    train_y = torch.from_numpy(train_y)
    model = PreferenceModel(train_x)
    likelihood = BernoulliLikelihood()

     # specify hyperparameters in a cell
    hypers = {
    'w_covar_module.base_kernel.lengthscale': torch.tensor(1.0),
    'w_covar_module.outputscale': torch.tensor(1.0),
    'i_covar_module.outputscale': torch.tensor(0.1**2),
    }
    model.initialize(**hypers)

    model.train()
    likelihood.train()

    # model.i_covar_module.raw_outputscale.requires_grad = False
    # model.w_covar_module.base_kernel.raw_lengthscale.requires_grad = False

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # "Loss" for variational GP
    mll = VariationalELBO(likelihood, model, train_y.numel())

    for i in range(training_iterations):
        # Zero backpropped gradients from previous iteration
        optimizer.zero_grad()
        # Get predictive output
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        if (i+1) % 50 == 0:
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()

    model.eval()
    likelihood.eval()

    with torch.no_grad():
        # Test x are all pairs in remaining data
        observed_pred = model(test_x)
        mus = observed_pred.mean.numpy()
        s2s = observed_pred.variance.numpy()
        BALD_scores = BALD(mus, s2s)
        query_idxs = []
        i = 1
        sorted_idx = np.argsort(BALD_scores)
        while len(query_idxs)<BATCH_SIZE:
            idx = sorted_idx[-i]
            if query_space[idx]:
                query_idxs.append(idx)
            i = i + 1

        # get latent score with anchor data x=(-1,0)
        observed_pred = model(anchor_data)
        zs = observed_pred.mean.numpy()
        zs = zs - np.mean(zs)

        # anchor_data = torch.zeros((100,2))
        # anchor_data[:,0] = torch.linspace(-3,3,100)
        # anchor_data[:,1] = torch.arange(-2000, -1000, 10)
        # observed_pred = model(anchor_data)
        # ss = observed_pred.mean.numpy()
        s2s = observed_pred.variance.numpy()

        # import matplotlib.pyplot as plt
        # XTICKS = anchor_data[:,1].numpy()
        # reindex = np.argsort(XTICKS)
        # plt.plot(XTICKS[reindex], zs[reindex])
        # plt.plot(XTICKS[reindex], zs[reindex]+2*np.sqrt(s2s[reindex]))
        # plt.plot(XTICKS[reindex], zs[reindex]-2*np.sqrt(s2s[reindex]))
        # plt.show()

    return mus, zs, np.array(query_idxs).reshape((-1,))
