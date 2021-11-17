import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.stats import norm, kendalltau, entropy
from utils.data import generate_data
from utils.policy import select_next_batch


NUM_FEATURES = 2
NUM_POINTS = 100
SEED = 5132020
NUM_CODERS = 10
USE_CODER_QUALITY = False
BUDGET = 50
BATCH_SIZE = 5

# NUM_POINTS = 100
# BUDGET = 500
# BATCH_SIZE = 20

def label_pair(s1, s2, coder_quality=1):
    '''
    Label a pair with coder quality.

    Math: 
        p = ncdf(coder*(s1-s2))
        label = 2*Bernouli(p) - 1
    '''
    p = norm.cdf(coder_quality*(s1-s2))
    label = np.random.binomial(n=1, p=p)
    return label

def main():
    xs, scores = generate_data(num_features=NUM_FEATURES, id_flag=True, num_points=NUM_POINTS, SEED=SEED)
    # plt.scatter(xs[:,1], scores, color='black', s=4, marker='o')
    # plt.show()

    idxs = [i for i in range(NUM_POINTS)]
    pairs = np.array(list(itertools.combinations(idxs, 2)))

    # data: [i, x_i, j, x_j]
    data = np.hstack((pairs[:,0].reshape((-1,1)), xs[pairs[:,0],1:],
                 pairs[:,1].reshape((-1,1)), xs[pairs[:,1],1:]))

    # data = np.hstack((xs[pairs[:,0]], xs[pairs[:,1]]))

    # anchor data
    anchor_data = np.hstack((xs,-np.ones((NUM_POINTS,1)), -100+np.zeros((NUM_POINTS,1))))
    # anchor_data = np.hstack((xs, np.zeros((NUM_POINTS,1))))

    # coder quality
    # model adversarial coders by allowing coder quality to be negative
    if USE_CODER_QUALITY:
        coder_qualities = np.exp(np.random.normal(loc=0, scale=0.5, size=NUM_CODERS))
    else:
        coder_qualities = np.ones((NUM_CODERS,))

    # true labels
    y_star = (scores[pairs[:,0]]>scores[pairs[:,1]])

    # initialize labels
    query_idxs = np.random.choice(data.shape[0], BATCH_SIZE, replace=False)
    query_space = np.ones((data.shape[0],),dtype=bool)

    train_x = np.zeros((0,2*NUM_FEATURES))
    train_y = np.zeros((0,))

    num_batch = int(BUDGET/BATCH_SIZE)
    ACC = np.zeros((num_batch,))
    RHO = np.zeros((num_batch,))
    TAU = np.zeros((num_batch,))

    for k in range(num_batch):
        # query labels from coders
        train_y_tmp = np.zeros((BATCH_SIZE,))
        for i in range(BATCH_SIZE):
            train_y_tmp[i] = label_pair(scores[pairs[query_idxs[i],0]],
                            scores[pairs[query_idxs[i], 1]], coder_quality=1)

        # train_x, train_y
        train_x = np.vstack((train_x, data[query_idxs]))
        train_y = np.hstack((train_y, train_y_tmp))

        print(train_y.shape)

        query_space[query_idxs] = False

        # select next batch of pairs to query
        # random query
        # query_idxs = np.random.choice(data.shape[0], BATCH_SIZE, replace=False)

        mus, zs, query_idxs = select_next_batch(train_x, train_y, 
                        data, BATCH_SIZE, query_space, anchor_data)

        # evaluate current iteration
        ACC[k] = np.mean((mus>0) == (y_star==1))
        RHO[k] = np.corrcoef(zs, scores)[0,1]
        TAU[k] = kendalltau(zs, scores)[0]

    plt.plot(np.arange(num_batch), ACC, label="acc")
    plt.plot(np.arange(num_batch), RHO, label="rho")
    plt.plot(np.arange(num_batch), TAU, label="tau")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()
    # x = np.linspace(-2,2,num=100)
    # phi = norm.cdf(x)
    # h1 = -phi*np.log2(phi) - (1-phi)*np.log2(1-phi)
    # h2 = np.exp(-x**2/3.1415926/np.log(2))
    # plt.plot(x, h1, label="truth")
    # plt.plot(x, h2, label="approximation")
    # plt.legend()
    # plt.show()

