import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

def main(args):
    NUM_FEATURES = int(args["feature"])
    NUM_POINTS = int(args["datapoint"])
    SEED = int(args["seed"])
    BUDGET = int(args["budget"])
    BATCH_SIZE = int(args["batch"])
    POLICY = int(args["policy"])
    num_batch = int(BUDGET/BATCH_SIZE)
    ACC = np.zeros((num_batch,POLICY+1, SEED))
    RHO = np.zeros((num_batch,POLICY+1, SEED))

    for p in range(POLICY+1):
        for seed in range(SEED):
            result_filename = "./results/f" + str(NUM_FEATURES) + "d" + str(NUM_POINTS) \
            + "s" + str(seed+1) + "B" + str(BUDGET) + "b" + str(BATCH_SIZE) + "p" + str(p) + ".csv"
            result = pd.read_csv(result_filename, index_col=False)
            ACC[:,p,seed] = result["acc"]
            RHO[:,p,seed] = result["rho"]
    
    ACC = np.mean(ACC, axis=2)
    RHO = np.mean(RHO, axis=2)

    POLICY_NAMES = ["Random", "Bald"]
    fig = plt.figure(figsize=(8.00,6.00))
    for p in range(POLICY+1):
        plt.plot(np.arange(num_batch), ACC[:,p], label=POLICY_NAMES[p])
    plt.ylabel("Accurary")
    plt.xlabel("num of batches")
    plt.legend()
    fig.savefig("./results/f" + str(NUM_FEATURES) + "d" + str(NUM_POINTS) \
         + "B" + str(BUDGET) + "b" + str(BATCH_SIZE) + "acc.eps", format='eps', dpi=1200)
    plt.close()

    fig = plt.figure(figsize=(8.00,6.00))
    for p in range(POLICY+1):
        plt.plot(np.arange(num_batch), RHO[:,p], label=POLICY_NAMES[p])
    plt.ylabel("Correlation")
    plt.xlabel("num of batches")
    plt.legend()
    fig.savefig("./results/f" + str(NUM_FEATURES) + "d" + str(NUM_POINTS) \
         + "B" + str(BUDGET) + "b" + str(BATCH_SIZE) + "rho.eps", format='eps', dpi=1200)
    plt.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='-f num_feature -d num_datapoints -s SEED -B budget -b batch_size -p policy')
    parser.add_argument('-f','--feature', help='number of features', required=True)
    parser.add_argument('-d','--datapoint', help='number of data', required=True)
    parser.add_argument('-s','--seed', help='random seed', required=True)
    parser.add_argument('-B','--budget', help='total budget', required=True)
    parser.add_argument('-b','--batch', help='batch size', required=True)
    parser.add_argument('-p','--policy', help='policy: 0 random 1 bald', required=True)
    args = vars(parser.parse_args())
    main(args)
