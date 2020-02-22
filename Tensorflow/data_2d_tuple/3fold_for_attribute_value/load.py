import numpy as np
np.set_printoptions(threshold=np.inf)

if __name__ == "__main__":
    loadData = np.load('part1_train.npy')
    print(len(loadData))