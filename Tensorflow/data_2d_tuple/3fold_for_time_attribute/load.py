import numpy as np
np.set_printoptions(threshold=np.inf)

if __name__ == "__main__":
    loadData = np.load('part1_test.npy')
    print(len(loadData))
