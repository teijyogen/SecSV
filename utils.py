import torch
from torch.utils.data import Dataset
import numpy as np
import sys
import tenseal as ts
import time
import random
import copy
import math
import os

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def make_dirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

class H5Dataset(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, data_tensor, target_tensor):
        assert data_tensor.shape[0] == target_tensor.shape[0]
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        # print(index)
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.shape[0]

def communicate(object, speed=1073741824):
    size = sys.getsizeof(object) * 8
    time = size / speed

    return time


def sigma(matrix):
    perm_mat = copy.deepcopy(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            index = (i + j) % matrix.shape[1]
            perm_mat[i, j] = matrix[i, index]

    return perm_mat

def tau(matrix):
    perm_mat = copy.deepcopy(matrix)
    for j in range(matrix.shape[1]):
        for i in range(matrix.shape[0]):
            index = (i + j) % matrix.shape[0]
            perm_mat[i, j] = matrix[index, j]

    return perm_mat

def phi(matrix):
    return np.roll(matrix, -1, axis=1)

def psi(matrix):
    return np.roll(matrix, -1, axis=0)

def get_diagonal(matrix, position):
    chunk_size = matrix.shape[0]
    size = matrix.shape[1]
    chunk_nb = int(size / chunk_size)
    diagonal = np.zeros(size)

    k = 0
    for c in range(chunk_nb):
        i = 0
        j = position + c * chunk_size
        while i < chunk_size and j < size:
            diagonal[k] = matrix[i, j]
            i += 1
            j += 1
            k += 1

        if j == size:
            j = 0
            while i < chunk_size and j < position:
                diagonal[k] = matrix[i, j]
                i += 1
                j += 1
                k += 1
            break

    return diagonal

if __name__ == '__main__':

    y = np.array([0, -4000, 10, 20])

    y = 1 / (1 + np.exp(-y))

    print(y)
    # matrix = np.array([i + 1 for i in range(7840)]).reshape(10, 784)
    # vec = np.random.randn(5, 784).T.reshape(-1)
    # print(communicate(vec) * 14)
    #
    # start = time.process_time()
    # for i in range(14):
    #     rotated = np.roll(vec, -i * 784)
    # print(time.process_time() - start)
    #
    # print(matrix @ vec.T)
    #
    # chunk_size = matrix.shape[0]
    # size = matrix.shape[1]
    # chunk_nb = int(size / chunk_size)
    # batch_size = vec.shape[0]
    #
    # vec = vec.T.reshape(-1)
    #
    # results = []
    # for i in range(chunk_size):
    #     diagonal = np.tile(get_diagonal(matrix, i).reshape(-1, 1), (1, batch_size)).reshape(-1)
    #     # print(diagonal)
    #     rotated = np.roll(vec, -i * batch_size)
    #     # print(rotated)
    #     results.append(diagonal * rotated)
    #
    # for i in range(chunk_size * chunk_nb, size):
    #     diagonal = np.tile(get_diagonal(matrix, i).reshape(-1, 1), (1, batch_size)).reshape(-1)
    #     rotated = np.roll(vec, -i * batch_size)
    #     # print(rotated)
    #     results.append(diagonal * rotated)
    #
    # result = np.sum(results, axis=0)
    # copy = copy.deepcopy(result)
    #
    # double_times = int(math.log(size / chunk_size, 2))
    # for k in range(double_times):
    #     result = result + np.roll(result, -chunk_size * batch_size * 2 ** k)
    #
    # for j in range(2 ** double_times, int(size / chunk_size)):
    #     result += np.roll(copy, -j * chunk_size * batch_size)
    #
    # print(result.reshape(-1, batch_size)[:chunk_size, :])







