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
import matplotlib.pyplot as plt
from multiprocessing import shared_memory

def share_data(data, name):
    unlink_shared_data(name)

    if type(data) != list:
        data = [data]
        shape = None
    else:
        data = np.array(data)
        shape = data.shape
        data = data.reshape(-1).tolist()
    shm_list = shared_memory.ShareableList(data, name=name)
    shm = shm_list.shm
    shm.close()

    return shape

def get_shared_data(name, context, shape=None):
    shm_list = shared_memory.ShareableList(name=name)
    if len(shm_list) == 1:
        data = shm_list[0]
        data = ts.CKKSVector.load(context, data)
    else:
        data = [ts.CKKSVector.load(context, entry) for entry in shm_list]

    if shape is not None:
        data = np.array(data).reshape(shape).tolist()

    shm_list.shm.close()

    return data

def share_context(context, name="context", save_secret_key=True):
    unlink_shared_data(name)
    context_bytes = context.serialize(save_secret_key=save_secret_key)
    shm = shared_memory.SharedMemory(create=True, size=len(context_bytes), name=name)
    shm.buf[:] = context_bytes
    shm.close()

    return context_bytes

def get_shared_context(name="context"):
    shm = shared_memory.SharedMemory(name=name)
    context = ts.Context.load(bytes(shm.buf))
    return context

def unlink_shared_data(name):
    try:
        shm = shared_memory.SharedMemory(name)
        shm.close()
        shm.unlink()
    except:
        return

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

LEGEND_FONT = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 10,
         }

LABEL_FONT = {'family': 'Times New Roman',
         'weight': 'black',
         'size': 24,
         }

TITLE_FONT = {'family': 'Times New Roman',
         'weight': 'black',
         'size': 18,
         }

def plot(x_list, y_list, labels, title, file_name, xlabel, ylabel, yscale='linear', xscale="linear"):
    market_color_ls = [
        ('v', 'b', 'solid', 0, 'b'),
        ('h', 'g', 'solid', 1, 'g'),
        ('d', 'm', 'solid', 2, 'm'),
        ('^', 'r', 'solid', 3, 'r'),
        ('o', 'c', 'solid', 4, 'c'),
        ('s', 'k', 'solid', 5, 'k'),
        ('*', 'y', 'solid', 6, 'y'),
        ('X', 'olive', 'solid', 7, 'olive')
    ]

    if xscale == "log":
        fig, ax = plt.subplots()
        ax.set_xscale('log', base=2)
        ax.set_yscale(yscale)
        for i in range(len(x_list)):
            ax.plot(x_list[i], y_list[i], label=labels[i], marker=market_color_ls[i][0], color=market_color_ls[i][1], alpha=1.0)
    else:
        for i in range(len(x_list)):
            plt.plot(x_list[i], y_list[i], label=labels[i], marker=market_color_ls[i][0], color=market_color_ls[i][1],
                     linestyle=market_color_ls[i][2], zorder=market_color_ls[i][3], markerfacecolor=market_color_ls[i][4])


    plt.yscale(yscale)
    plt.tick_params(labelsize=8)
    plt.legend(prop=LEGEND_FONT)
    plt.xlabel(xlabel, LABEL_FONT)
    plt.ylabel(ylabel, LABEL_FONT)
    plt.title(title, TITLE_FONT)
    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['figure.dpi'] = 400
    plt.savefig(file_name, bbox_inches='tight')
    plt.show()
    plt.close()


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







