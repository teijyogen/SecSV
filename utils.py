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

def make_dir(file_dir):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

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

def communicate(data, speed=134217728):
    if type(data) == np.ndarray:
        size = data.nbytes
    else:
        size = sys.getsizeof(data)
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






