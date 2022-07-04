import torch
from torch.utils.data import Dataset
import numpy as np
import sys
import tenseal as ts
import time
import random
import copy

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
    perm_mat = copy.deepcopy(matrix)
    for j in range(matrix.shape[1]):
        index = (j + 1) % matrix.shape[1]
        perm_mat[:, j] = matrix[:, index]

    return perm_mat

def psi(matrix):
    perm_mat = copy.deepcopy(matrix)
    for i in range(matrix.shape[0]):
        index = (i + 1) % matrix.shape[0]
        perm_mat[i, :] = matrix[index, :]

    return perm_mat
