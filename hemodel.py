import torch
from torchvision import datasets
import torchvision.transforms as transforms
import tenseal as ts
import math
import numpy as np
import tenseal.sealapi as sealapi
import random
from tqdm import tqdm
from torch.utils.data import Dataset
import time
from utils import *
import copy
import random
from collections import Counter
import random128


class HEModel:
    def __init__(self):
        self.image_nb = 64
        self.measure_time = True
        self.context = None
        self.time_dict = {}

    def encrypt(self, vector):
        if self.measure_time:
            start = time.process_time()
        enc_vec = ts.ckks_vector(self.context, vector)

        if self.measure_time:
            self.time_dict["encryption"] += time.process_time() - start
        return enc_vec

    def send_enc_vector(self, enc_vec, return_bytes=False):
        enc_vec_bytes = enc_vec.serialize()
        if self.measure_time:
            self.time_dict["communication"] += communicate(enc_vec_bytes)
        if return_bytes:
            return enc_vec_bytes
        enc_vec = ts.CKKSVector.load(self.context, enc_vec_bytes)
        return enc_vec

    def encrypt_conv(self, conv_weight, conv_bias, kernel_len, conv_windows_nb):
        enc_channels = []
        for weight, bias in zip(conv_weight, conv_bias):
            flat_wt = weight.view(-1)
            enc_weights = []
            # print(flat_wt.shape)
            for i in range(kernel_len ** 2):
                rep_wt = flat_wt[i].repeat(conv_windows_nb * self.image_nb)
                enc_weight = self.encrypt(rep_wt.view(-1))
                # enc_weight = self.encrypt(flat_wt[i].view(-1))
                enc_weight = self.send_enc_vector(enc_weight)
                enc_weights.append(enc_weight)

            rep_bias = bias.view(-1).repeat(conv_windows_nb * self.image_nb)
            enc_bias = self.encrypt(rep_bias.view(-1))
            # enc_bias = self.encrypt(bias.view(-1))
            enc_bias = self.send_enc_vector(enc_bias)
            enc_channels.append((enc_weights, enc_bias))

        return enc_channels

    def encrypt_fc(self, fc_weight, fc_bias, channel_nb):
        enc_channels = []
        chunk_size = int(fc_weight.shape[1] / channel_nb)
        for c in range(channel_nb):
            weight = fc_weight[:, c * chunk_size:(c + 1) * chunk_size]
            enc_channels.append(self.enc_perm_mats(weight, weight.shape[0], left=True))

        rep_bias = fc_bias.view(-1, 1).repeat(1, self.image_nb)
        enc_bias = self.encrypt(rep_bias.view(-1))
        enc_bias = self.send_enc_vector(enc_bias)

        return (enc_channels, enc_bias)

    def enc_perm_mats(self, matrix, mat_nb, left=True, return_bytes=False):
        nrows = matrix.shape[0]
        ncols = matrix.shape[1]

        if nrows == ncols:
            square_mat = matrix
        else:
            mat_len = max(nrows, ncols)
            subdim = min(nrows, ncols)
            square_mat = np.zeros((mat_len, mat_len))

            if nrows < ncols:
                for i in range(int(mat_len / subdim)):
                    for j in range(subdim):
                        for k in range(mat_len):
                            square_mat[i * subdim + j, k] = matrix[j, k]
            else:
                for i in range(int(mat_len / subdim)):
                    for j in range(subdim):
                        for k in range(mat_len):
                            square_mat[k, i * subdim + j] = matrix[k, j]
        if left:
            mat = sigma(square_mat)
            enc_mat = self.encrypt(mat.reshape(-1))
            enc_mat = self.send_enc_vector(enc_mat, return_bytes=return_bytes)
            enc_mats = [enc_mat]

            for i in range(1, mat_nb):
                mat = phi(mat)
                enc_mat = self.encrypt(mat.reshape(-1))
                enc_mat = self.send_enc_vector(enc_mat, return_bytes=return_bytes)
                enc_mats.append(enc_mat)
        else:
            mat = tau(square_mat)
            enc_mat = self.encrypt(mat.reshape(-1))
            enc_mat = self.send_enc_vector(enc_mat, return_bytes=return_bytes)
            enc_mats = [enc_mat]

            for i in range(1, mat_nb):
                mat = psi(mat)
                enc_mat = self.encrypt(mat.reshape(-1))
                enc_mat = self.send_enc_vector(enc_mat, return_bytes=return_bytes)
                enc_mats.append(enc_mat)

        return enc_mats

    def he_matmul(self, mat1_ls, mat2_ls):
        subdim = len(mat1_ls)
        mat_len = int(mat1_ls[0].size() ** 0.5)

        enc_y = mat1_ls[0] * mat2_ls[0]

        for i in range(1, subdim):
            enc_y += mat1_ls[i] * mat2_ls[i]

        if subdim == mat_len and mat_len == self.image_nb:
            return enc_y

        if subdim < mat_len:
            y = np.array(enc_y.decrypt()).reshape(mat_len, mat_len)
            true_y = copy.deepcopy(y)[0 * subdim:(0 + 1) * subdim, :]
            for j in range(1, int(mat_len / subdim)):
                true_y += y[j * subdim:(j + 1) * subdim, :]
            true_y = true_y[:, :self.image_nb]
            enc_y = ts.ckks_vector(self.context, true_y.reshape(-1))
        else:
            true_y = np.array(enc_y.decrypt()).reshape(mat_len, mat_len)
            true_y = true_y[:, :self.image_nb]
            enc_y = ts.ckks_vector(self.context, true_y.reshape(-1))

        return enc_y

    def sec_conv(self, enc_conv, enc_features):
        enc_y_channel = []
        for (enc_wt, enc_bias) in enc_conv:
            if self.measure_time:
                start = time.process_time()

            enc_y = enc_features[0] * enc_wt[0]
            for i in range(1, len(enc_features)):
                enc_y += enc_features[i] * enc_wt[i]
            enc_y += enc_bias

            if self.measure_time:
                self.time_dict["HE computation"] += time.process_time() - start

            enc_y = self.send_enc_vector(enc_y)
            enc_y_channel.append(enc_y)

        return enc_y_channel

    def sec_fc(self, enc_fc, enc_x, channel=False):
        if self.measure_time:
            start = time.process_time()

        if channel:
            enc_wts, enc_bias = enc_fc
            enc_y = self.he_matmul(enc_wts[0], enc_x[0])
            for i in range(1, len(enc_wts)):
                enc_y += self.he_matmul(enc_wts[i], enc_x[i])
            enc_y += enc_bias
        else:
            enc_wt = enc_fc[0][0]
            enc_bias = enc_fc[1]
            enc_y = self.he_matmul(enc_wt, enc_x)
            enc_y += enc_bias

        if self.measure_time:
            self.time_dict["HE computation"] += time.process_time() - start

        enc_y = self.send_enc_vector(enc_y)

        return enc_y

    def preprocess_for_conv(self, x, windows_nb, kernel_len, stride):

        dk = int(windows_nb ** 0.5)
        enc_features = []
        for i in range(kernel_len):
            for j in range(kernel_len):

                feature = np.zeros((dk, dk, x.shape[0]))
                for i_prime in range(dk):
                    for j_prime in range(dk):
                        feature[i_prime, j_prime, :] = x[:, stride * i_prime + i, stride * j_prime + j].view(1, 1,
                                                                                                             x.shape[0])

                enc_feature = self.encrypt(feature.reshape(-1))
                enc_feature = self.send_enc_vector(enc_feature)
                enc_features.append(enc_feature)

        return enc_features

    def decrypt(self, enc_vec):
        if self.measure_time:
            start = time.process_time()
        vec = np.array(enc_vec.decrypt())
        if self.measure_time:
            self.time_dict["decryption"] += time.process_time() - start
        return vec

    def predict(self, enc_y, output_size):

        output = self.decrypt(enc_y).reshape(output_size, self.image_nb)
        pred = np.argmax(output, axis=0)

        enc_pred = self.encrypt(pred.reshape(-1))
        enc_pred = self.send_enc_vector(enc_pred)

        return enc_pred

    def encrypt_truth(self, truth):
        enc_truth = self.encrypt(truth.view(-1))
        enc_truth_bytes = enc_truth.serialize()

        if self.measure_time:
            self.time_dict["communication"] += communicate(enc_truth_bytes)

        return enc_truth_bytes

    def sec_compare(self, enc_pred, enc_truth):
        if self.measure_time:
            start = time.process_time()
        enc_compare = enc_pred - enc_truth
        if self.measure_time:
            self.time_dict["HE computation"] += time.process_time() - start

        enc_compare = self.send_enc_vector(enc_compare)
        compare = self.decrypt(enc_compare)
        correct = (np.abs(compare) < 0.5).sum().item()
        if self.measure_time:
            self.time_dict["communication"] += communicate(correct)

        return correct


class HE_CNN1_MNIST(HEModel):
    def __init__(self, image_nb=64):
        super(HE_CNN1_MNIST, self).__init__()
        self.image_nb = image_nb
        self.image_len = 28
        self.conv1_stride = 3
        self.conv1_kernel_len = 7
        self.conv1_channel_nb = 4
        self.conv1_windows_nb = 64
        self.fc1_output_size = 64
        self.fc2_output_size = 10
        self.context = None
        self.enc_conv1 = None
        self.enc_fc1 = None
        self.enc_fc2 = None

    def init_model_paras(self, context, model_paras):
        self.context = context
        self.enc_conv1 = self.encrypt_conv(model_paras["conv1.weight"], model_paras["conv1.bias"],
                                           self.conv1_kernel_len, self.conv1_windows_nb)
        self.enc_fc1 = self.encrypt_fc(model_paras["fc1.weight"], model_paras["fc1.bias"], self.conv1_channel_nb)
        self.enc_fc2 = self.encrypt_fc(model_paras["fc2.weight"], model_paras["fc2.bias"], 1)

    def clear_model_paras(self):
        self.context = None
        self.enc_conv1 = None
        self.enc_fc1 = None
        self.enc_fc2 = None

    def encrypt_input(self, x):
        time_dict = self.time_dict
        windows_nb = self.conv1_windows_nb
        kernel_len = self.conv1_kernel_len
        stride = self.conv1_stride

        dk = int(windows_nb ** 0.5)
        enc_features_bytes = []
        for i in range(kernel_len):
            for j in range(kernel_len):

                feature = np.zeros((dk, dk, x.shape[0]))
                for i_prime in range(dk):
                    for j_prime in range(dk):
                        feature[i_prime, j_prime, :] = x[:, stride * i_prime + i, stride * j_prime + j].view(1, 1,
                                                                                                             x.shape[0])

                enc_feature = self.encrypt(feature.reshape(-1))
                enc_feature_byte = enc_feature.serialize()
                if self.measure_time:
                    time_dict["communication"] += communicate(enc_feature_byte)

                enc_features_bytes.append(enc_feature_byte)

        return enc_features_bytes

    def square_activate(self, vec):
        if self.measure_time:
            start = time.process_time()
        res = vec * vec
        if self.measure_time:
            self.time_dict["activation"] += time.process_time() - start
        return res

    def sec_activate(self, enc_y, n_rows, n_cols, mat_nb, channel=False):
        if channel:
            enc_x = []
            for i in range(len(enc_y)):
                y = self.decrypt(enc_y[i])
                # print(y.reshape(64, 8, 8)[0,:,:])
                # print("-----------------------")
                activated_y = self.square_activate(y).reshape(n_rows, n_cols)
                enc_x.append(self.enc_perm_mats(activated_y, mat_nb, left=False))
        else:
            y = self.decrypt(enc_y)
            # print(y.reshape(64, 64))
            # print("-----------------------")
            activated_y = self.square_activate(y).reshape(n_rows, n_cols)
            enc_x = self.enc_perm_mats(activated_y, mat_nb, left=False)

        return enc_x

    def forward(self, enc_features, enc_truth):

        enc_y_channel = self.sec_conv(self.enc_conv1, enc_features)

        enc_x_channel = self.sec_activate(enc_y_channel, self.conv1_windows_nb, self.image_nb, self.fc1_output_size,
                                          channel=True)

        enc_y = self.sec_fc(self.enc_fc1, enc_x_channel, channel=True)

        enc_x = self.sec_activate(enc_y, self.fc1_output_size, self.image_nb, self.fc2_output_size)

        enc_y = self.sec_fc(self.enc_fc2, enc_x)

        enc_pred = self.predict(enc_y, self.fc2_output_size)

        return self.sec_compare(enc_pred, enc_truth)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class HE_Logi_MNIST(HEModel):
    def __init__(self, image_nb=64):
        super(HE_Logi_MNIST, self).__init__()
        self.image_nb = image_nb
        self.input_size = 784
        self.fc_output_size = 10

        self.context = None
        self.enc_fc = None

    def init_model_paras(self, context, model_paras):
        self.context = context
        self.enc_fc = self.encrypt_fc(model_paras["linear.weight"], model_paras["linear.bias"], 1)

    def clear_model_paras(self):
        self.context = None
        self.enc_fc = None

    def encrypt_input(self, x):
        return self.enc_perm_mats(x.T.reshape(self.input_size, self.image_nb), self.fc_output_size, left=False,
                                  return_bytes=True)

    def sigmoid_predict(self, enc_y):
        y = self.decrypt(enc_y).reshape(self.fc_output_size, self.image_nb)

        if self.measure_time:
            start = time.process_time()
        y = 1 / (1 + np.exp(-y))
        if self.measure_time:
            self.time_dict["activation"] += time.process_time() - start

        pred = np.argmax(y, axis=0)

        enc_pred = self.encrypt(pred.reshape(-1))
        enc_pred = self.send_enc_vector(enc_pred)

        return enc_pred

    def forward(self, enc_features, enc_truth):

        enc_y = self.sec_fc(self.enc_fc, enc_features)

        enc_pred = self.sigmoid_predict(enc_y)

        return self.sec_compare(enc_pred, enc_truth)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)