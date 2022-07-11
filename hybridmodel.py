import torch
from torchvision import datasets
import torchvision.transforms as transforms
import tenseal as ts
import math
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset
import time
from utils import *
import copy
import random
from collections import Counter
from random128 import random128
# import tenseal.sealapi as sealapi
import gc
from hemodel import EncModel
import skimage.measure
np.seterr(all="ignore")


class HybridModel(EncModel):
    def __init__(self, gamma1=16, gamma2=111):
        super(HybridModel, self).__init__()
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.modulus_size = 1 + self.gamma1 + self.gamma2

    def prepare_shares_for_conv(self, x, windows_nb, kernel_len, stride, pad_width=((0, 0), (0, 0), (0, 0)), in_channels_nb=1, vir_channels_nb=1):
        share1, share2 = self.generate_shares(x)
        self.time_dict["communication"] += communicate(share1) + communicate(share2)

        processed_share1 = self.preprocess_for_conv(share1, windows_nb, kernel_len, stride, pad_width, in_channels_nb, vir_channels_nb)


        start = time.process_time()
        processed_share2 = self.preprocess_for_conv(share2, windows_nb, kernel_len, stride, pad_width, in_channels_nb, vir_channels_nb)
        self.time_dict["total"] -= time.process_time() - start

        return (processed_share1, processed_share2)

    def preprocess_for_conv(self, x, windows_nb, kernel_len, stride, pad_width=((0, 0), (0, 0), (0, 0)), in_channels_nb=1, vir_channels_nb=1):

        dk = int(windows_nb ** 0.5)
        padded_x = np.pad(x, pad_width)

        enc_features_ic = []
        for ic in range(in_channels_nb):
            x_ic = padded_x[ic]
            enc_features = []
            for i in range(kernel_len):
                for j in range(kernel_len):
                    feature = np.zeros((dk, dk, self.input_nb))
                    for i_prime in range(dk):
                        for j_prime in range(dk):
                            feature[i_prime, j_prime, :] = x_ic[stride * i_prime + i, stride * j_prime + j, :].reshape(
                                1, 1, self.input_nb)

                    feature = feature.reshape(-1, self.input_nb)
                    chunk_size = math.ceil(feature.shape[0] / vir_channels_nb)
                    added_rows_nb = chunk_size * vir_channels_nb - feature.shape[0]
                    padded_feature = np.pad(feature, ((0, added_rows_nb), (0, 0)))

                    enc_features_vc = []
                    for vc in range(vir_channels_nb):
                        plain_feature_vc = self.plaintext(padded_feature[vc*chunk_size:(vc+1)*chunk_size, :])
                        enc_features_vc.append(plain_feature_vc)

                    enc_features.append(enc_features_vc)

            enc_features_ic.append(enc_features)

        enc_features_ic = np.array(enc_features_ic)
        enc_features_vc = np.transpose(enc_features_ic, (2, 0, 1))

        return enc_features_vc.tolist()

    def sec_conv(self, enc_conv, shares):
        enc_y_channel = []
        share1, share2 = shares
        for (enc_wt, enc_bias) in enc_conv:
            enc_y_vc = []
            for vc in range(len(share1)):
                start = time.process_time()
                enc_y_share1 = self.big_mat_dotmul(enc_wt[0], share1[vc][0])
                for ic in range(1, len(enc_wt)):
                    enc_y_share1 += self.big_mat_dotmul(enc_wt[ic], share1[vc][ic])
                self.time_dict["total"] -= time.process_time() - start

                enc_y_share1 = self.send_enc_vector(enc_y_share1)

                start = time.process_time()
                enc_y_share2 = self.big_mat_dotmul(enc_wt[0], share2[vc][0])
                for ic in range(1, len(enc_wt)):
                    enc_y_share2 += self.big_mat_dotmul(enc_wt[ic], share2[vc][ic])
                enc_y = enc_y_share1 + enc_y_share2 + enc_bias
                self.time_dict["HE computation"] += time.process_time() - start

                enc_y = self.send_enc_vector(enc_y)
                enc_y_vc.append(enc_y)

            enc_y_channel.append(enc_y_vc)

        return enc_y_channel

    def encrypt_fc(self, fc_weight, fc_bias, vir_channels_nb):
        fc_weight = fc_weight.numpy()
        fc_bias = fc_bias.numpy()

        chunk_size = math.ceil(fc_weight.shape[0] / vir_channels_nb)
        added_rows_nb = chunk_size * vir_channels_nb - fc_weight.shape[0]
        padded_weight = np.pad(fc_weight, ((0, added_rows_nb), (0, 0)))
        padded_bias = np.pad(fc_bias, (0, added_rows_nb))
        rep_bias = np.tile(padded_bias.reshape(-1, 1), (1, self.input_nb))

        enc_vc = []
        for vc in range(vir_channels_nb):
            # weight = fc_weight[vc*chunk_size:(vc+1)*chunk_size, :]
            weight = padded_weight[vc*chunk_size:(vc+1)*chunk_size, :]
            enc_wts = self.enc_perm_mats(weight)

            bias = rep_bias[vc*chunk_size:(vc+1)*chunk_size, :]
            enc_bias = self.encrypt(bias.reshape(-1))
            enc_bias = self.send_enc_vector(enc_bias, nb_receviers=2)

            enc_vc.append((enc_wts, enc_bias))

        return enc_vc

    def prepare_shares_for_fc(self, x, n_rows_left, vir_channles_nb=1):
        share1, share2 = self.generate_shares(x)
        self.time_dict["communication"] += communicate(share1) + communicate(share2)

        processed_share1 = self.preprocess_for_fc(share1, n_rows_left, vir_channles_nb)

        start = time.process_time()
        processed_share2 = self.preprocess_for_fc(share2, n_rows_left, vir_channles_nb)
        self.time_dict["total"] -= time.process_time() - start

        return (processed_share1, processed_share2)

    def preprocess_for_fc(self, matrix, n_rows_left, vir_channles_nb=1):
        n_subrows_left = math.ceil(n_rows_left / vir_channles_nb)
        enc_mats = self.perm_mats(matrix, n_subrows_left)
        enc_mats_vc = [enc_mats for vc in range(vir_channles_nb)]

        return enc_mats_vc

    def sec_fc(self, enc_fc, x_shares, send_back=True):

        enc_y_vc = []
        for vc in range(len(enc_fc)):
            enc_wts, enc_bias = enc_fc[vc]

            if x_shares is None:
                enc_y = enc_bias
            else:
                x_share1 = x_shares[0][vc]
                x_share2 = x_shares[1][vc]

                start = time.process_time()
                enc_y_share1 = self.he_matmul(enc_wts, x_share1)
                self.time_dict["total"] -= time.process_time() - start

                enc_y_share1 = self.send_enc_vector(enc_y_share1)


                start = time.process_time()
                enc_y_share2 = self.he_matmul(enc_wts, x_share2)
                enc_y = enc_y_share1 + enc_y_share2 + enc_bias
                self.time_dict["HE computation"] += time.process_time() - start

            if send_back:
                enc_y = self.send_enc_vector(enc_y)
            enc_y_vc.append(enc_y)

        return enc_y_vc

    def enc_perm_mats(self, matrix):
        mat_nb = matrix.shape[1]
        if self.input_nb > mat_nb:
            rep_times = math.ceil(self.input_nb / mat_nb)
            mat = np.repeat(matrix, rep_times, axis=0).reshape(matrix.shape[0], -1)
        else:
            mat = matrix
        mat = sigma(mat)
        enc_mat = self.encrypt(mat[:, :self.input_nb])
        enc_mat = self.send_enc_vector(enc_mat, nb_receviers=2)
        enc_mats = [enc_mat]

        for i in range(1, mat_nb):
            mat = phi(mat)
            enc_mat = self.encrypt(mat[:, :self.input_nb])
            enc_mat = self.send_enc_vector(enc_mat, nb_receviers=2)
            enc_mats.append(enc_mat)

        return enc_mats

    def perm_mats(self, matrix, n_rows_left):
        if matrix.shape[1] > matrix.shape[0]:
            rep_times = math.ceil(matrix.shape[1] / matrix.shape[0])
            mat = np.repeat(matrix, rep_times, axis=0).reshape(matrix.shape[0], -1)[:, :matrix.shape[1]]
        else:
            mat = matrix
        mat = tau(mat)
        plain_mat = self.plaintext(mat[:n_rows_left, :])
        # plain_mat = mat[:n_rows_left, :].reshape(-1)
        plain_mats = [plain_mat]
        # plain_mats = [mat.reshape(-1)]
        for i in range(1, mat.shape[0]):
            mat = psi(mat)
            # plain_mats.append(mat.reshape(-1))
            plain_mat = self.plaintext(mat[:n_rows_left, :])
            # plain_mat = mat[:n_rows_left, :].reshape(-1)
            plain_mats.append(plain_mat)

        return plain_mats

    def he_matmul(self, mat1_ls, mat2_ls):
        enc_y = mat1_ls[0] * mat2_ls[0]
        for i in range(1, len(mat1_ls)):
            enc_y += mat1_ls[i] * mat2_ls[i]

        return enc_y

    def encrypt_gru(self, gru_weights, gru_biases, vir_channel_nb):
        hidden_size = int(gru_weights.shape[0] / 3)
        enc_paras = []
        for i in range(3):
            weight = gru_weights[i*hidden_size:(i+1)*hidden_size]
            bias = gru_biases[i*hidden_size:(i+1)*hidden_size]
            enc_para = self.encrypt_fc(weight, bias, vir_channel_nb)
            enc_paras.append(enc_para)

        return enc_paras

    def generate_shares(self, secret):
        start = time.process_time()

        if self.modulus_size >= 64:
            secret_nb = secret.reshape(-1).shape[0]
            max_frac = np.float128(2 ** self.gamma2)
            modulus = np.float128(2 ** self.modulus_size)
            max_pos = np.float128(2 ** (self.modulus_size - 1))

            secret_int = np.float128(np.floor(np.float128(secret) * max_frac))
            secret_int = np.remainder(secret_int, modulus)

            mask_int = random128.random_array(secret_nb, self.modulus_size).reshape(secret_int.shape)
            # csprng = random.SystemRandom()
            # mask_int = [csprng.randint(0, 2 ** self.modulus_size) for i in range(secret_nb)]
            #
            # mask_int = np.array(mask_int, dtype=np.float128).reshape(secret_int.shape)

            share_int = np.remainder(secret_int - mask_int, modulus)

            mask_int[mask_int >= max_pos] = mask_int[mask_int >= max_pos] - modulus
            share_int[share_int >= max_pos] = share_int[share_int >= max_pos] - modulus

            mask = mask_int / max_frac
            share = share_int / max_frac

        else:
            secret_int = np.floor(secret*2 ** self.gamma2)
            secret_int = np.remainder(secret_int, 2 ** self.modulus_size)

            rng = np.random.default_rng()
            mask_int = rng.integers(0, high=2**self.modulus_size, size=secret_int.shape)
            share_int = np.mod(secret_int-mask_int, 2**self.modulus_size)

            mask_int[mask_int >= 2 ** (self.modulus_size - 1)] = mask_int[
                                                                mask_int >= 2 ** (self.modulus_size - 1)] - 2 ** self.modulus_size
            share_int[share_int >= 2 ** (self.modulus_size - 1)] = share_int[
                                                                share_int >= 2 ** (self.modulus_size - 1)] - 2 ** self.modulus_size

            mask = mask_int / 2 ** self.gamma2
            share = share_int / 2 ** self.gamma2

        self.time_dict["shares generation"] += time.process_time() - start

        return [share, mask]

    def predict(self, enc_y, output_size):
        y = self.decrypt(enc_y[0])
        y = y.reshape(-1, self.input_nb)[:output_size, :self.truth_nb]
        output = self.remainder(y)

        pred = np.argmax(output, axis=0)

        share1, share2 = self.generate_shares(pred)
        self.time_dict["communication"] += communicate(share1) + communicate(share2)

        return share1.reshape(-1), share2.reshape(-1)

    def sec_compare(self, pred_shares, truth_shares):
        start = time.process_time()
        compare_share1 = pred_shares[0] - truth_shares[0]
        self.time_dict["total"] -= time.process_time() - start

        self.time_dict["communication"] += communicate(compare_share1)

        compare_share2 = pred_shares[1] - truth_shares[1]
        compare = self.remainder(compare_share1 + compare_share2)
        correct_idxs = np.where(np.abs(compare) < 0.5)[0]

        return correct_idxs

    def remainder(self, secret):
        if self.modulus_size >= 64:
            secret_int = np.floor(np.float128(secret) * np.float128(2 ** self.gamma2))
            secret_int = np.remainder(secret_int, np.float128(2 ** self.modulus_size))

            secret_int[secret_int >= np.float128(2 ** (self.modulus_size - 1))] = secret_int[secret_int >= np.float128(
                2 ** (self.modulus_size - 1))] - np.float128(2 ** self.modulus_size)

            recovered_secret = secret_int / np.float128(2 ** self.gamma2)

        else:
            secret_int = np.floor(secret * 2 ** self.gamma2)
            secret_int = np.remainder(secret_int, 2 ** self.modulus_size)

            secret_int[secret_int >= 2 ** (self.modulus_size - 1)] = secret_int[
                                                                secret_int >= 2 ** (self.modulus_size - 1)] - 2 ** self.modulus_size

            recovered_secret = secret_int / 2 ** self.gamma2

        return recovered_secret.astype(np.float32)

    def cal_optimal_batch_size(self, test_size):
        allowed_sizes = self.n_slots // np.array(self.x_row_nbs).reshape(1, -1)
        batch_nbs = np.ceil(test_size / allowed_sizes)
        batch_nb_min = batch_nbs.min()
        batch_nb_max = batch_nbs.max()

        batch_nbs = np.arange(batch_nb_max, batch_nb_min-1, step=-1)
        batch_sizes = np.ceil(test_size / batch_nbs)
        batch_sizes_mat = np.repeat(batch_sizes.reshape(-1, 1), allowed_sizes.shape[0], axis=1)

        allowed_sizes, channel_nbs = self.cal_optimal_allowed_size(batch_sizes)
        cipher_nbs = np.ceil(batch_sizes_mat / allowed_sizes)
        rep_times = np.array(self.x_mat_nbs).reshape(1, -1) * channel_nbs

        cipher_nbs = (cipher_nbs * rep_times).sum(axis=1) * batch_nbs
        optimal_batch_size = int(batch_sizes[np.argmin(cipher_nbs)])

        return optimal_batch_size

    def cal_optimal_allowed_size(self, batch_size):
        batch_size = batch_size.reshape(-1, 1)
        batch_size_nb = batch_size.shape[0]

        allowed_size_layers = []
        channel_nb_layers = []
        for x_row_nb in self.x_row_nbs:
            channel_nb_candi = np.arange(x_row_nb, 0, step=-1)
            channel_nb_candi = np.tile(channel_nb_candi, (batch_size_nb, 1))
            allowed_sizes = self.n_slots // np.ceil(x_row_nb / channel_nb_candi)
            cipher_nbs = np.ceil(batch_size / allowed_sizes) * channel_nb_candi
            allowed_size = np.take_along_axis(allowed_sizes, np.argmin(cipher_nbs, axis=1).reshape(-1, 1), axis=1)
            channel_nb = np.take_along_axis(channel_nb_candi, np.argmin(cipher_nbs, axis=1).reshape(-1, 1), axis=1)

            allowed_size_layers.append(allowed_size)
            channel_nb_layers.append(channel_nb)

        allowed_size_layers = np.concatenate(allowed_size_layers, axis=1)
        channel_nb_layers = np.concatenate(channel_nb_layers, axis=1)

        return allowed_size_layers, channel_nb_layers


class Sec_MNIST_CNN(HybridModel):
    def __init__(self, input_nb=409):
        super(Sec_MNIST_CNN, self).__init__()
        self.input_nb = input_nb
        self.truth_nb = input_nb
        self.input_shape = (-1, 1, 28, 28)
        self.image_len = 28

        self.conv1_stride = 3
        self.conv1_kernel_len = 7
        self.conv1_in_channel_nb = 1
        self.conv1_out_channel_nb = 4
        self.conv1_windows_nb = 8 * 8
        self.conv1_padding = ((0, 0), (0, 0), (0, 0), (0, 0))
        self.conv1_vir_channel_nb = 1

        self.fc1_input_size = 256
        self.fc1_output_size = 64
        self.fc1_channel_nb = 1

        self.fc2_input_size = 64
        self.fc2_output_size = 10
        self.fc2_channel_nb = 1

        self.context = None
        self.enc_conv1 = None
        self.enc_fc1 = None
        self.enc_fc2 = None

        self.x_row_nbs = [self.conv1_windows_nb, self.fc1_output_size, self.fc2_output_size]
        self.x_mat_nbs = [
            self.conv1_kernel_len ** 2 * self.conv1_in_channel_nb * self.conv1_out_channel_nb,
            self.fc1_input_size,
            self.fc2_input_size,
        ]

    def init_model_paras(self, context, model_paras):
        self.context = context
        self.enc_conv1 = self.encrypt_conv(model_paras["conv1.weight"], model_paras["conv1.bias"],
                                           self.conv1_kernel_len, self.conv1_windows_nb, self.conv1_in_channel_nb, self.conv1_vir_channel_nb)
        self.enc_fc1 = self.encrypt_fc(model_paras["fc1.weight"], model_paras["fc1.bias"], self.fc1_channel_nb)
        self.enc_fc2 = self.encrypt_fc(model_paras["fc2.weight"], model_paras["fc2.bias"], self.fc2_channel_nb)

    def clear_model_paras(self):
        self.context = None
        self.enc_conv1 = None
        self.enc_fc1 = None
        self.enc_fc2 = None

    def set_input_nb(self, input_nb):
        print("Set input size:", input_nb)
        self.input_nb = input_nb
        _, channel_nbs = self.cal_optimal_allowed_size(np.array(input_nb))
        channel_nbs = channel_nbs.reshape(-1).tolist()
        self.conv1_vir_channel_nb, self.fc1_channel_nb, self.fc2_channel_nb = channel_nbs

        # print(input_nb)
        # print(self.conv1_vir_channel_nb)
        # print(self.fc1_channel_nb)
        # print(self.fc2_channel_nb)

    def sec_square1(self, enc_y):
        y_oc = []
        for oc in range(len(enc_y)):
            y_vc = []
            for vc in range(len(enc_y[0])):
                y = self.decrypt(enc_y[oc][vc])
                y_vc.append(y)
            y = np.array(y_vc).reshape(-1, self.input_nb)[:self.conv1_windows_nb, :]
            y = self.remainder(y)
            y = self.square(y)
            y_oc.append(y)
        x = np.array(y_oc).reshape(-1, self.input_nb)
        return x

    def sec_square2(self, enc_y):
        if type(enc_y) == list:
            y_list = []
            for vc in range(len(enc_y)):
                y = self.decrypt(enc_y[vc])
                y_list.append(y)
            y = np.array(y_list)
        else:
            y = self.decrypt(enc_y)

        y = y.reshape(-1, self.input_nb)[:self.fc1_output_size, :]
        y = self.remainder(y)
        x = self.square(y)
        return x

    def preprocess_input(self, x):
        x = x.reshape(self.input_nb, self.conv1_in_channel_nb, self.image_len, self.image_len)
        x = np.transpose(x, (1, 2, 3, 0))
        return self.preprocess_for_conv(x, self.conv1_windows_nb, self.conv1_kernel_len, self.conv1_stride,
                                        self.conv1_padding, self.conv1_in_channel_nb, self.conv1_vir_channel_nb)

    def forward(self, x_shares, truth_shares):

        enc_y_channel = self.sec_conv(self.enc_conv1, x_shares)
        x = self.sec_square1(enc_y_channel)


        x_shares = self.prepare_shares_for_fc(x, self.fc1_output_size, vir_channles_nb=self.fc1_channel_nb)
        enc_y = self.sec_fc(self.enc_fc1, x_shares)
        x = self.sec_square2(enc_y).reshape(self.fc2_input_size, self.input_nb)

        x_shares = self.prepare_shares_for_fc(x, self.fc2_output_size, vir_channles_nb=self.fc2_channel_nb)
        enc_y = self.sec_fc(self.enc_fc2, x_shares)
        pred_shares = self.predict(enc_y, self.fc2_output_size)

        return self.sec_compare(pred_shares, truth_shares)


# class Sec_CIFAR_CNN(HybridModel):
#     def __init__(self, input_nb=409):
#         super(Sec_CIFAR_CNN, self).__init__()
#         self.input_nb = input_nb
#         self.truth_nb = input_nb
#         self.input_shape = (-1, 3, 32, 32)
#         self.image_len = 32
#
#         self.conv1_stride = 1
#         self.conv1_kernel_len = 5
#         self.conv1_in_channel_nb = 3
#         self.conv1_out_channel_nb = 6
#         self.conv1_windows_nb = 28 * 28
#         self.conv1_padding = ((0, 0), (0, 0), (0, 0), (0, 0))
#         self.conv1_vir_channel_nb = 1
#
#         self.conv2_stride = 1
#         self.conv2_kernel_len = 5
#         self.conv2_in_channel_nb = 6
#         self.conv2_out_channel_nb = 16
#         self.conv2_windows_nb = 10 * 10
#         self.conv2_padding = self.conv1_padding
#         self.conv2_vir_channel_nb = 1
#
#         self.fc1_input_size = 400
#         self.fc1_output_size = 120
#         self.fc1_channel_nb = 1
#
#         self.fc2_input_size = 120
#         self.fc2_output_size = 84
#         self.fc2_channel_nb = 1
#
#         self.fc3_input_size = 84
#         self.fc3_output_size = 10
#         self.fc3_channel_nb = 1
#
#         self.context = None
#         self.enc_conv1 = None
#         self.enc_conv2 = None
#         self.enc_fc1 = None
#         self.enc_fc2 = None
#         self.enc_fc3 = None
#
#         self.x_row_nbs = [
#             self.conv1_windows_nb,
#             self.conv2_windows_nb,
#             self.fc1_output_size,
#             self.fc2_output_size,
#             self.fc3_output_size
#         ]
#         self.x_mat_nbs = [
#             self.conv1_kernel_len ** 2 * self.conv1_in_channel_nb * self.conv1_out_channel_nb,
#             self.conv2_kernel_len ** 2 * self.conv2_in_channel_nb * self.conv2_out_channel_nb,
#             self.fc1_input_size,
#             self.fc2_input_size,
#             self.fc3_input_size
#         ]
#
#     def set_input_nb(self, input_nb):
#         self.input_nb = input_nb
#         _, channel_nbs = self.cal_optimal_allowed_size(np.array(input_nb))
#         channel_nbs = channel_nbs.reshape(-1).tolist()
#         self.conv1_vir_channel_nb, self.conv2_vir_channel_nb, self.fc1_channel_nb, self.fc2_channel_nb, self.fc3_channel_nb = channel_nbs
#
#         # print(input_nb)
#         # print(self.conv1_vir_channel_nb)
#         # print(self.fc1_channel_nb)
#         # print(self.fc2_channel_nb)
#
#     def init_model_paras(self, context, model_paras):
#         self.context = context
#         self.enc_conv1 = self.encrypt_conv(model_paras["conv1.weight"], model_paras["conv1.bias"],
#                                            self.conv1_kernel_len, self.conv1_windows_nb, self.conv1_in_channel_nb, self.conv1_vir_channel_nb)
#         self.enc_conv2 = self.encrypt_conv(model_paras["conv2.weight"], model_paras["conv2.bias"],
#                                            self.conv2_kernel_len, self.conv2_windows_nb, self.conv2_in_channel_nb, self.conv2_vir_channel_nb)
#
#         self.enc_fc1 = self.encrypt_fc(model_paras["fc1.weight"], model_paras["fc1.bias"], self.fc1_channel_nb)
#         self.enc_fc2 = self.encrypt_fc(model_paras["fc2.weight"], model_paras["fc2.bias"], self.fc2_channel_nb)
#         self.enc_fc3 = self.encrypt_fc(model_paras["fc3.weight"], model_paras["fc3.bias"], self.fc3_channel_nb)
#
#     def clear_model_paras(self):
#         self.context = None
#         self.enc_conv1 = None
#         self.enc_conv2 = None
#         self.enc_fc1 = None
#         self.enc_fc2 = None
#         self.enc_fc3 = None
#
#     def sec_relu_maxpool(self, enc_y, shape, win_nb):
#         y_oc = []
#         for oc in range(len(enc_y)):
#             y_vc = []
#             for vc in range(len(enc_y[0])):
#                 y = self.decrypt(enc_y[oc][vc])
#                 y_vc.append(y)
#             y = np.array(y_vc)
#             y = y.reshape(-1, self.input_nb)[:win_nb, :]
#             y = y.reshape(shape)
#             y = self.remainder(y)
#             y = self.relu(y)
#             y = self.maxpool(y)
#             y_oc.append(y)
#         x = np.array(y_oc)
#         return x
#
#     def sec_relu(self, enc_y, out_size):
#         if type(enc_y) == list:
#             y_list = []
#             for vc in range(len(enc_y)):
#                 y = self.decrypt(enc_y[vc])
#                 y_list.append(y)
#             y = np.array(y_list)
#         else:
#             y = self.decrypt(enc_y)
#
#         y = y.reshape(-1, self.input_nb)[:out_size, :]
#         y = self.remainder(y)
#         x = self.relu(y)
#         return x
#
#     def preprocess_input(self, x):
#         x = x.reshape(self.input_nb, self.conv1_in_channel_nb, self.image_len, self.image_len)
#         x = np.transpose(x, (1, 2, 3, 0))
#         return self.preprocess_for_conv(x, self.conv1_windows_nb, self.conv1_kernel_len, self.conv1_stride,
#                                         self.conv1_padding, self.conv1_in_channel_nb, self.conv1_vir_channel_nb)
#
#     def forward(self, x_shares, truth_shares):
#
#         enc_y_channel = self.sec_conv(self.enc_conv1, x_shares)
#         x = self.sec_relu_maxpool(enc_y_channel, [int(self.conv1_windows_nb ** 0.5), int(self.conv1_windows_nb ** 0.5), self.input_nb], self.conv1_windows_nb)
#
#         x_shares = self.prepare_shares_for_conv(x, self.conv2_windows_nb, self.conv2_kernel_len, self.conv2_stride,
#                                          self.conv2_padding, self.conv2_in_channel_nb, self.conv2_vir_channel_nb)
#         enc_y_channel = self.sec_conv(self.enc_conv2, x_shares)
#         x = self.sec_relu_maxpool(enc_y_channel, [int(self.conv2_windows_nb ** 0.5), int(self.conv2_windows_nb ** 0.5), self.input_nb], self.conv2_windows_nb).reshape(self.fc1_input_size, self.input_nb)
#
#         x_shares = self.prepare_shares_for_fc(x, self.fc1_output_size, vir_channles_nb=self.fc1_channel_nb)
#         enc_y = self.sec_fc(self.enc_fc1, x_shares)
#         x = self.sec_relu(enc_y, self.fc1_output_size).reshape(self.fc2_input_size, self.input_nb)
#
#         x_shares = self.prepare_shares_for_fc(x, self.fc2_output_size, vir_channles_nb=self.fc2_channel_nb)
#         enc_y = self.sec_fc(self.enc_fc2, x_shares)
#         x = self.sec_relu(enc_y, self.fc2_output_size).reshape(self.fc3_input_size, self.input_nb)
#
#         x_shares = self.prepare_shares_for_fc(x, self.fc3_output_size, vir_channles_nb=self.fc3_channel_nb)
#         enc_y = self.sec_fc(self.enc_fc3, x_shares)
#         pred_shares = self.predict(enc_y, self.fc3_output_size)
#
#         return self.sec_compare(pred_shares, truth_shares)
#
#     def __call__(self, *args, **kwargs):
#         return self.forward(*args, **kwargs)


class Sec_mRNA_RNN(HybridModel):
    def __init__(self, input_nb=128):
        super(Sec_mRNA_RNN, self).__init__()
        self.input_nb = input_nb
        self.truth_nb = input_nb
        self.input_shape = (-1, 10, 64)

        self.gru_input_size = self.input_shape[2]
        self.gru_output_size = 32
        self.gru_channel_nb = 1
        self.seq_len = self.input_shape[1]

        self.fc_input_size = 32
        self.fc_output_size = 2
        self.fc_channel_nb = 1

        self.context = None
        self.enc_gru_ir = None
        self.enc_gru_hr = None
        self.enc_gru_iz = None
        self.enc_gru_hz = None
        self.enc_gru_in = None
        self.enc_gru_hn = None
        self.enc_fc = None

        self.x_row_nbs = [self.gru_output_size, self.fc_output_size]
        self.x_mat_nbs = [
            (self.gru_input_size + self.gru_output_size) * 3 + 3,
            self.fc_input_size,
        ]

    def set_input_nb(self, input_nb):
        self.input_nb = input_nb

    def cal_optimal_batch_size(self, test_size):
        return 128

    # def set_input_nb(self, input_nb):
    #     print("Set input size:", input_nb)
    #     self.input_nb = input_nb
    #     _, channel_nbs = self.cal_optimal_allowed_size(np.array(input_nb))
    #     channel_nbs = channel_nbs.reshape(-1).tolist()
    #     self.gru_channel_nb = channel_nbs[0]
    #     print(self.gru_channel_nb)

    def init_model_paras(self, context, model_paras):
        self.context = context
        self.enc_gru_ir, self.enc_gru_iz, self.enc_gru_in = self.encrypt_gru(model_paras["rnn.weight_ih_l0"],
                                                            model_paras["rnn.bias_ih_l0"], self.gru_channel_nb)
        self.enc_gru_hr, self.enc_gru_hz, self.enc_gru_hn = self.encrypt_gru(model_paras["rnn.weight_hh_l0"],
                                                            model_paras["rnn.bias_hh_l0"], self.gru_channel_nb)
        self.enc_fc = self.encrypt_fc(model_paras["fc.weight"], model_paras["fc.bias"], self.fc_channel_nb)

    def clear_model_paras(self):
        self.context = None
        self.enc_gru_ir = None
        self.enc_gru_hr = None
        self.enc_gru_iz = None
        self.enc_gru_hz = None
        self.enc_gru_in = None
        self.enc_gru_hn = None
        self.enc_fc = None

    def preprocess_input(self, x):
        x = x.reshape(self.input_nb, self.seq_len, self.gru_input_size)
        x = np.transpose(x, (1, 2, 0))
        inputs = []
        for i in range(self.seq_len):
            input = x[i]
            input = self.preprocess_for_fc(input, self.gru_output_size, self.gru_channel_nb)
            inputs.append(input)
        return inputs

    def sec_sigmoid(self, enc_y):
        if type(enc_y) == list:
            y = []
            for ele in enc_y:
                y.append(self.decrypt(ele))
            y = np.concatenate(y)
        else:
            y = self.decrypt(enc_y)
        y = y.reshape(-1, self.input_nb)
        y = self.remainder(y)
        x = self.sigmoid(y)
        return x

    def sec_tanh(self, enc_y):
        if type(enc_y) == list:
            y = []
            for ele in enc_y:
                y.append(self.decrypt(ele))
            y = np.concatenate(y)
        else:
            y = self.decrypt(enc_y)
        y = y.reshape(-1, self.input_nb)
        y = self.remainder(y)
        x = self.tanh(y)
        return x

    def prepare_shares_for_hadamard(self, x):
        share1, share2 = self.generate_shares(x)
        self.time_dict["communication"] += communicate(share1) + communicate(share2)

        processed_share1 = self.plaintext(share1)

        start = time.process_time()
        processed_share2 = self.plaintext(share2)
        self.time_dict["total"] -= time.process_time() - start

        return (processed_share1, processed_share2)

    def sec_hardmard_prodcut(self, enc_x1, x2_shares, send_back=True):
        start = time.process_time()
        enc_y_share1 = enc_x1 * x2_shares[0]
        self.time_dict["total"] -= time.process_time() - start

        enc_y_share1 = self.send_enc_vector(enc_y_share1)

        start = time.process_time()
        enc_y_share2 = enc_x1 * x2_shares[1]
        enc_y = enc_y_share1 + enc_y_share2
        self.time_dict["HE computation"] += time.process_time() - start

        if send_back:
            enc_y = self.send_enc_vector(enc_y)

        return enc_y

    def compute_enc_gru_r(self, x, h):
        enc_r = self.sec_fc(self.enc_gru_ir, x, send_back=False)[0]
        enc_r += self.sec_fc(self.enc_gru_hr, h, send_back=False)[0]
        enc_r = self.send_enc_vector(enc_r)
        r = self.sec_sigmoid(enc_r)
        r_shares = self.prepare_shares_for_hadamard(r)
        return r_shares

    def compute_enc_gru_z(self, x, h):
        enc_z = self.sec_fc(self.enc_gru_iz, x, send_back=False)[0]
        enc_z += self.sec_fc(self.enc_gru_hz, h, send_back=False)[0]
        enc_z = self.send_enc_vector(enc_z)
        z = self.sec_sigmoid(enc_z)
        enc_z = self.encrypt(z)
        enc_z = self.send_enc_vector(enc_z)
        return enc_z

    def compute_enc_gru_n(self, x, h, r):
        enc_n = self.sec_fc(self.enc_gru_in, x, send_back=False)[0] \
                + self.sec_hardmard_prodcut(self.sec_fc(self.enc_gru_hn, h, send_back=False)[0], r, send_back=False)
        enc_n = self.send_enc_vector(enc_n)
        n = self.sec_tanh(enc_n)
        n_shares = self.prepare_shares_for_hadamard(n)

        return n_shares

    def compute_enc_gru_h(self, z, n, h):
        one_minus_z = self.plaintext(np.ones(z.size())) - z
        enc_h = self.sec_hardmard_prodcut(one_minus_z, n, send_back=False)
        if not (h is None):
            enc_h += self.sec_hardmard_prodcut(z, h, send_back=False)
        enc_h = self.send_enc_vector(enc_h)
        return enc_h

    def sec_rnn_gru(self, x_shares_seq, h_shares_fc=None, h_shares_hardmard=None):
        h = None
        for i in range(self.seq_len):
            x_shares = (x_shares_seq[0][i], x_shares_seq[1][i])
            r_shares = self.compute_enc_gru_r(x_shares, h_shares_fc)
            enc_z = self.compute_enc_gru_z(x_shares, h_shares_fc)
            n_shares = self.compute_enc_gru_n(x_shares, h_shares_fc, r_shares)
            enc_h = self.compute_enc_gru_h(enc_z, n_shares, h_shares_hardmard)
            h = self.decrypt(enc_h).reshape(-1, self.input_nb)

            if i < self.seq_len - 1:
                h_shares_fc = self.prepare_shares_for_fc(h, h.shape[0])
                h_shares_hardmard = self.prepare_shares_for_hadamard(h)

        return h

    def forward(self, x_shares_seq, truth_shares):


        # h_shares = self.gru_init_h_shares
        # print(id(h_shares))
        h = self.sec_rnn_gru(x_shares_seq)

        x_shares = self.prepare_shares_for_fc(h, self.fc_output_size, vir_channles_nb=self.fc_channel_nb)
        enc_y = self.sec_fc(self.enc_fc, x_shares)
        pred_shares = self.predict(enc_y, self.fc_output_size)

        return self.sec_compare(pred_shares, truth_shares)


class Sec_AGNEWS_Logi(HybridModel):
    def __init__(self, input_nb=1024):
        super(Sec_AGNEWS_Logi, self).__init__()
        self.input_nb = input_nb
        self.truth_nb = input_nb
        self.input_shape = (-1, 300)

        self.fc_input_size = self.input_shape[1]
        self.fc_output_size = 4
        self.fc_channel_nb = 1

        self.context = None
        self.enc_fc = None

        self.x_row_nbs = [self.fc_output_size]
        self.x_mat_nbs = [
            self.fc_input_size,
        ]

    def init_model_paras(self, context, model_paras):
        self.context = context
        self.enc_fc = self.encrypt_fc(model_paras["fc.weight"], model_paras["fc.bias"], self.fc_channel_nb)

    def clear_model_paras(self):
        self.context = None
        self.enc_fc = None

    def set_input_nb(self, input_nb):
        print("Set input size:", input_nb)
        self.input_nb = input_nb
        _, channel_nbs = self.cal_optimal_allowed_size(np.array(input_nb))
        channel_nbs = channel_nbs.reshape(-1).tolist()
        self.fc_channel_nb = channel_nbs[0]

    def preprocess_input(self, x):
        x = x.reshape(self.input_shape)
        x = np.transpose(x, (1, 0))
        return self.preprocess_for_fc(x, self.fc_output_size)

    def forward(self, x_shares, truth_shares):

        enc_y = self.sec_fc(self.enc_fc, x_shares)
        pred_shares = self.predict(enc_y, self.fc_output_size)

        return self.sec_compare(pred_shares, truth_shares)


class Sec_BANK_Logi(HybridModel):
    def __init__(self, input_nb=2048):
        super(Sec_BANK_Logi, self).__init__()
        self.input_nb = input_nb
        self.truth_nb = input_nb
        self.input_shape = (-1, 48)

        self.fc_input_size = self.input_shape[1]
        self.fc_output_size = 2
        self.fc_channel_nb = 1

        self.context = None
        self.enc_fc = None

        self.x_row_nbs = [self.fc_output_size]
        self.x_mat_nbs = [
            self.fc_input_size,
        ]

    def init_model_paras(self, context, model_paras):
        self.context = context
        self.enc_fc = self.encrypt_fc(model_paras["fc.weight"], model_paras["fc.bias"], self.fc_channel_nb)

    def clear_model_paras(self):
        self.context = None
        self.enc_fc = None

    def set_input_nb(self, input_nb):
        print("Set input size:", input_nb)
        self.input_nb = input_nb
        _, channel_nbs = self.cal_optimal_allowed_size(np.array(input_nb))
        channel_nbs = channel_nbs.reshape(-1).tolist()
        self.fc_channel_nb = channel_nbs[0]

    def preprocess_input(self, x):
        x = x.reshape(self.input_shape)
        x = np.transpose(x, (1, 0))
        return self.preprocess_for_fc(x, self.fc_output_size)

    def forward(self, x_shares, truth_shares):

        enc_y = self.sec_fc(self.enc_fc, x_shares)
        pred_shares = self.predict(enc_y, self.fc_output_size)

        return self.sec_compare(pred_shares, truth_shares)



