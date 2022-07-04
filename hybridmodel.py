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
from hemodel_ import EncModel
import skimage.measure


class HybridModel(EncModel):
    def __init__(self, gamma1=16, gamma2=111):
        super(HybridModel, self).__init__()
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.modulus_size = 1 + self.gamma1 + self.gamma2

    # def plaintext(self, mat):
    #     return ts.plain_tensor(mat.reshape(-1), dtype='float')

    # def pre_process_for_conv(self, x, windows_nb, kernel_len, stride, pad_width=((0, 0), (0, 0), (0, 0))):
    #     dk = int(windows_nb ** 0.5)
    #     padded_x = np.pad(x, pad_width)
    #     plain_features = []
    #     for i in range(kernel_len):
    #         for j in range(kernel_len):
    #
    #             feature = np.zeros((dk, dk, padded_x.shape[0]))
    #             for i_prime in range(dk):
    #                 for j_prime in range(dk):
    #                     feature[i_prime, j_prime, :] = padded_x[:, stride*i_prime+i, stride*j_prime+j].reshape(1, 1, padded_x.shape[0])
    #
    #             plain_feature = self.plaintext(feature)
    #             plain_features.append(plain_feature)
    #
    #     return plain_features

    def prepare_shares_for_conv(self, x, windows_nb, kernel_len, stride, pad_width=((0, 0), (0, 0), (0, 0)), in_channels_nb=1):
        share1, share2 = self.generate_shares(x)
        if self.measure_time:
            self.time_dict["communication"] += communicate(share1) + communicate(share2)
        if self.measure_time:
            start = time.process_time()

        processed_share1 = self.preprocess_for_conv(share1, windows_nb, kernel_len, stride, pad_width, in_channels_nb)
        if self.measure_time:
            start = time.process_time()
        processed_share2 = self.preprocess_for_conv(share2, windows_nb, kernel_len, stride, pad_width, in_channels_nb)
        if self.measure_time:
            self.time_dict["total"] -= time.process_time() - start

        return (processed_share1, processed_share2)

    def preprocess_for_conv(self, x, windows_nb, kernel_len, stride, pad_width=((0, 0), (0, 0), (0, 0)), in_channels_nb=1):

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

                    plain_feature = self.plaintext(feature)
                    enc_features.append(plain_feature)

            enc_features_ic.append(enc_features)

        return enc_features_ic

    def sec_conv(self, enc_conv, shares):
        enc_y_channel = []
        share1, share2 = shares
        for (enc_wt, enc_bias) in enc_conv:
            if self.measure_time:
                start = time.process_time()
            enc_y_share1 = self.big_mat_dotmul(enc_wt[0], share1[0])
            for ic in range(1, len(enc_wt)):
                enc_y_share1 += self.big_mat_dotmul(enc_wt[ic], share1[ic])
            if self.measure_time:
                self.time_dict["total"] -= time.process_time() - start

            enc_y_share1 = self.send_enc_vector(enc_y_share1)

            if self.measure_time:
                start = time.process_time()
            enc_y_share2 = self.big_mat_dotmul(enc_wt[0], share2[0])
            for ic in range(1, len(enc_wt)):
                enc_y_share2 += self.big_mat_dotmul(enc_wt[ic], share2[ic])
            enc_y = enc_y_share1 + enc_y_share2 + enc_bias
            if self.measure_time:
                self.time_dict["HE computation"] += time.process_time() - start

            enc_y = self.send_enc_vector(enc_y)
            enc_y_channel.append(enc_y)

        return enc_y_channel

    def encrypt_fc(self, fc_weight, fc_bias, channel_nb):
        rep_bias = fc_bias.view(-1, 1).repeat(1, self.input_nb)
        enc_channels = []
        chunk_size = int(fc_weight.shape[1] / channel_nb)
        for vc in range(channel_nb):
            weight = fc_weight[:, vc*chunk_size:(vc+1)*chunk_size]
            enc_wts = self.enc_perm_mats(weight)
            enc_channels.append(enc_wts)

        enc_bias = self.encrypt(rep_bias.view(-1))
        enc_bias = self.send_enc_vector(enc_bias, nb_receviers=2)

        return enc_channels, enc_bias
        # rep_bias = fc_bias.view(-1, 1).repeat(1, self.input_nb)
        # enc_wts = self.enc_perm_mats(fc_weight)
        #
        # enc_bias = self.encrypt(rep_bias.view(-1))
        # enc_bias = self.send_enc_vector(enc_bias, nb_receviers=2)
        #
        # return enc_wts, enc_bias

    def prepare_shares_for_fc(self, x, n_rows_left):
        share1, share2 = self.generate_shares(x)
        if self.measure_time:
            self.time_dict["communication"] += communicate(share1) + communicate(share2)
        if self.measure_time:
            start = time.process_time()

        processed_share1 = self.preprocess_for_fc(share1, n_rows_left)
        if self.measure_time:
            start = time.process_time()
        processed_share2 = self.preprocess_for_fc(share2, n_rows_left)
        if self.measure_time:
            self.time_dict["total"] -= time.process_time() - start

        return (processed_share1, processed_share2)

    def preprocess_for_fc(self, matrix, n_rows_left):
        enc_mats = self.perm_mats(matrix, n_rows_left)

        return enc_mats

    def sec_fc(self, enc_fc, x_shares):


        enc_wts, enc_bias = enc_fc
        x_share1 = x_shares[0]
        x_share2 = x_shares[1]

        if self.measure_time:
            start = time.process_time()
        # print(self.decrypt(enc_wts[0]).shape)
        # print(x_share1[0].shape())
        enc_y_share1 = self.he_matmul(enc_wts[0], x_share1[0])
        for ic in range(1, len(x_share1)):
            enc_y_share1 += self.he_matmul(enc_wts[ic], x_share1[ic])
        if self.measure_time:
            self.time_dict["total"] -= time.process_time() - start

        enc_y_share1 = self.send_enc_vector(enc_y_share1)

        if self.measure_time:
            start = time.process_time()
        enc_y_share2 = self.he_matmul(enc_wts[0], x_share2[0])
        for ic in range(1, len(x_share2)):
            enc_y_share2 += self.he_matmul(enc_wts[ic], x_share2[ic])
        enc_y = enc_y_share1 + enc_y_share2 + enc_bias
        if self.measure_time:
            self.time_dict["HE computation"] += time.process_time() - start

        enc_y = self.send_enc_vector(enc_y)

        return enc_y

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

    def generate_shares(self, secret):
        if self.measure_time:
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

        if self.measure_time:
            self.time_dict["shares generation"] += time.process_time() - start

        return [share, mask]

    def predict(self, enc_y):
        y = self.decrypt(enc_y)
        y = y.reshape(-1, self.input_nb)[:, :self.truth_nb]
        output = self.remainder(y)

        pred = np.argmax(output, axis=0)

        share1, share2 = self.generate_shares(pred)

        if self.measure_time:
            self.time_dict["communication"] += communicate(share1) + communicate(share2)

        return share1.reshape(-1), share2.reshape(-1)

    def sec_compare(self, pred_shares, truth_shares):

        if self.measure_time:
            start = time.process_time()
        compare_share1 = pred_shares[0] - truth_shares[0]
        if self.measure_time:
            self.time_dict["total"] -= time.process_time() - start

        if self.measure_time:
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




class Sec_CNN1_MNIST(HybridModel):
    def __init__(self, input_nb=409):
        super(Sec_CNN1_MNIST, self).__init__()
        self.input_nb = input_nb
        self.truth_nb = input_nb
        self.image_len = 28

        self.conv1_stride = 3
        self.conv1_kernel_len = 7
        self.conv1_in_channel_nb = 1
        self.conv1_out_channel_nb = 4
        self.conv1_windows_nb = 8 * 8
        self.conv1_padding = ((0, 0), (0, 0), (0, 0), (0, 0))

        self.fc1_input_size = 256
        self.fc1_output_size = 64

        self.fc2_input_size = 64
        self.fc2_output_size = 10

        self.context = None
        self.enc_conv1 = None
        self.enc_fc1 = None
        self.enc_fc2 = None

    def init_model_paras(self, context, model_paras):
        self.context = context
        self.enc_conv1 = self.encrypt_conv(model_paras["conv1.weight"], model_paras["conv1.bias"],
                                           self.conv1_kernel_len, self.conv1_windows_nb, self.conv1_in_channel_nb)
        self.enc_fc1 = self.encrypt_fc(model_paras["fc1.weight"], model_paras["fc1.bias"], 4)
        self.enc_fc2 = self.encrypt_fc(model_paras["fc2.weight"], model_paras["fc2.bias"], 1)

    def clear_model_paras(self):
        self.context = None
        self.enc_conv1 = None
        self.enc_fc1 = None
        self.enc_fc2 = None

    def set_input_nb(self, input_nb):
        self.input_nb = input_nb
        # conv1_vir_channel_nb_candi = -np.sort(-np.array([1, 8*8, 2, 4*8, 4, 2*8, 8, 1*8]))
        # fc1_channel_nb_candi = -np.sort(-np.array([1, 64, 2, 32, 4, 16, 8, 8]))
        # fc2_channel_nb_candi = -np.sort(-np.array([1, 10, 2, 5]))
        #
        # conv1_max_input_nb = self.n_slots // (self.conv1_windows_nb // conv1_vir_channel_nb_candi)
        # fc1_max_input_nb = self.n_slots // (self.fc1_output_size // fc1_channel_nb_candi)
        # fc2_max_input_nb = self.n_slots // (self.fc2_output_size // fc2_channel_nb_candi)
        #
        # conv1_cipher_nb = np.ceil(input_nb / conv1_max_input_nb) * conv1_vir_channel_nb_candi
        # fc1_cipher_nb = np.ceil(input_nb / fc1_max_input_nb) * fc1_channel_nb_candi
        # fc2_cipher_nb = np.ceil(input_nb / fc2_max_input_nb) * fc2_channel_nb_candi
        #
        # print(conv1_vir_channel_nb_candi)
        # print(fc1_channel_nb_candi)
        # print(fc2_channel_nb_candi)
        #
        # print(conv1_cipher_nb)
        # print(fc1_cipher_nb)
        # print(fc2_cipher_nb)
        #
        # self.conv1_vir_channel_nb = conv1_vir_channel_nb_candi[np.argmin(conv1_cipher_nb)]
        # self.fc1_channel_nb = fc1_channel_nb_candi[np.argmin(fc1_cipher_nb)]
        # self.fc2_channel_nb = fc2_channel_nb_candi[np.argmin(fc2_cipher_nb)]
        #
        # print(self.conv1_vir_channel_nb)
        # print(self.fc1_channel_nb)
        # print(self.fc2_channel_nb)

    def sec_square1(self, enc_y):
        shares1 = []
        shares2 = []
        for oc in range(len(enc_y)):
            y = self.decrypt(enc_y[oc])
            y = self.remainder(y)
            y = self.square(y).reshape(-1, self.input_nb)
            share1, share2 = self.prepare_shares_for_fc(y, self.fc1_output_size)
            shares1.append(share1)
            shares2.append(share2)

        return shares1, shares2

    def sec_square2(self, enc_y):
        y = self.decrypt(enc_y)
        y = self.remainder(y)
        x = self.square(y).reshape(-1, self.input_nb)
        share1, share2 = self.prepare_shares_for_fc(x, self.fc2_output_size)
        return [share1], [share2]

    def preprocess_input(self, x):
        x = x.reshape(self.input_nb, self.conv1_in_channel_nb, self.image_len, self.image_len)
        x = np.transpose(x, (1, 2, 3, 0))
        return self.preprocess_for_conv(x, self.conv1_windows_nb, self.conv1_kernel_len, self.conv1_stride,
                                        self.conv1_padding, self.conv1_in_channel_nb)

    def forward(self, x_shares, truth_shares):
        time1 = time.process_time()
        enc_y_channel = self.sec_conv(self.enc_conv1, x_shares)
        print(time.process_time() - time1)

        x_shares = self.sec_square1(enc_y_channel)

        # x_shares = self.prepare_shares_for_fc(x, self.fc1_output_size)
        enc_y = self.sec_fc(self.enc_fc1, x_shares)
        x_shares = self.sec_square2(enc_y)

        # x_shares = self.prepare_shares_for_fc(x, self.fc2_output_size)
        enc_y = self.sec_fc(self.enc_fc2, x_shares)

        pred_shares = self.predict(enc_y)

        return self.sec_compare(pred_shares, truth_shares)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Sec_CNN2_CIFAR10(HybridModel):
    def __init__(self, input_nb=409):
        super(Sec_CNN2_CIFAR10, self).__init__()
        self.input_nb = input_nb
        self.truth_nb = input_nb
        self.image_len = 32

        self.conv1_stride = 1
        self.conv1_kernel_len = 5
        self.conv1_in_channel_nb = 3
        self.conv1_out_channel_nb = 6
        self.conv1_windows_nb = 28 * 28
        self.conv1_padding = ((0, 0), (0, 0), (0, 0), (0, 0))
        self.conv1_vir_channel_nb = 1

        self.conv2_stride = 1
        self.conv2_kernel_len = 5
        self.conv2_in_channel_nb = 6
        self.conv2_out_channel_nb = 16
        self.conv2_windows_nb = 10 * 10
        self.conv2_padding = self.conv1_padding
        self.conv2_vir_channel_nb = 1

        self.fc1_input_size = 400
        self.fc1_output_size = 120
        self.fc1_channel_nb = 1

        self.fc2_input_size = 120
        self.fc2_output_size = 84
        self.fc2_channel_nb = 1

        self.fc3_input_size = 84
        self.fc3_output_size = 10
        self.fc3_channel_nb = 1

        self.context = None
        self.enc_conv1 = None
        self.enc_conv2 = None
        self.enc_fc1 = None
        self.enc_fc2 = None
        self.enc_fc3 = None

    def init_model_paras(self, context, model_paras):
        self.context = context
        self.enc_conv1 = self.encrypt_conv(model_paras["conv1.weight"], model_paras["conv1.bias"],
                                           self.conv1_kernel_len, self.conv1_windows_nb, self.conv1_in_channel_nb, self.conv1_vir_channel_nb)
        self.enc_conv2 = self.encrypt_conv(model_paras["conv2.weight"], model_paras["conv2.bias"],
                                           self.conv2_kernel_len, self.conv2_windows_nb, self.conv2_in_channel_nb, self.conv2_vir_channel_nb)

        self.enc_fc1 = self.encrypt_fc(model_paras["fc1.weight"], model_paras["fc1.bias"], self.fc1_channel_nb)
        self.enc_fc2 = self.encrypt_fc(model_paras["fc2.weight"], model_paras["fc2.bias"], self.fc2_channel_nb)
        self.enc_fc3 = self.encrypt_fc(model_paras["fc3.weight"], model_paras["fc3.bias"], self.fc3_channel_nb)

    def clear_model_paras(self):
        self.context = None
        self.enc_conv1 = None
        self.enc_conv2 = None
        self.enc_fc1 = None
        self.enc_fc2 = None
        self.enc_fc3 = None

    def sec_relu_maxpool(self, enc_y, shape):
        y_oc = []
        for oc in range(len(enc_y)):
            y_vc = []
            for vc in range(len(enc_y[0])):
                y = self.decrypt(enc_y[oc][vc])
                y_vc.append(y)
            y = np.array(y_vc)
            y = y.reshape(shape)
            y = self.relu(y)
            y = self.maxpool(y)
            y_oc.append(y)
        x = np.array(y_oc)
        return x

    def set_input_nb(self, input_nb):
        self.input_nb = input_nb
        conv1_vir_channel_nb_candi = -np.sort(-np.array([1, 28*28, 2, 14*28, 4, 7*28, 7, 4*28, 8, 7*14, 14, 2*28, 16, 7*7]))
        conv2_vir_channel_nb_candi = -np.sort(-np.array([1, 10*10, 2, 5*10, 4, 5*5, 5, 2*10, 10, 1*10]))
        fc1_channel_nb_candi = -np.sort(-np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120]))
        fc2_channel_nb_candi = -np.sort(-np.array([1, 2, 3, 4, 6, 7, 12, 14, 21, 28, 42, 84]))
        fc3_channel_nb_candi = -np.sort(-np.array([1, 2, 5, 10]))

        conv1_max_input_nb = self.n_slots // (self.conv1_windows_nb // conv1_vir_channel_nb_candi)
        conv2_max_input_nb = self.n_slots // (self.conv2_windows_nb // conv2_vir_channel_nb_candi)
        fc1_max_input_nb = self.n_slots // (self.fc1_output_size // fc1_channel_nb_candi)
        fc2_max_input_nb = self.n_slots // (self.fc2_output_size // fc2_channel_nb_candi)
        fc3_max_input_nb = self.n_slots // (self.fc3_output_size // fc3_channel_nb_candi)

        conv1_cipher_nb = np.ceil(input_nb / conv1_max_input_nb) * conv1_vir_channel_nb_candi
        conv2_cipher_nb = np.ceil(input_nb / conv2_max_input_nb) * conv2_vir_channel_nb_candi
        fc1_cipher_nb = np.ceil(input_nb / fc1_max_input_nb) * fc1_channel_nb_candi
        fc2_cipher_nb = np.ceil(input_nb / fc2_max_input_nb) * fc2_channel_nb_candi
        fc3_cipher_nb = np.ceil(input_nb / fc3_max_input_nb) * fc3_channel_nb_candi

        print(conv1_cipher_nb)
        print(conv2_cipher_nb)
        print(fc1_cipher_nb)
        print(fc2_cipher_nb)
        print(fc3_cipher_nb)

        self.conv1_vir_channel_nb = conv1_vir_channel_nb_candi[np.argmin(conv1_cipher_nb)]
        self.conv2_vir_channel_nb = conv2_vir_channel_nb_candi[np.argmin(conv2_cipher_nb)]
        self.fc1_channel_nb = fc1_channel_nb_candi[np.argmin(fc1_cipher_nb)]
        self.fc2_channel_nb = fc2_channel_nb_candi[np.argmin(fc2_cipher_nb)]
        self.fc3_channel_nb = fc3_channel_nb_candi[np.argmin(fc3_cipher_nb)]

        print(self.conv1_vir_channel_nb)
        print(self.conv2_vir_channel_nb)
        print(self.fc1_channel_nb)
        print(self.fc2_channel_nb)
        print(self.fc3_channel_nb)

    def sec_relu(self, enc_y):
        if type(enc_y) == list:
            y_list = []
            for vc in range(len(enc_y)):
                y = self.decrypt(enc_y[vc])
                y_list.append(y)
            y = np.array(y_list)
        else:
            y = self.decrypt(enc_y)

        y = y.reshape(-1, self.input_nb)
        x = self.relu(y)
        return x

    def preprocess_input(self, x):
        x = x.reshape(self.input_nb, self.conv1_in_channel_nb, self.image_len, self.image_len)
        x = np.transpose(x, (1, 2, 3, 0))
        return self.preprocess_for_conv(x, self.conv1_windows_nb, self.conv1_kernel_len, self.conv1_stride,
                                        self.conv1_padding, self.conv1_in_channel_nb, self.conv1_vir_channel_nb)

    def forward(self, x_shares, truth_shares):

        enc_y_channel = self.sec_conv(self.enc_conv1, x_shares)

        x = self.sec_relu_maxpool(enc_y_channel, [int(self.conv1_windows_nb ** 0.5), int(self.conv1_windows_nb ** 0.5), self.input_nb])

        x_shares = self.prepare_shares_for_conv(x, self.conv2_windows_nb, self.conv2_kernel_len, self.conv2_stride,
                                         self.conv2_padding, self.conv2_in_channel_nb, self.conv2_vir_channel_nb)
        enc_y_channel = self.sec_conv(self.enc_conv2, x_shares)
        x = self.sec_relu_maxpool(enc_y_channel, [int(self.conv2_windows_nb ** 0.5), int(self.conv2_windows_nb ** 0.5), self.input_nb]).reshape(self.fc1_input_size, self.input_nb)

        x_shares = self.prepare_shares_for_fc(x, self.fc1_output_size, vir_channles_nb=self.fc1_channel_nb)
        enc_y = self.sec_fc(self.enc_fc1, x_shares)
        x = self.sec_relu(enc_y).reshape(self.fc2_input_size, self.input_nb)

        x_shares = self.prepare_shares_for_fc(x, self.fc2_output_size, vir_channles_nb=self.fc2_channel_nb)
        enc_y = self.sec_fc(self.enc_fc2, x_shares)
        x = self.sec_relu(enc_y).reshape(self.fc3_input_size, self.input_nb)

        x_shares = self.prepare_shares_for_fc(x, self.fc3_output_size, vir_channles_nb=self.fc3_channel_nb)
        enc_y = self.sec_fc(self.enc_fc3, x_shares)

        pred_shares = self.predict(enc_y)

        return self.sec_compare(pred_shares, truth_shares)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)