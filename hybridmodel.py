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
try:
    from random128 import random128
except ImportError:
    import random128
import gc
from hemodel import EncModel
import skimage.measure
np.seterr(all="ignore")


class HybridModel(EncModel):
    def __init__(self, gamma1=27, gamma2=100):
        super(HybridModel, self).__init__()
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.modulus_size = 1 + self.gamma1 + self.gamma2
        # self.divide_big_mat = False
        self.input_nb = None
        self.linear = False
        self.n_processes = 10
        self.server_nb = 2

    # def encrypt(self, mat):
    #     mat = mat.reshape(-1)
    #     mat_size = mat.shape[0]
    #     big_mat = mat_size > self.n_slots
    #
    #     if big_mat and self.divide_big_mat:
    #         enc_vec = []
    #         nb_mat = math.ceil(mat_size / self.n_slots)
    #
    #         for i in range(nb_mat):
    #             vec = mat[i*self.n_slots:(i+1)*self.n_slots]
    #             enc_vec.append(self.encrypt_(vec))
    #     else:
    #         enc_vec = self.encrypt_(mat)
    #
    #     return enc_vec

    def plaintext(self, mat):
        mat = mat.reshape(-1)
        mat_size = mat.shape[0]
        big_mat = mat_size > self.n_slots

        if big_mat and self.divide_big_mat:
            plain_vec = []
            nb_mat = math.ceil(mat_size / self.n_slots)

            for i in range(nb_mat):
                vec = mat[i*self.n_slots:(i+1)*self.n_slots]
                plain_vec.append(self.plaintext_(vec))
        else:
            plain_vec = self.plaintext_(mat)

        return plain_vec

    def preprocess_shares(self, share1, share2):
        start = time.process_time()
        size = share1.shape[0]
        share1 = share1.reshape(size, -1)
        share1 = np.pad(share1, ((0, self.input_nb - size), (0, 0))).reshape(self.input_nb, -1)
        processed_share1 = self.preprocess_input(share1)
        self.time_dict["repeated"] += time.process_time() - start

        share2 = share2.reshape(size, -1)
        share2 = np.pad(share2, ((0, self.input_nb - size), (0, 0))).reshape(self.input_nb, -1)
        processed_share2 = self.preprocess_input(share2)

        return processed_share1, processed_share2


    def preprocess_for_conv(self, x, windows_nb, kernel_len, stride, pad_width=((0, 0), (0, 0), (0, 0))):
        return super().preprocess_for_conv(x, windows_nb, kernel_len, stride, pad_width, encryption=False)

    def prepare_shares_for_conv(self, x, windows_nb, kernel_len, stride, pad_width=((0, 0), (0, 0), (0, 0))):
        share1, share2 = self.generate_shares(x)
        self.time_dict["communication"] += communicate(share1) + communicate(share2)

        processed_share1 = self.preprocess_for_conv(share1, windows_nb, kernel_len, stride, pad_width)


        start = time.process_time()
        processed_share2 = self.preprocess_for_conv(share2, windows_nb, kernel_len, stride, pad_width)
        self.time_dict["repeated"] += time.process_time() - start

        return processed_share1, processed_share2

    def sec_conv(self, enc_conv, shares):
        enc_y_oc = []
        enc_wt_oc, enc_bias_oc = enc_conv
        share1, share2 = shares

        for oc in range(len(enc_wt_oc)):
            enc_wt_ic = enc_wt_oc[oc]
            enc_bias = enc_bias_oc[oc]

            start = time.process_time()
            enc_y_share1 = self.mul_and_sum_mat_pairs(enc_wt_ic[0], share1[0])
            for ic in range(1, len(enc_wt_ic)):
                result = self.mul_and_sum_mat_pairs(enc_wt_ic[ic], share1[ic])
                enc_y_share1 = self.add(enc_y_share1, result)
            self.time_dict["repeated"] += time.process_time() - start

            enc_y_share1 = self.send_enc_vector(enc_y_share1)

            start = time.process_time()
            enc_y_share2 = self.mul_and_sum_mat_pairs(enc_wt_ic[0], share2[0])
            for ic in range(1, len(enc_wt_ic)):
                result = self.mul_and_sum_mat_pairs(enc_wt_ic[ic], share2[ic])
                enc_y_share2 = self.add(enc_y_share2, result)

            enc_y = self.add(enc_y_share1, enc_y_share2)
            enc_y = self.add(enc_bias, enc_y)
            self.time_dict["HE computation"] += time.process_time() - start

            enc_y = self.send_enc_vector(enc_y)
            enc_y_oc.append(enc_y)

        return enc_y_oc

    def encrypt_fc(self, fc_weight, fc_bias, add_cols=0, return_bytes=False):
        fc_weight = fc_weight.numpy()
        fc_bias = fc_bias.numpy()

        rep_bias = np.tile(fc_bias.reshape(-1, 1), (1, self.input_nb))
        enc_wts = self.enc_perm_mats(fc_weight, return_bytes=return_bytes)
        enc_bias = self.encrypt_and_send(rep_bias, recevier_nb=self.server_nb, return_bytes=return_bytes)

        return enc_wts, enc_bias

    def prepare_shares_for_fc(self, x, d_out):
        share1, share2 = self.generate_shares(x)
        self.time_dict["communication"] += communicate(share1) + communicate(share2)

        processed_share1 = self.preprocess_for_fc(share1, d_out)

        start = time.process_time()
        processed_share2 = self.preprocess_for_fc(share2, d_out)
        self.time_dict["repeated"] += time.process_time() - start

        return processed_share1, processed_share2

    def preprocess_for_fc(self, matrix, d_out):
        enc_mats = self.perm_mats(matrix, d_out)
        return enc_mats

    def sec_fc(self, enc_fc, x_shares, send_back=True):

        enc_wts, enc_bias = enc_fc
        # enc_bias = copy.deepcopy(enc_bias)

        if x_shares is None:
            enc_y = enc_bias
        else:
            x_share1 = x_shares[0]
            x_share2 = x_shares[1]

            start = time.process_time()
            enc_y_share1 = self.he_matmul(enc_wts, x_share1)
            self.time_dict["repeated"] += time.process_time() - start

            enc_y_share1 = self.send_enc_vector(enc_y_share1)

            start = time.process_time()
            enc_y_share2 = self.he_matmul(enc_wts, x_share2)
            enc_y = self.add(enc_y_share1, enc_y_share2)
            enc_y = self.add(enc_bias, enc_y)

            self.time_dict["HE computation"] += time.process_time() - start

        if send_back:
            enc_y = self.send_enc_vector(enc_y)

        return enc_y

    def enc_perm_mats(self, matrix, return_bytes=False):
        mat_nb = matrix.shape[1]
        if self.input_nb > mat_nb:
            rep_times = math.ceil(self.input_nb / mat_nb)
            mat = np.repeat(matrix, rep_times, axis=0).reshape(matrix.shape[0], -1)
        else:
            mat = matrix
        mat = sigma(mat)
        enc_mat = self.encrypt_and_send(mat[:, :self.input_nb], recevier_nb=self.server_nb, return_bytes=return_bytes)
        enc_mats = [enc_mat]

        for i in range(1, mat_nb):
            mat = phi(mat)
            enc_mat = self.encrypt_and_send(mat[:, :self.input_nb], recevier_nb=self.server_nb, return_bytes=return_bytes)
            enc_mats.append(enc_mat)

        return enc_mats

    def perm_mats(self, matrix, d_out):
        if matrix.shape[1] > matrix.shape[0]:
            rep_times = math.ceil(matrix.shape[1] / matrix.shape[0])
            mat = np.repeat(matrix, rep_times, axis=0).reshape(matrix.shape[0], -1)[:, :matrix.shape[1]]
        else:
            mat = matrix

        mat = tau(mat)
        plain_mat = self.plaintext(mat[:d_out, :])
        plain_mats = [plain_mat]
        for i in range(1, mat.shape[0]):
            mat = psi(mat)
            plain_mat = self.plaintext(mat[:d_out, :])
            plain_mats.append(plain_mat)

        return plain_mats

    def he_matmul(self, mat1_ls, mat2_ls):
        enc_y = self.mul_and_sum_mat_pairs(mat1_ls, mat2_ls)
        return enc_y

    def predict(self, enc_y):
        y = self.decrypt(enc_y)
        y = y.reshape(-1, self.input_nb)[:, :self.truth_nb]
        output = self.remainder(y)

        pred = self.argmax(output)

        share1, share2 = self.generate_shares(pred)
        self.time_dict["communication"] += communicate(share1) + communicate(share2)

        return share1.reshape(-1), share2.reshape(-1)

    def sec_compare(self, pred_shares, truth_shares):
        start = time.process_time()
        compare_share1 = pred_shares[0] - truth_shares[0]
        self.time_dict["repeated"] += time.process_time() - start

        self.time_dict["communication"] += communicate(compare_share1)

        compare_share2 = pred_shares[1] - truth_shares[1]
        compare = self.remainder(compare_share1 + compare_share2)
        correct_idxs = np.where(np.abs(compare) < 0.5)[0]
        self.time_dict["communication"] += communicate(correct_idxs)

        return correct_idxs

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

        return recovered_secret


class Sec_MNIST_CNN(HybridModel):
    def __init__(self, input_nb=32):
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

        self.fc1_input_size = 256
        self.fc1_output_size = 64

        self.fc2_input_size = 64
        self.fc2_output_size = 10

        self.output_size = 10

    def init_model_param(self, param, return_bytes=False):
        self.enc_param = {}
        self.enc_param["conv1.weight"], self.enc_param["conv1.bias"] = self.encrypt_conv(
            param["conv1.weight"], param["conv1.bias"], self.conv1_windows_nb, return_bytes=return_bytes)
        self.enc_param["fc1.weight"], self.enc_param["fc1.bias"] = self.encrypt_fc(
            param["fc1.weight"], param["fc1.bias"], return_bytes=return_bytes)
        self.enc_param["fc2.weight"], self.enc_param["fc2.bias"] = self.encrypt_fc(
            param["fc2.weight"], param["fc2.bias"], return_bytes=return_bytes)

    def sec_square1(self, enc_y):
        y_oc = []
        for oc in range(len(enc_y)):
            y = self.decrypt(enc_y[oc])
            y = y.reshape(-1, self.input_nb)[:self.conv1_windows_nb, :]
            y = self.remainder(y)
            y = self.square(y)
            y_oc.append(y)
        x = np.array(y_oc).reshape(-1, self.input_nb)
        return x

    def sec_square2(self, enc_y):
        y = self.decrypt(enc_y)
        y = y.reshape(-1, self.input_nb)[:self.fc1_output_size, :]
        y = self.remainder(y)
        x = self.square(y)
        return x

    def preprocess_input(self, x):
        x = x.reshape(self.input_shape)
        x = np.transpose(x, (1, 2, 3, 0))
        return self.preprocess_for_conv(x, self.conv1_windows_nb, self.conv1_kernel_len, self.conv1_stride,
                                        self.conv1_padding)

    def forward(self, x_shares, truth_shares):

        enc_y_channel = self.sec_conv((self.enc_param["conv1.weight"], self.enc_param["conv1.bias"]), x_shares)
        x = self.sec_square1(enc_y_channel)


        x_shares = self.prepare_shares_for_fc(x, self.fc1_output_size)
        enc_y = self.sec_fc((self.enc_param["fc1.weight"], self.enc_param["fc1.bias"]), x_shares)
        x = self.sec_square2(enc_y).reshape(self.fc2_input_size, self.input_nb)

        x_shares = self.prepare_shares_for_fc(x, self.fc2_output_size)
        enc_y = self.sec_fc((self.enc_param["fc2.weight"], self.enc_param["fc2.bias"]), x_shares)
        pred_shares = self.predict(enc_y)

        return self.sec_compare(pred_shares, truth_shares)


class Sec_mRNA_RNN(HybridModel):
    def __init__(self, input_nb=64):
        super(Sec_mRNA_RNN, self).__init__()
        self.input_nb = input_nb
        self.truth_nb = input_nb
        self.input_shape = (-1, 10, 64)

        self.gru_input_size = self.input_shape[2]
        self.gru_output_size = 32
        self.seq_len = self.input_shape[1]

        self.fc_input_size = 32
        self.fc_output_size = 2

        self.output_size = 2

    def init_model_param(self, param, return_bytes=False):
        self.enc_param = {}
        enc_param = self.enc_param
        (enc_param["gru_ir.weight"], enc_param["gru_ir.bias"]), (enc_param["gru_iz.weight"], enc_param["gru_iz.bias"]), \
        (enc_param["gru_in.weight"], enc_param["gru_in.bias"]) \
            = self.encrypt_gru(param["rnn.weight_ih_l0"], param["rnn.bias_ih_l0"], return_bytes=return_bytes)

        (enc_param["gru_hr.weight"], enc_param["gru_hr.bias"]), (enc_param["gru_hz.weight"], enc_param["gru_hz.bias"]), \
        (enc_param["gru_hn.weight"], enc_param["gru_hn.bias"]) \
            = self.encrypt_gru(param["rnn.weight_hh_l0"], param["rnn.bias_hh_l0"],
                               add_cols_to_hidden=self.gru_input_size - self.gru_output_size, return_bytes=return_bytes)

        enc_param["fc.weight"], enc_param["fc.bias"] = self.encrypt_fc(param["fc.weight"], param["fc.bias"],
                                                                              return_bytes=return_bytes)

    def preprocess_input(self, x):
        x = x.reshape(self.input_nb, self.seq_len, self.gru_input_size)
        x = np.transpose(x, (1, 2, 0))
        inputs = []
        for i in range(self.seq_len):
            input = x[i]
            input = self.preprocess_for_fc(input, self.gru_output_size)
            inputs.append(input)
        return inputs

    def sec_sigmoid(self, enc_y):
        y = self.decrypt(enc_y)
        y = y.reshape(-1, self.input_nb)
        y = self.remainder(y)
        x = self.sigmoid(y)
        return x

    def sec_tanh(self, enc_y):
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
        self.time_dict["repeated"] += time.process_time() - start

        return processed_share1, processed_share2

    def sec_hardmard_prodcut(self, enc_x1, x2_shares, send_back=True):
        start = time.process_time()
        enc_y_share1 = self.entrywise_mul(enc_x1, x2_shares[0])
        self.time_dict["repeated"] += time.process_time() - start

        enc_y_share1 = self.send_enc_vector(enc_y_share1)

        start = time.process_time()
        enc_y_share2 = self.entrywise_mul(enc_x1, x2_shares[1])
        enc_y = self.add(enc_y_share1, enc_y_share2)
        self.time_dict["HE computation"] += time.process_time() - start

        if send_back:
            enc_y = self.send_enc_vector(enc_y)

        return enc_y

    def compute_enc_gru_r(self, x, h):
        enc_r1 = self.sec_fc((self.enc_param["gru_ir.weight"], self.enc_param["gru_ir.bias"]), x, send_back=False)
        enc_r2 = self.sec_fc((self.enc_param["gru_hr.weight"], self.enc_param["gru_hr.bias"]), h, send_back=False)
        enc_r = self.add(enc_r1, enc_r2)
        enc_r = self.send_enc_vector(enc_r)
        r = self.sec_sigmoid(enc_r)
        r_shares = self.prepare_shares_for_hadamard(r)
        return r_shares

    def compute_enc_gru_z(self, x, h):
        enc_z1 = self.sec_fc((self.enc_param["gru_iz.weight"], self.enc_param["gru_iz.bias"]), x, send_back=False)
        enc_z2 = self.sec_fc((self.enc_param["gru_hz.weight"], self.enc_param["gru_hz.bias"]), h, send_back=False)
        enc_z = self.add(enc_z1, enc_z2)
        enc_z = self.send_enc_vector(enc_z)
        z = self.sec_sigmoid(enc_z)
        enc_z = self.encrypt_and_send(z)
        return enc_z

    def compute_enc_gru_n(self, x, h, r):
        enc_n1 = self.sec_fc((self.enc_param["gru_in.weight"], self.enc_param["gru_in.bias"]), x, send_back=False)
        enc_n2 = self.sec_fc((self.enc_param["gru_hn.weight"], self.enc_param["gru_hn.bias"]), h, send_back=False)

        if not (h is None) and self.aggregated:
            enc_n2 = self.send_enc_vector(enc_n2)
            n2 = self.decrypt(enc_n2)
            enc_n2 = self.encrypt_and_send(n2)

        enc_n2 = self.sec_hardmard_prodcut(enc_n2, r, send_back=False)
        enc_n = self.add(enc_n1, enc_n2)
        enc_n = self.send_enc_vector(enc_n)
        n = self.sec_tanh(enc_n)
        n_shares = self.prepare_shares_for_hadamard(n)

        return n_shares

    def compute_enc_gru_h(self, z, n, h):
        one_minus_z = self.subtract(self.plaintext(np.ones(self.calc_size(z))), z)
        enc_h = self.sec_hardmard_prodcut(one_minus_z, n, send_back=False)
        if not (h is None):
            result = self.sec_hardmard_prodcut(z, h, send_back=False)
            enc_h = self.add(enc_h, result)
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

        h = self.sec_rnn_gru(x_shares_seq)

        x_shares = self.prepare_shares_for_fc(h, self.fc_output_size)
        enc_y = self.sec_fc((self.enc_param["fc.weight"], self.enc_param["fc.bias"]), x_shares)
        pred_shares = self.predict(enc_y)

        return self.sec_compare(pred_shares, truth_shares)


class Sec_Logi(HybridModel):
    def __init__(self, input_nb, input_size, output_size):
        super(Sec_Logi, self).__init__()
        self.input_nb = input_nb
        self.input_shape = (-1, input_size)

        self.fc_input_size = input_size
        self.fc_output_size = output_size

        self.linear = True

    def init_model_param(self, param, return_bytes=False):
        self.enc_param = {}
        self.enc_param["fc.weight"], self.enc_param["fc.bias"] = self.encrypt_fc(param["fc.weight"], param["fc.bias"],
                                                                                 return_bytes=return_bytes)

    def preprocess_input(self, x):
        x = x.reshape(self.input_shape)
        x = np.transpose(x, (1, 0))
        return self.preprocess_for_fc(x, self.fc_output_size)

    def forward(self, x_shares, truth_shares):

        enc_y = self.sec_fc((self.enc_param["fc.weight"], self.enc_param["fc.bias"]), x_shares)
        pred_shares = self.predict(enc_y)

        return self.sec_compare(pred_shares, truth_shares)


class Sec_AGNEWS_Logi(Sec_Logi):
    def __init__(self, input_nb=512):
        self.input_shape = (-1, 300)
        self.output_size = 4
        super(Sec_AGNEWS_Logi, self).__init__(input_nb, self.input_shape[1], self.output_size)


class Sec_BANK_Logi(Sec_Logi):
    def __init__(self, input_nb=1024):
        self.input_shape = (-1, 48)
        self.output_size = 2
        super(Sec_BANK_Logi, self).__init__(input_nb, self.input_shape[1], self.output_size)


class Sec_DNN(HybridModel):
    def __init__(self, input_nb, layer_nb=5):
        super(Sec_DNN, self).__init__()
        self.input_nb = input_nb
        self.layer_nb = layer_nb
        self.hidden_layer_size = 64

    def init_model_param(self, param, return_bytes=False):
        self.enc_param = {}
        enc_param = self.enc_param
        enc_param["input.weight"], enc_param["input.bias"] = self.encrypt_fc(param["input_layer.weight"],
                                                                             param["input_layer.bias"],
                                                                             return_bytes=return_bytes)
        for l in range(self.layer_nb):
            enc_param[f"hidden.{l}.weight"], enc_param[f"hidden.{l}.bias"] = self.encrypt_fc(
                param[f"hidden_layers.{l * 2}.weight"], param[f"hidden_layers.{l * 2}.bias"], return_bytes=return_bytes)

        enc_param["output.weight"], enc_param["output.bias"] = self.encrypt_fc(param["output_layer.weight"],
                                                                             param["output_layer.bias"],
                                                                             return_bytes=return_bytes)

    def preprocess_input(self, x):
        x = x.reshape(self.input_shape)
        x = np.transpose(x, (1, 0))
        return self.preprocess_for_fc(x, self.hidden_layer_size)

    def sec_relu(self, enc_y):
        y = self.decrypt(enc_y).reshape(-1, self.input_nb)
        y = self.remainder(y)
        x = self.relu(y)
        return x

    def forward(self, x_shares, truth_shares):
        enc_y = self.sec_fc((self.enc_param["input.weight"], self.enc_param["input.bias"]), x_shares)
        x = self.sec_relu(enc_y)

        for l in range(self.layer_nb):
            x_shares = self.prepare_shares_for_fc(x, self.hidden_layer_size)
            enc_y = self.sec_fc((self.enc_param[f"hidden.{l}.weight"], self.enc_param[f"hidden.{l}.bias"]), x_shares)
            x = self.sec_relu(enc_y)

        x_shares = self.prepare_shares_for_fc(x, self.output_size)
        enc_y = self.sec_fc((self.enc_param["output.weight"], self.enc_param["output.bias"]), x_shares)
        pred_shares = self.predict(enc_y)

        return self.sec_compare(pred_shares, truth_shares)


class Sec_BANK_DNN(Sec_DNN):
    def __init__(self, input_nb=32, layer_nb=5):
        super(Sec_BANK_DNN, self).__init__(input_nb=input_nb, layer_nb=layer_nb)
        self.input_shape = (-1, 48)
        self.output_size = 2


class Sec_AGNEWS_DNN(Sec_DNN):
    def __init__(self, input_nb=32, layer_nb=5):
        super(Sec_AGNEWS_DNN, self).__init__(input_nb=input_nb, layer_nb=layer_nb)
        self.input_shape = (-1, 300)
        self.output_size = 4


class Sec_MNIST_DNN(Sec_DNN):
    def __init__(self, input_nb=32, layer_nb=5):
        super(Sec_MNIST_DNN, self).__init__(input_nb=input_nb, layer_nb=layer_nb)
        self.input_shape = (-1, 784)
        self.output_size = 10


class Sec_mRNA_DNN(Sec_DNN):
    def __init__(self, input_nb=32, layer_nb=5):
        super(Sec_mRNA_DNN, self).__init__(input_nb=input_nb, layer_nb=layer_nb)
        self.input_shape = (-1, 640)
        self.output_size = 2
