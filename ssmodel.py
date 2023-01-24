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
from fxpmath import Fxp

np.seterr(all="ignore")


class SSModel(EncModel):
    def __init__(self, gamma1=27, gamma2=100):
        super(EncModel, self).__init__()
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.trunc_bits = self.gamma2 // 2
        self.modulus_size = 1 + self.gamma1 + self.gamma2
        # self.input_nb = 64
        self.input_nb = None
        self.model_shares = {}
        self.n_processes = 10

    def trunc(self, data):
        # trunc_data = data.astype(np.float64)
        trunc_data = np.floor(data / np.float128(2 ** self.trunc_bits))

        return trunc_data

    def trunc_mul(self, secret1, secret2, matmul=False):
        if matmul:
            result = self.trunc(secret1) @ self.trunc(secret2)
        else:
            result = self.trunc(secret1) * self.trunc(secret2)

        # result = np.floor(result / np.float128(2 ** (self.gamma2 - 2 * self.trunc_bits)))

        return result

    def unsigned_to_signed(self, unsigned):
        modulus = np.float128(2 ** self.modulus_size)
        max_pos = np.float128(2 ** (self.modulus_size - 1))

        unsigned = self.remainder(unsigned)
        signed = copy.deepcopy(unsigned)
        signed[signed >= max_pos] = signed[signed >= max_pos] - modulus

        return signed

    def signed_to_unsigned(self, signed):
        modulus = np.float128(2 ** self.modulus_size)

        unsigned = copy.deepcopy(signed)
        unsigned[unsigned < 0] = unsigned[unsigned < 0] + modulus
        unsigned = self.remainder(unsigned)

        return unsigned

    def decimal_to_signed(self, decimal):
        signed = np.floor(np.float128(decimal) * np.float128(2 ** self.gamma2))
        return signed

    def signed_to_decimal(self, signed):
        decimal = signed / np.float(2 ** self.gamma2)
        return decimal

    def generate_mask(self, secret):
        secret_nb = secret.reshape(-1).shape[0]
        mask_int = random128.random_array(secret_nb, self.modulus_size)
        mask_int = self.unsigned_to_signed(mask_int)

        return mask_int.reshape(secret.shape)

    def generate_shares(self, secret, decimal_to_int=False):
        start = time.process_time()

        if decimal_to_int:
            secret_signed = self.decimal_to_signed(secret)
        else:
            secret_signed = secret

        secret_unsigned = self.signed_to_unsigned(secret_signed)
        mask = self.generate_mask(secret_unsigned)
        mask_unsigned = self.signed_to_unsigned(mask)
        share = self.remainder(secret_unsigned - mask_unsigned)
        share = self.unsigned_to_signed(share)

        self.time_dict["shares generation"] += time.process_time() - start

        return [share, mask]

    def remainder(self, secret):
        secret_int = np.remainder(secret, np.float128(2 ** self.modulus_size))
        return secret_int

    def recover_secret(self, shares):
        secret_signed = shares[0] + shares[1]
        secret_unsigned = self.signed_to_unsigned(secret_signed)
        secret_unsigned = self.remainder(secret_unsigned)
        secret_signed = self.unsigned_to_signed(secret_unsigned)
        secret_decimal = self.signed_to_decimal(secret_signed)

        return secret_decimal

    def send_shares(self, shares):
        for share in shares:
            self.time_dict["communication"] += communicate(share)
        return shares

    def generate_and_send_shares(self, secret, decimal_to_int=True):
        shares = self.generate_shares(secret, decimal_to_int=decimal_to_int)
        shares = self.send_shares(shares)
        return shares

    def model_shares_for_conv(self, weight, bias, conv_windows_nb):
        weight = weight.numpy()
        bias = bias.numpy()

        weight, bias = self.preprocess_conv(weight, bias, conv_windows_nb)

        weight_shares = self.generate_and_send_shares(weight)
        bias_shares = self.generate_and_send_shares(bias)

        return weight_shares, bias_shares
        # weight_share1, bias_share1 = self.preprocess_conv(weight_shares[0], bias_shares[0], conv_windows_nb)
        # start = time.process_time()
        # weight_share2, bias_share2 = self.preprocess_conv(weight_shares[1], bias_shares[1], conv_windows_nb)
        # self.time_dict["repeated"] += time.process_time() - start

        # return (weight_share1, weight_share2), (bias_share1, bias_share2)

    def preprocess_conv(self, conv_weight, conv_bias, conv_windows_nb):
        # repeated_times = conv_windows_nb * self.input_nb
        conv_weight = np.expand_dims(conv_weight, axis=-1)
        # conv_weight = np.repeat(conv_weight, repeated_times, axis=-1)
        conv_bias = conv_bias.reshape(-1, 1)
        # conv_bias = np.repeat(conv_bias, repeated_times, axis=-1)

        return conv_weight, conv_bias

    def model_shares_for_fc(self, weight, bias):
        weight = weight.numpy()
        bias = bias.numpy().reshape(-1, 1)

        weight_shares = self.generate_and_send_shares(weight)
        bias_shares = self.generate_and_send_shares(bias)

        return weight_shares, bias_shares

    def model_shares_for_gru(self, gru_weights, gru_biases):
        hidden_size = int(gru_weights.shape[0] / 3)
        enc_paras = []
        for i in range(3):
            weight = gru_weights[i * hidden_size:(i + 1) * hidden_size]
            bias = gru_biases[i * hidden_size:(i + 1) * hidden_size]
            enc_para = self.model_shares_for_fc(weight, bias)
            enc_paras.append(enc_para)

        return enc_paras

    def preprocess_input(self, x):
        x = x.reshape(self.input_shape)
        x = np.transpose(x, (1, 0))
        x_shares = self.generate_and_send_shares(x)
        return x_shares

    def preprocess_truth(self, truth):
        truth = truth.reshape(-1)
        truth_shares = self.generate_and_send_shares(truth)
        return truth_shares

    def preprocess_data_for_conv(self, x, windows_nb, kernel_len, stride, pad_width=((0, 0), (0, 0), (0, 0))):

        dk = int(windows_nb ** 0.5)
        padded_x = np.pad(x, pad_width)
        input_nb = padded_x.shape[-1]

        features_ic = []
        for ic in range(padded_x.shape[0]):
            x_ic = padded_x[ic]
            features = []
            for i in range(kernel_len):
                for j in range(kernel_len):
                    feature = np.zeros((dk, dk, input_nb))
                    for i_prime in range(dk):
                        for j_prime in range(dk):
                            feature[i_prime, j_prime, :] = x_ic[stride * i_prime + i, stride * j_prime + j, :].reshape(
                                1, 1, input_nb)

                    features.append(feature.reshape(1, -1))

            features = np.concatenate(features, axis=0).reshape(kernel_len, kernel_len, -1)
            features = np.expand_dims(features, axis=0)
            features_ic.append(features)

        features_ic = np.concatenate(features_ic, axis=0)
        features_ic = np.expand_dims(features_ic, axis=0)
        # features_ic = np.repeat(features_ic, self.conv1_out_channel_nb, axis=0)

        return features_ic

    def generate_triplets(self, weight, data, matmul=False):
        mask_a = self.generate_mask(weight)
        mask_b = self.generate_mask(data)
        if matmul:
            mask_c = self.trunc(mask_a) @ self.trunc(mask_b)
        else:
            mask_c = self.trunc(mask_a) * self.trunc(mask_b)

        a_shares = self.generate_shares(mask_a, decimal_to_int=False)
        b_shares = self.generate_shares(mask_b, decimal_to_int=False)
        c_shares = self.generate_shares(mask_c, decimal_to_int=False)
        self.send_shares([a_shares[0]])
        self.send_shares([b_shares[0]])
        self.send_shares([c_shares[0]])

        return a_shares, b_shares, c_shares

    def multiply_shares(self, weight_shares, x_shares, matmul=False):
        weight_share1, weight_share2 = weight_shares
        x_share1, x_share2 = x_shares

        (a_share1, a_share2), (b_share1, b_share2), (c_share1, c_share2) = self.generate_triplets(weight_share1,
                                                                                                  x_share1, matmul)

        start = time.process_time()
        e_share1 = weight_share1 - a_share1
        f_share1 = x_share1 - b_share1
        self.time_dict["repeated"] += time.process_time() - start

        start = time.process_time()
        e_share2 = weight_share2 - a_share2
        f_share2 = x_share2 - b_share2
        self.time_dict["Sec computation"] += time.process_time() - start

        e_shares = self.send_shares([e_share1, e_share2])
        f_shares = self.send_shares([f_share1, f_share2])

        start = time.process_time()
        secret_e = self.recover_secret(e_shares)
        secret_e = self.decimal_to_signed(secret_e)
        secret_f = self.recover_secret(f_shares)
        secret_f = self.decimal_to_signed(secret_f)
        y_share1 = self.trunc_mul(weight_share1, secret_f, matmul=matmul) + self.trunc_mul(secret_e, x_share1,
                                                                                           matmul=matmul) \
                   + c_share1
        self.time_dict["repeated"] += time.process_time() - start

        start = time.process_time()
        secret_e = self.recover_secret(e_shares)
        secret_e = self.decimal_to_signed(secret_e)
        secret_f = self.recover_secret(f_shares)
        secret_f = self.decimal_to_signed(secret_f)
        y_share2 = self.trunc_mul(weight_share2, secret_f, matmul=matmul) + self.trunc_mul(secret_e, x_share2,
                                                                                           matmul=matmul) \
                   + c_share2 - self.trunc_mul(secret_e, secret_f, matmul=matmul)
        self.time_dict["Sec computation"] += time.process_time() - start

        return y_share1, y_share2

    def sec_conv(self, model_shares, x_shares):

        weight_shares, (bias_share1, bias_share2) = model_shares
        # print(weight_shares[0].shape)
        # print(x_shares[0].shape)
        # print(bias_share1.shape)

        # x_share1, x_share2 = x_shares
        # x_share1 = x_share1.repeat(weight_shares[0].shape[0], axis=0)
        # x_share2 = x_share2.repeat(weight_shares[0].shape[0], axis=0)
        # x_shares = (x_share1, x_share2)

        y_share1, y_share2 = self.multiply_shares(weight_shares, x_shares, matmul=False)

        start = time.process_time()
        y_share1 = y_share1.reshape(y_share1.shape[0], -1, y_share1.shape[-1])
        y_share1 = y_share1.sum(axis=1)
        y_share1 += bias_share1
        self.time_dict["repeated"] += time.process_time() - start

        start = time.process_time()
        y_share2 = y_share2.reshape(y_share2.shape[0], -1, y_share2.shape[-1])
        y_share2 = y_share2.sum(axis=1)
        y_share2 += bias_share2
        self.time_dict["Sec computation"] += time.process_time() - start

        y_shares = self.send_shares([y_share1, y_share2])

        return y_shares

    def sec_fc(self, model_shares, x_shares, send_back=False):
        weight_shares, (bias_share1, bias_share2) = model_shares

        if x_shares is None:
            return bias_share1, bias_share2

        y_share1, y_share2 = self.multiply_shares(weight_shares, x_shares, matmul=True)

        start = time.process_time()
        y_share1 = y_share1 + bias_share1.reshape(-1, 1)
        self.time_dict["repeated"] += time.process_time() - start

        start = time.process_time()
        y_share2 = y_share2 + bias_share2.reshape(-1, 1)
        self.time_dict["Sec computation"] += time.process_time() - start

        if send_back:
            y_shares = self.send_shares([y_share1, y_share2])
        else:
            y_shares = (y_share1, y_share2)

        return y_shares

    def predict(self, y_shares):
        y = self.recover_secret(y_shares).reshape(self.output_size, -1)
        pred = self.argmax(y)

        share1, share2 = self.generate_and_send_shares(pred)

        return share1.reshape(-1), share2.reshape(-1)

    def sec_compare(self, pred_shares, truth_shares):
        start = time.process_time()
        compare_share1 = pred_shares[0] - truth_shares[0]
        self.time_dict["repeated"] += time.process_time() - start

        self.time_dict["communication"] += communicate(compare_share1)

        compare_share2 = pred_shares[1] - truth_shares[1]
        compare = self.recover_secret([compare_share1, compare_share2])
        correct = (np.abs(compare) < 0.5).sum().item()

        return correct

    def aggregate(self, param_size_pairs):
        params = [pair[0] for pair in param_size_pairs]
        sizes = [pair[1] for pair in param_size_pairs]
        keys = params[0].keys()

        sizes = np.array(sizes)
        weights = sizes / sizes.sum()

        for key in keys:
            start = time.process_time()
            aggr_param = self.constant_mul(params[0][key], weights[0])
            for i in range(1, len(params)):
                weighted_param = self.constant_mul(params[i][key], weights[i])
                aggr_param = self.add(aggr_param, weighted_param)
            self.model_shares[key] = aggr_param
            self.time_dict["Sec computation"] += time.process_time() - start
        self.aggregated = True


class SS_mRNA_RNN(SSModel):
    def __init__(self):
        super(SS_mRNA_RNN, self).__init__()
        self.input_shape = (-1, 10, 64)

        self.gru_input_size = self.input_shape[2]
        self.gru_output_size = 32
        self.seq_len = self.input_shape[1]

        self.fc_input_size = 32
        self.fc_output_size = 2

        self.output_size = 2

    def init_model_param(self, param):
        self.model_shares["gru_ir"], self.model_shares["gru_iz"], self.model_shares[
            "gru_in"] = self.model_shares_for_gru(
            param["rnn.weight_ih_l0"], param["rnn.bias_ih_l0"])
        self.model_shares["gru_hr"], self.model_shares["gru_hz"], self.model_shares[
            "gru_hn"] = self.model_shares_for_gru(
            param["rnn.weight_hh_l0"], param["rnn.bias_hh_l0"])
        self.model_shares["fc"] = self.model_shares_for_fc(param["fc.weight"], param["fc.bias"])

    def preprocess_input(self, x):
        x = x.reshape(-1, self.seq_len, self.gru_input_size)
        x = np.transpose(x, (1, 2, 0))
        inputs = []
        for i in range(self.seq_len):
            input = x[i]
            input = self.generate_and_send_shares(input)
            inputs.append(input)
        return inputs

    def sec_sigmoid(self, y_shares):
        y = self.recover_secret(y_shares)
        y = y.reshape(-1, self.input_nb)
        x = self.sigmoid(y)
        return x

    def sec_tanh(self, y_shares):
        y = self.recover_secret(y_shares)
        y = y.reshape(-1, self.input_nb)
        x = self.tanh(y)
        return x

    def compute_enc_gru_r(self, x, h):
        r1_shares = self.sec_fc(self.model_shares["gru_ir"], x, send_back=False)
        r2_shares = self.sec_fc(self.model_shares["gru_hr"], h, send_back=False)

        start = time.process_time()
        r_share1 = r1_shares[0] + r2_shares[0]
        self.time_dict["repeated"] += time.process_time() - start

        r_share2 = r1_shares[1] + r2_shares[1]
        r_shares = self.send_shares([r_share1, r_share2])

        r = self.sec_sigmoid(r_shares)
        r_shares = self.generate_and_send_shares(r)
        return r_shares

    def compute_enc_gru_z(self, x, h):
        z1_shares = self.sec_fc(self.model_shares["gru_iz"], x, send_back=False)
        z2_shares = self.sec_fc(self.model_shares["gru_hz"], h, send_back=False)

        start = time.process_time()
        z_share1 = z1_shares[0] + z2_shares[0]
        self.time_dict["repeated"] += time.process_time() - start

        z_share2 = z1_shares[1] + z2_shares[1]
        z_shares = self.send_shares([z_share1, z_share2])

        z = self.sec_sigmoid(z_shares)
        one_minus_z = 1. - z
        z_shares = self.generate_and_send_shares(z)
        one_minus_z_shares = self.generate_and_send_shares(one_minus_z)

        return z_shares, one_minus_z_shares

    def compute_enc_gru_n(self, x, h, r):
        n1_shares = self.sec_fc(self.model_shares["gru_in"], x, send_back=False)
        n2_shares = self.sec_fc(self.model_shares["gru_hn"], h, send_back=False)
        # n2 = self.recover_secret(n2_shares)
        # n2_shares = self.generate_shares(n2)
        n2_shares = self.multiply_shares(n2_shares, r)

        start = time.process_time()
        n_share1 = n1_shares[0] + n2_shares[0]
        self.time_dict["repeated"] += time.process_time() - start

        n_share2 = n1_shares[1] + n2_shares[1]
        n_shares = self.send_shares([n_share1, n_share2])

        n = self.sec_tanh(n_shares)
        n_shares = self.generate_and_send_shares(n)

        return n_shares

    def compute_enc_gru_h(self, z, one_minus_z, n, h):
        h1_shares = self.multiply_shares(one_minus_z, n)

        if not (h is None):
            h2_shares = self.multiply_shares(z, h)

            start = time.process_time()
            h_share1 = h1_shares[0] + h2_shares[0]
            self.time_dict["repeated"] += time.process_time() - start

            h_share2 = h1_shares[1] + h2_shares[1]

            return h_share1, h_share2
        else:
            return h1_shares

    def sec_rnn_gru(self, x_shares_seq, h_shares=None):

        for i in range(self.seq_len):
            x_shares = x_shares_seq[i]
            r_shares = self.compute_enc_gru_r(x_shares, h_shares)
            z_shares, one_minus_z_shares = self.compute_enc_gru_z(x_shares, h_shares)
            n_shares = self.compute_enc_gru_n(x_shares, h_shares, r_shares)
            h_shares = self.compute_enc_gru_h(z_shares, one_minus_z_shares, n_shares, h_shares)
            h_shares = self.send_shares(h_shares)
            h = self.recover_secret(h_shares)
            h_shares = self.generate_and_send_shares(h)

        return h_shares

    def forward(self, x_shares_seq, truth_shares):
        self.input_nb = truth_shares[0].shape[0]
        h_shares = self.sec_rnn_gru(x_shares_seq)

        y_shares = self.sec_fc(self.model_shares["fc"], h_shares)
        pred_shares = self.predict(y_shares)

        return self.sec_compare(pred_shares, truth_shares)


class SS_MNIST_CNN(SSModel):
    def __init__(self):
        super(SS_MNIST_CNN, self).__init__()
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

    def init_model_param(self, param):
        self.model_shares["conv1"] = self.model_shares_for_conv(param["conv1.weight"], param["conv1.bias"],
                                                                self.conv1_windows_nb)
        self.model_shares["fc1"] = self.model_shares_for_fc(param["fc1.weight"], param["fc1.bias"])
        self.model_shares["fc2"] = self.model_shares_for_fc(param["fc2.weight"], param["fc2.bias"])

    def sec_square(self, y_shares):
        y = self.recover_secret(y_shares)
        x = self.square(y)

        return x

    def preprocess_input(self, x):
        x = x.reshape(self.input_shape)
        x = np.transpose(x, (1, 2, 3, 0))

        x = self.preprocess_data_for_conv(x, self.conv1_windows_nb, self.conv1_kernel_len, self.conv1_stride,
                                          self.conv1_padding)
        x_shares = self.generate_and_send_shares(x)

        return x_shares

        # start = time.process_time()
        # x_share1 = self.preprocess_data_for_conv(x_shares[0], self.conv1_windows_nb, self.conv1_kernel_len, self.conv1_stride,
        #                                 self.conv1_padding)
        # self.time_dict["repeated"] += time.process_time() - start
        #
        # x_share2 = self.preprocess_data_for_conv(x_shares[1], self.conv1_windows_nb, self.conv1_kernel_len,
        #                                          self.conv1_stride,
        #                                          self.conv1_padding)

        # return x_share1, x_share2

    def forward(self, x_shares, truth_shares):
        y_shares = self.sec_conv(self.model_shares["conv1"], x_shares)
        x = self.sec_square(y_shares).reshape(self.fc1_input_size, -1)

        x_shares = self.generate_and_send_shares(x)
        y_shares = self.sec_fc(self.model_shares["fc1"], x_shares)
        x = self.sec_square(y_shares).reshape(self.fc2_input_size, -1)

        x_shares = self.generate_and_send_shares(x)
        y_shares = self.sec_fc(self.model_shares["fc2"], x_shares)
        pred_shares = self.predict(y_shares)

        return self.sec_compare(pred_shares, truth_shares)


class SS_Logi(SSModel):
    def __init__(self, input_size, output_size):
        super(SS_Logi, self).__init__()
        self.input_shape = (-1, input_size)

        self.fc_input_size = input_size
        self.fc_output_size = output_size

    def init_model_param(self, param):
        self.model_shares["fc"] = self.model_shares_for_fc(param["fc.weight"], param["fc.bias"])

    def forward(self, x_shares, truth_shares):
        y_shares = self.sec_fc(self.model_shares["fc"], x_shares)
        pred_shares = self.predict(y_shares)

        return self.sec_compare(pred_shares, truth_shares)


class SS_AGNEWS_Logi(SS_Logi):
    def __init__(self):
        self.input_shape = (-1, 300)
        self.output_size = 4
        super(SS_AGNEWS_Logi, self).__init__(self.input_shape[1], self.output_size)


class SS_BANK_Logi(SS_Logi):
    def __init__(self):
        self.input_shape = (-1, 48)
        self.output_size = 2
        super(SS_BANK_Logi, self).__init__(self.input_shape[1], self.output_size)


