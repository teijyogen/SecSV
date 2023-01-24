import torch
import tenseal as ts
import math
import numpy as np
# import tenseal.sealapi as sealapi
import random
from tqdm import tqdm
from torch.utils.data import Dataset
import time
from utils import *
import copy
import random
from collections import Counter
import random128
import skimage.measure
import gc
from collections.abc import Iterable

class EncModel:
    def __init__(self):
        self.context = None
        self.context_sk = None
        self.time_dict = {}
        self.enc_param = {}
        self.aggregated = False
        self.server_nb = 1

    def init_context(self):
        self.context = get_shared_context("context")
        self.context_sk = get_shared_context(name="context_sk")

    def clear_context(self):
        self.context = None
        self.context_sk = None

    def calc_size(self, enc_vec):
        if type(enc_vec) == list:
            size = 0
            for ele in enc_vec:
                size += ele.size()
        else:
            size = enc_vec.size()
        return size

    def encrypt_(self, vector):
        start = time.process_time()
        enc_vec = ts.ckks_vector(self.context, vector.reshape(-1))
        self.time_dict["encryption"] += time.process_time() - start
        return enc_vec

    def encrypt(self, mat):
        enc_vec = self.encrypt_(mat)
        return enc_vec

    def send_enc_vector_(self, enc_vec, return_bytes=False, recevier_nb=1):
        enc_vec_bytes = enc_vec.serialize()
        self.time_dict["communication"] += communicate(enc_vec_bytes) * recevier_nb
        if return_bytes:
            return enc_vec_bytes
        else:
            enc_vec = ts.CKKSVector.load(self.context, enc_vec_bytes)
            return enc_vec

    def send_enc_vector(self, enc_vec, return_bytes=False, recevier_nb=1):
        if type(enc_vec) == list:
            enc_vec_ls = []
            for ele in enc_vec:
                new_ele = self.send_enc_vector_(ele, return_bytes=return_bytes, recevier_nb=recevier_nb)
                enc_vec_ls.append(new_ele)
            return enc_vec_ls
        else:
            enc_vec = self.send_enc_vector_(enc_vec, return_bytes=return_bytes, recevier_nb=recevier_nb)
            return enc_vec

    def encrypt_and_send(self, mat, return_bytes=False, recevier_nb=1):
        enc_mat = self.encrypt(mat)
        enc_mat = self.send_enc_vector(enc_mat, return_bytes=return_bytes, recevier_nb=recevier_nb)
        return enc_mat

    def decrypt_(self, enc_vec):
        start = time.process_time()
        vec = np.array(enc_vec.decrypt(self.context_sk.secret_key()))
        self.time_dict["decryption"] += time.process_time() - start
        return vec

    def decrypt(self, enc_vec):
        if type(enc_vec) == list:
            y = np.array([])
            for i in range(len(enc_vec)):
                sub_y = self.decrypt_(enc_vec[i])
                y = np.append(y, sub_y)
        else:
            y = self.decrypt_(enc_vec)
        return y

    def plaintext_(self, mat):
        return ts.plain_tensor(mat.reshape(-1), dtype='float')

    def relu(self, vec):
        start = time.process_time()
        res = np.maximum(0, vec)
        self.time_dict["activation"] += time.process_time() - start
        return res

    def sigmoid(self, vec):
        start = time.process_time()
        res = 1 / (1 + np.exp(-vec))
        self.time_dict["activation"] += time.process_time() - start
        return res

    def tanh(self, vec):
        start = time.process_time()
        res = np.tanh(vec)
        self.time_dict["activation"] += time.process_time() - start
        return res

    def maxpool(self, vec):
        start = time.process_time()
        res = skimage.measure.block_reduce(vec, (2, 2, 1), np.max)
        self.time_dict["activation"] += time.process_time() - start
        return res

    def square(self, vec):
        start = time.process_time()
        res = vec * vec
        self.time_dict["activation"] += time.process_time() - start
        return res

    def argmax(self, vec):
        start = time.process_time()
        res = np.argmax(vec, axis=0)
        self.time_dict["activation"] += time.process_time() - start
        return res

    def add(self, mat1, mat2):
        if isinstance(mat1, (list, tuple)) and isinstance(mat2, (list, tuple)):
            enc_y = []
            for i in range(len(mat1)):
                enc_y.append(self.add(mat1[i], mat2[i]))
        elif not isinstance(mat1, (list, tuple)) and not isinstance(mat2, (list, tuple)):
            enc_y = mat1 + mat2
        else:
            raise TypeError("Add not supported for the data types")

        return enc_y

    def subtract(self, mat1, mat2):
        if type(mat1) == list and type(mat2) == list:
            if len(mat1) != len(mat2):
                raise ValueError("Matrix numbers do not match")
            enc_y = []
            for i in range(len(mat1)):
                enc_y.append(self.subtract(mat1[i], mat2[i]))
        elif type(mat1) != list and type(mat2) != list:
            enc_y = mat1 - mat2
        else:
            raise TypeError("Subtract not supported between list and tensor ")

        return enc_y

    def entrywise_mul(self, mat1, mat2):
        if type(mat1) == list and type(mat2) == list:
            if len(mat1) != len(mat2):
                raise ValueError("Matrix numbers do not match")
            enc_y = []
            for i in range(len(mat1)):
                enc_y.append(self.entrywise_mul(mat1[i], mat2[i]))
        elif type(mat1) != list and type(mat2) != list:
            enc_y = mat1 * mat2
        else:
            raise TypeError("Dotmul not supported between list and tensor ")

        return enc_y

    def constant_mul(self, mat, constant):
        if isinstance(mat, (list, tuple)):
            enc_y = []
            for submat in mat:
                new_submat = self.constant_mul(submat, constant)
                enc_y.append(new_submat)
        else:
            enc_y = mat * constant
        return enc_y

    def mul_and_sum_mat_pairs(self, mat1, mat2):
        if len(mat1) != len(mat2):
            raise ValueError("Matrix numbers do not match")

        mat_nb = len(mat1)
        enc_y = self.entrywise_mul(mat1[0], mat2[0])
        for i in range(1, mat_nb):
            result = self.entrywise_mul(mat1[i], mat2[i])
            enc_y = self.add(enc_y, result)

        return enc_y

    def aggregate(self, param_size_pairs):
        params = [pair[0] for pair in param_size_pairs]
        sizes = [pair[1] for pair in param_size_pairs]
        keys = params[0].keys()

        sizes = np.array(sizes)
        weights = sizes / sizes.sum()

        self.enc_param = {}
        for key in keys:
            start = time.process_time()
            aggr_param = self.constant_mul(params[0][key], weights[0])
            for i in range(1, len(params)):
                weighted_param = self.constant_mul(params[i][key], weights[i])
                aggr_param = self.add(aggr_param, weighted_param)
            self.enc_param[key] = aggr_param
            self.time_dict["HE computation"] += time.process_time() - start
        self.aggregated = True

    def encrypt_conv(self, conv_weight, conv_bias, conv_windows_nb, return_bytes=False):
        conv_weight = conv_weight.numpy()
        conv_bias = conv_bias.numpy()

        repeated_times = conv_windows_nb * self.input_nb
        conv_weight = np.expand_dims(conv_weight, axis=-1)
        conv_weight = np.repeat(conv_weight, repeated_times, axis=-1)
        conv_bias = conv_bias.reshape(-1, 1)
        conv_bias = np.repeat(conv_bias, repeated_times, axis=-1)

        enc_wt_oc, enc_bias_oc = [], []
        for weight, bias in zip(conv_weight, conv_bias):
            enc_wt_ic = []
            for ic in range(conv_weight.shape[1]):
                ic_wt = weight[ic].reshape(-1, repeated_times)
                enc_weights = []
                for wt in ic_wt:
                    enc_weight = self.encrypt_and_send(wt, recevier_nb=self.server_nb, return_bytes=return_bytes)
                    enc_weights.append(enc_weight)

                enc_wt_ic.append(enc_weights)

            enc_wt_oc.append(enc_wt_ic)

            enc_bias = self.encrypt_and_send(bias, recevier_nb=self.server_nb, return_bytes=return_bytes)
            enc_bias_oc.append(enc_bias)

        return enc_wt_oc, enc_bias_oc

    def preprocess_for_conv(self, x, windows_nb, kernel_len, stride, pad_width=((0, 0), (0, 0), (0, 0)), return_bypes=False, encryption=True):

        dk = int(windows_nb ** 0.5)
        padded_x = np.pad(x, pad_width)

        features_ic = []
        for x_ic in padded_x:
            features = []
            for i in range(kernel_len):
                for j in range(kernel_len):
                    feature = np.zeros((dk, dk, self.input_nb))
                    for i_prime in range(dk):
                        for j_prime in range(dk):
                            feature[i_prime, j_prime, :] = x_ic[stride * i_prime + i, stride * j_prime + j, :].reshape(
                                1, 1, self.input_nb)

                    if encryption:
                        enc_feature = self.encrypt_and_send(feature, return_bytes=return_bypes)
                        features.append(enc_feature)
                    else:
                        plain_feature = self.plaintext(feature)
                        features.append(plain_feature)

            features_ic.append(features)

        return features_ic

    def encrypt_fc(self, weight, bias, add_cols=0, return_bytes=False):
        pass

    def encrypt_gru(self, gru_weights, gru_biases, add_cols_to_hidden=0, return_bytes=False):
        hidden_size = int(gru_weights.shape[0] / 3)
        enc_params = []
        for i in range(3):
            weight = gru_weights[i*hidden_size:(i+1)*hidden_size]
            bias = gru_biases[i*hidden_size:(i+1)*hidden_size]
            enc_param = self.encrypt_fc(weight, bias, add_cols=add_cols_to_hidden, return_bytes=return_bytes)
            enc_params.append(enc_param)

        return enc_params

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class HEModel(EncModel):
    def __init__(self):
        super(HEModel, self).__init__()
        self.n_slots = 2048
        self.max_len = int(self.n_slots ** 0.5)

    def set_n_slots(self, n_slots):
        self.n_slots = n_slots
        self.max_len = int(n_slots ** 0.5)

    def shift_bycols(self, enc_vec, col_id, step=1):
        n_rows = enc_vec.shape[0] // n_cols
        mask = np.zeros(enc_vec.shape[0]).reshape(n_rows, n_cols)
        mask[:, :1] = 1.0
        mask = mask.reshape(-1)

        temp = enc_vec * mask
        res = enc_vec - temp

        res.rotate_vector_inplace(step)
        temp.rotate_vector_inplace(step-n_cols)

        return res + temp

    def sec_conv(self, enc_conv, enc_features):
        enc_y_oc = []
        enc_wt_oc, enc_bias_oc = enc_conv

        for oc in range(len(enc_wt_oc)):
            enc_wt_ic = enc_wt_oc[oc]
            enc_bias = enc_bias_oc[oc]

            start = time.process_time()
            enc_y = self.mul_and_sum_mat_pairs(enc_features[0], enc_wt_ic[0])
            for ic in range(1, len(enc_wt_ic)):
                enc_y += self.mul_and_sum_mat_pairs(enc_features[ic], enc_wt_ic[ic])
            enc_y = enc_bias + enc_y
            self.time_dict["HE computation"] += time.process_time() - start

            enc_y = self.send_enc_vector(enc_y)
            enc_y_oc.append(enc_y)

        return enc_y_oc

    def encrypt_fc(self, fc_weight, fc_bias, add_cols=0, return_bytes=False):
        fc_weight = fc_weight.numpy()
        fc_bias = fc_bias.numpy()

        fc_weight = np.pad(fc_weight, ((0, 0), (0, add_cols)))

        d_out, d_in = fc_weight.shape
        max_len = self.max_len

        if d_in > max_len:
            pad_cols = (max_len - d_in % max_len) % max_len
            fc_weight = np.pad(fc_weight, ((0, 0), (0, pad_cols)))

        fc_bias = fc_bias.reshape(-1, 1)
        fc_bias = fc_bias.repeat(min(max_len, d_in), axis=1)
        if d_out > max_len:
            pad_rows = (max_len - d_out % max_len) % max_len
            fc_bias = np.pad(fc_bias, ((0, pad_rows), (0, 0)))
        else:
            pad_rows = min(d_in, max_len) - d_out
            fc_bias = np.pad(fc_bias, ((0, pad_rows), (0, 0)))

        J = math.ceil(d_out / max_len)
        K = math.ceil(d_in / max_len)

        enc_wt_row_channels = []
        enc_bias_row_channels = []
        for j in range(J):
            enc_wt_col_channels = []
            for k in range(K):
                weight = fc_weight[j * max_len:(j + 1) * max_len, k * max_len:(k + 1) * max_len]
                enc_weight = self.enc_perm_mats(weight, weight.shape[0], left=True, return_bytes=return_bytes)
                enc_wt_col_channels.append(enc_weight)

            enc_wt_row_channels.append(enc_wt_col_channels)

            bias = fc_bias[j * max_len:(j + 1) * max_len, :]
            enc_bias = self.encrypt_and_send(bias, return_bytes=return_bytes)
            enc_bias_row_channels.append(enc_bias)

        return enc_wt_row_channels, enc_bias_row_channels

    def preprocess_for_fc(self, matrix, d_out, return_bytes=False, add_rows=0):
        matrix = np.pad(matrix, ((0, add_rows), (0, 0)))

        d_in = matrix.shape[0]
        max_len = self.max_len

        if d_in > max_len:
            pad_rows = (max_len - d_in % max_len) % max_len
            matrix = np.pad(matrix, ((0, pad_rows), (0, 0)))

        J = math.ceil(d_out / max_len)
        K = math.ceil(d_in / max_len)

        enc_row_channels = []
        for j in range(J):
            enc_col_channels = []
            for k in range(K):
                sub_mat = matrix[k*max_len:(k+1)*max_len]
                enc_mat = self.enc_perm_mats(sub_mat, min(d_out, self.max_len), left=False, return_bytes=return_bytes)
                enc_col_channels.append(enc_mat)
            enc_row_channels.append(enc_col_channels)

        return enc_row_channels

    def sec_fc(self, enc_fc, enc_xs, send_back=True):

        enc_wt_rc, enc_bias_rc = enc_fc
        enc_row_channels = []
        for j in range(len(enc_wt_rc)):
            enc_wt_cc = enc_wt_rc[j]
            enc_bias = enc_bias_rc[j]

            if enc_xs is None:
                enc_y = enc_bias
            else:
                start = time.process_time()
                enc_y = self.he_matmul(enc_xs[j][0], enc_wt_cc[0])
                for k in range(1, len(enc_wt_cc)):
                    enc_y += self.he_matmul(enc_xs[j][k], enc_wt_cc[k])
                enc_y = enc_bias + enc_y
                self.time_dict["HE computation"] += time.process_time() - start

            if send_back:
                enc_y = self.send_enc_vector(enc_y)

            enc_row_channels.append(enc_y)

        return enc_row_channels

    def enc_perm_mats(self, matrix, d_out, left=True, return_bytes=False):

        nrows = matrix.shape[0]
        ncols = matrix.shape[1]

        if left:
            d_in = ncols
        else:
            d_in = nrows

        if d_out < d_in:
            if d_in % d_out != 0:
                for i in range(d_in // d_out, 0, -1):
                    if d_in // i == d_in / i:
                        d_out += d_in // i - d_out
                        break

        if left:
            matrix = np.pad(matrix, ((0, d_out - nrows), (0, 0)))
            square_mat = np.concatenate([matrix for i in range(d_in // d_out)], axis=0)
        else:
            square_mat = np.pad(matrix, ((0, 0), (0, min(d_in, self.max_len) - ncols)))

        enc_mats = self.generate_perms(square_mat, d_out, left=left, return_bytes=return_bytes)

        return enc_mats

    def generate_perms(self, mat, d_out, left=True, return_bytes=False):

        if left:
            mat = sigma(mat)
            enc_mats = [self.encrypt_and_send(mat, return_bytes=return_bytes)]

            for i in range(1, d_out):
                new_mat = np.roll(mat, -i, axis=1)
                enc_mats.append(
                    self.encrypt_and_send(new_mat, return_bytes=return_bytes))
        else:
            mat = tau(mat)
            enc_mats = [self.encrypt_and_send(mat, return_bytes=return_bytes)]
            for i in range(1, d_out):
                new_mat = np.roll(mat, -i, axis=0)
                enc_mats.append(
                    self.encrypt_and_send(new_mat, return_bytes=return_bytes))
        return enc_mats

    def he_matmul(self, mat1_ls, mat2_ls):
        enc_y = self.mul_and_sum_mat_pairs(mat1_ls, mat2_ls)

        d_in = int(mat1_ls[0].size() ** 0.5)
        d_out = len(mat1_ls)
        double_times = int(math.log(d_in / d_out, 2))

        if double_times > 0:
            result = enc_y + enc_y.rotate_vector(d_in * d_out * 2 ** 0)
            for k in range(1, double_times):
                result += result.rotate_vector(d_in * d_out * 2 ** k)
        else:
            return enc_y

        if 2 ** double_times < d_in / d_out:
            result += enc_y.rotate_vector_inplace(2 ** double_times * d_in * d_out)
            for j in range(2 ** double_times + 1, d_in // d_out):
                result += enc_y.rotate_vector_inplace(d_in * d_out)

        return result

    def predict(self, enc_y):

        y = self.decrypt(enc_y)
        y = y.reshape(-1, self.input_nb)[:self.output_size, :self.truth_nb]
        pred = self.argmax(y)

        enc_pred = self.encrypt_and_send(pred)

        return enc_pred

    def encrypt_truth(self, truth):
        enc_truth_bytes = self.encrypt_and_send(truth, return_bytes=True)
        return enc_truth_bytes

    def sec_compare(self, enc_pred, enc_truth):
        start = time.process_time()
        enc_compare = enc_pred - enc_truth
        self.time_dict["HE computation"] += time.process_time() - start

        enc_compare = self.send_enc_vector(enc_compare)
        compare = self.decrypt(enc_compare)
        correct = (np.abs(compare) < 0.5).sum().item()
        self.time_dict["communication"] += communicate(correct)

        return correct


class HE_MNIST_CNN(HEModel):
    def __init__(self, input_nb=45):
        super(HE_MNIST_CNN, self).__init__()
        self.input_nb = input_nb
        self.input_shape = (-1, 1, 28, 28)
        self.image_len = 28
        self.n_processes = 10

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

    def encrypt_input(self, x):
        x = x.reshape(self.input_nb, self.conv1_in_channel_nb, self.image_len, self.image_len)
        x = np.transpose(x, (1, 2, 3, 0))
        return self.preprocess_for_conv(x, self.conv1_windows_nb, self.conv1_kernel_len, self.conv1_stride,
                                        self.conv1_padding, True)

    def sec_square(self, enc_y, n_rows):
        y = self.decrypt(enc_y).reshape(-1, self.max_len)[:n_rows, :self.input_nb]
        x = self.square(y)
        return x

    def forward(self, enc_x, enc_truth):

        enc_y_channel = self.sec_conv((self.enc_param["conv1.weight"], self.enc_param["conv1.bias"]), enc_x)
        x = self.sec_square(enc_y_channel, self.fc1_input_size)

        enc_x = self.preprocess_for_fc(x, self.fc1_output_size)
        enc_y = self.sec_fc((self.enc_param["fc1.weight"], self.enc_param["fc1.bias"]), enc_x)
        x = self.sec_square(enc_y, self.fc1_output_size)

        enc_x = self.preprocess_for_fc(x, self.fc2_output_size)
        enc_y = self.sec_fc((self.enc_param["fc2.weight"], self.enc_param["fc2.bias"]), enc_x)
        enc_pred = self.predict(enc_y)

        return self.sec_compare(enc_pred, enc_truth)


class HE_mRNA_RNN(HEModel):
    def __init__(self, input_nb=32):
        super(HE_mRNA_RNN, self).__init__()
        self.input_nb = input_nb
        self.truth_nb = input_nb
        self.input_shape = (-1, 10, 64)
        self.n_processes = 10

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

    def encrypt_input(self, x):
        x = x.reshape(self.input_nb, self.seq_len, self.gru_input_size)
        x = np.transpose(x, (1, 2, 0))
        inputs = []
        for i in range(self.seq_len):
            input = x[i]
            input = self.preprocess_for_fc(input, self.gru_output_size, return_bytes=True)
            inputs.append(input)
        return inputs

    def sec_sigmoid(self, enc_y):
        y = self.decrypt(enc_y)
        x = self.sigmoid(y)
        return x

    def sec_tanh(self, enc_y):
        y = self.decrypt(enc_y)
        x = self.tanh(y)
        return x

    def compute_enc_gru_r(self, x, h):
        enc_r = self.sec_fc((self.enc_param["gru_ir.weight"], self.enc_param["gru_ir.bias"]), x, send_back=False)[0]
        enc_r += self.sec_fc((self.enc_param["gru_hr.weight"], self.enc_param["gru_hr.bias"]), h, send_back=False)[0]
        enc_r = self.send_enc_vector(enc_r)
        r = self.sec_sigmoid(enc_r)
        enc_r = self.encrypt_and_send(r)
        return enc_r

    def compute_enc_gru_z(self, x, h):
        enc_z = self.sec_fc((self.enc_param["gru_iz.weight"], self.enc_param["gru_iz.bias"]), x, send_back=False)[0]
        enc_z += self.sec_fc((self.enc_param["gru_hz.weight"], self.enc_param["gru_hz.bias"]), h, send_back=False)[0]
        enc_z = self.send_enc_vector(enc_z)
        z = self.sec_sigmoid(enc_z)
        enc_z = self.encrypt_and_send(z)
        return enc_z

    def compute_enc_gru_n(self, x, h, r):
        enc_n = self.sec_fc((self.enc_param["gru_in.weight"], self.enc_param["gru_in.bias"]), x, send_back=False)[0]
        enc_n_ = self.sec_fc((self.enc_param["gru_hn.weight"], self.enc_param["gru_hn.bias"]), h, send_back=False)[0]

        if not (h is None) and self.aggregated:
            enc_n_ = self.send_enc_vector(enc_n_)
            n_ = self.decrypt(enc_n_)
            enc_n_ = self.encrypt_and_send(n_)

        enc_n += enc_n_ * r
        enc_n = self.send_enc_vector(enc_n)
        n = self.sec_tanh(enc_n)
        enc_n = self.encrypt_and_send(n)

        return enc_n

    def compute_enc_gru_h(self, z, n, h):
        one_minus_z = self.plaintext_(np.ones(z.size())) - z
        enc_h = one_minus_z * n
        if not (h is None):
            enc_h += z * h
        enc_h = self.send_enc_vector(enc_h)
        return enc_h

    def sec_rnn_gru(self, enc_x_seq, enc_h_fc=None, enc_h_hardmard=None):
        dec_size = min(self.gru_input_size, self.max_len)
        for i in range(self.seq_len):
            enc_x = enc_x_seq[i]
            enc_r = self.compute_enc_gru_r(enc_x, enc_h_fc)
            enc_z = self.compute_enc_gru_z(enc_x, enc_h_fc)
            enc_n = self.compute_enc_gru_n(enc_x, enc_h_fc, enc_r)
            enc_h_hardmard = self.compute_enc_gru_h(enc_z, enc_n, enc_h_hardmard)

            if i < self.seq_len - 1:
                h = self.decrypt(enc_h_hardmard)
                enc_h_hardmard = self.encrypt_and_send(h)
                h = h.reshape(dec_size, dec_size)[:self.gru_output_size, :self.input_nb]
                enc_h_fc = self.preprocess_for_fc(h, h.shape[0], add_rows=self.gru_input_size-self.gru_output_size)

        h = self.decrypt(enc_h_hardmard).reshape(dec_size, dec_size)[:self.gru_output_size, :self.input_nb]
        return h

    def forward(self, enc_x_seq, enc_truth):

        h = self.sec_rnn_gru(enc_x_seq)

        enc_x = self.preprocess_for_fc(h, self.fc_output_size)
        enc_y = self.sec_fc((self.enc_param["fc.weight"], self.enc_param["fc.bias"]), enc_x)
        enc_pred = self.predict(enc_y)

        return self.sec_compare(enc_pred, enc_truth)

class HE_Logi(HEModel):
    def __init__(self, input_nb, input_size, output_size):
        super(HE_Logi, self).__init__()
        self.input_nb = input_nb
        self.input_shape = (-1, input_size)

        self.fc_input_size = input_size
        self.fc_output_size = output_size

    def init_model_param(self, param, return_bytes=False):
        self.enc_param = {}
        self.enc_param["fc.weight"], self.enc_param["fc.bias"] = self.encrypt_fc(param["fc.weight"], param["fc.bias"],
                                                                                 return_bytes=return_bytes)

    def encrypt_input(self, x):
        x = x.reshape(self.input_shape)
        x = np.transpose(x, (1, 0))
        return self.preprocess_for_fc(x, self.fc_output_size, return_bytes=True)

    def forward(self, enc_x, enc_truth):
        enc_y = self.sec_fc((self.enc_param["fc.weight"], self.enc_param["fc.bias"]), enc_x)
        enc_pred = self.predict(enc_y)

        return self.sec_compare(enc_pred, enc_truth)


class HE_BANK_Logi(HE_Logi):
    def __init__(self, input_nb=45):
        self.input_shape = (-1, 48)
        self.output_size = 2
        super(HE_BANK_Logi, self).__init__(input_nb, self.input_shape[1], self.output_size)
        self.n_processes = 10

class HE_AGNEWS_Logi(HE_Logi):
    def __init__(self, input_nb=45):
        self.input_shape = (-1, 300)
        self.output_size = 4
        super(HE_AGNEWS_Logi, self).__init__(input_nb, self.input_shape[1], self.output_size)
        self.n_processes = 10

class HE_DNN(HEModel):
    def __init__(self, input_nb, layer_nb):
        super(HE_DNN, self).__init__()
        self.input_nb = input_nb
        self.layer_nb = layer_nb

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

    def encrypt_input(self, x):
        x = x.reshape(self.input_shape)
        x = np.transpose(x, (1, 0))
        return self.preprocess_for_fc(x, self.hidden_layer_size, return_bytes=True)

    def sec_relu(self, enc_y):
        y = self.decrypt(enc_y).reshape(-1, self.input_nb)[:self.hidden_layer_size]
        x = self.relu(y)
        return x

    def forward(self, enc_x, enc_truth):
        enc_y = self.sec_fc((self.enc_param["input.weight"], self.enc_param["input.bias"]), enc_x)
        x = self.sec_relu(enc_y)

        for l in range(self.layer_nb):
            enc_x = self.preprocess_for_fc(x, self.hidden_layer_size)
            enc_y = self.sec_fc((self.enc_param[f"hidden.{l}.weight"], self.enc_param[f"hidden.{l}.bias"]), enc_x)
            x = self.sec_relu(enc_y)

        enc_x = self.preprocess_for_fc(x, self.output_size)
        enc_y = self.sec_fc((self.enc_param["output.weight"], self.enc_param["output.bias"]), enc_x)
        enc_pred = self.predict(enc_y)

        return self.sec_compare(enc_pred, enc_truth)

class HE_BANK_DNN(HE_DNN):
    def __init__(self, input_nb=45, layer_nb=5):
        super(HE_BANK_DNN, self).__init__(input_nb=input_nb, layer_nb=layer_nb)
        self.input_shape = (-1, 48)
        self.n_processes = 10

        self.hidden_layer_size = 48
        self.output_size = 2

class HE_AGNEWS_DNN(HE_DNN):
    def __init__(self, input_nb=45, layer_nb=5):
        super(HE_AGNEWS_DNN, self).__init__(input_nb=input_nb, layer_nb=layer_nb)
        self.input_shape = (-1, 300)
        self.n_processes = 10

        self.hidden_layer_size = 64
        self.output_size = 4

class HE_MNIST_DNN(HE_DNN):
    def __init__(self, input_nb=45, layer_nb=5):
        super(HE_MNIST_DNN, self).__init__(input_nb=input_nb, layer_nb=layer_nb)
        self.input_shape = (-1, 784)
        self.n_processes = 10

        self.hidden_layer_size = 64
        self.output_size = 10

class HE_mRNA_DNN(HE_DNN):
    def __init__(self, input_nb=45, layer_nb=5):
        super(HE_mRNA_DNN, self).__init__(input_nb=input_nb, layer_nb=layer_nb)
        self.input_shape = (-1, 640)
        self.n_processes = 10

        self.hidden_layer_size = 64
        self.output_size = 2