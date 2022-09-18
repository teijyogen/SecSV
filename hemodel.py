import torch
from torchvision import datasets
import torchvision.transforms as transforms
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

class EncModel:
    def __init__(self):
        self.context = None
        self.time_dict = {}

    def encrypt_(self, vector):
        start = time.process_time()
        enc_vec = ts.ckks_vector(self.context, vector.reshape(-1))
        self.time_dict["encryption"] += time.process_time() - start
        return enc_vec

    def decrypt_(self, enc_vec):
        start = time.process_time()
        vec = np.array(enc_vec.decrypt())
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

    def send_enc_vector_(self, enc_vec, return_bytes=False, nb_receviers=1):
        enc_vec_bytes = enc_vec.serialize()
        self.time_dict["communication"] += communicate(enc_vec_bytes) * nb_receviers
        if return_bytes:
            return enc_vec_bytes
        else:
            enc_vec = ts.CKKSVector.load(self.context, enc_vec_bytes)
            return enc_vec

    def send_enc_vector(self, enc_vec, return_bytes=False, nb_receviers=1):
        if type(enc_vec) == list:
            enc_vec_ls = []
            for ele in enc_vec:
                new_ele = self.send_enc_vector_(ele, return_bytes=return_bytes, nb_receviers=nb_receviers)
                enc_vec_ls.append(new_ele)
            return enc_vec_ls
        else:
            enc_vec = self.send_enc_vector_(enc_vec, return_bytes=return_bytes, nb_receviers=nb_receviers)
            return enc_vec

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

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class HEModel(EncModel):
    def __init__(self):
        super(HEModel, self).__init__()
        self.n_slots = 4096
        self.max_len = 64

    def encrypt(self, mat):
        enc_vec = self.encrypt_(mat)
        return enc_vec

    def encrypt_and_send(self, mat, return_bytes=False):
        enc_mat = self.encrypt(mat)
        enc_mat = self.send_enc_vector(enc_mat, return_bytes=return_bytes)
        return enc_mat

    def dotmul(self, mat1, mat2):
        if type(mat1) == list and type(mat2) == list:
            enc_y = mat1[0] * mat2[0]
            for i in range(1, len(mat1)):
                enc_y += mat1[i] * mat2[i]
        elif type(mat1) != list and type(mat2) != list:
            enc_y = mat1 * mat2
        else:
            raise TypeError("Dotmul not supported between list and tensor ")
        return enc_y

    def multiple_mat_dotmul(self, mat1_ls, mat2_ls):
        return self.dotmul(mat1_ls, mat2_ls)

    def encrypt_conv(self, conv_weight, conv_bias, kernel_len, conv_windows_nb):
        conv_weight = conv_weight.numpy()
        conv_bias = conv_bias.numpy()
        repeated_times = conv_windows_nb * self.input_nb
        enc_channels = []
        for weight, bias in zip(conv_weight, conv_bias):
            enc_weights_ic = []
            in_channels_nb = len(weight)
            for ic in range(in_channels_nb):
                ic_wt = weight[ic].reshape(-1)
                enc_weights = []
                # print(flat_wt.shape)
                for i in range(kernel_len ** 2):
                    rep_wt = ic_wt[i].repeat(repeated_times)
                    enc_weight = self.encrypt(rep_wt)
                    # enc_weight = self.encrypt(flat_wt[i].view(-1))
                    enc_weight = self.send_enc_vector(enc_weight)
                    enc_weights.append(enc_weight)

                enc_weights_ic.append(enc_weights)

            rep_bias = bias.reshape(-1).repeat(repeated_times)
            enc_bias = self.encrypt(rep_bias)
            # enc_bias = self.encrypt(bias.view(-1))
            enc_bias = self.send_enc_vector(enc_bias)
            enc_channels.append((enc_weights_ic, enc_bias))

        return enc_channels

    def preprocess_for_conv(self, x, windows_nb, kernel_len, stride, pad_width=((0, 0), (0, 0), (0, 0)), return_bypes=False):

        dk = int(windows_nb ** 0.5)
        padded_x = np.pad(x, pad_width)

        enc_features_ic = []
        in_channels_nb = padded_x.shape[0]
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

                    enc_feature = self.encrypt_and_send(feature, return_bytes=return_bypes)
                    enc_features.append(enc_feature)

            enc_features_ic.append(enc_features)

        return enc_features_ic

    def sec_conv(self, enc_conv, enc_features):
        enc_y_oc = []
        for (enc_wt, enc_bias) in enc_conv:
            start = time.process_time()
            enc_y = self.multiple_mat_dotmul(enc_wt[0], enc_features[0])
            for ic in range(1, len(enc_wt)):
                enc_y += self.multiple_mat_dotmul(enc_wt[ic], enc_features[ic])
            enc_y += enc_bias
            self.time_dict["HE computation"] += time.process_time() - start

            enc_y = self.send_enc_vector(enc_y)
            enc_y_oc.append(enc_y)

        return enc_y_oc

    def encrypt_fc(self, fc_weight, fc_bias, add_cols_to_hidden=0):
        fc_weight = fc_weight.numpy()
        fc_bias = fc_bias.numpy()


        if add_cols_to_hidden:
            fc_weight = np.pad(fc_weight, ((0, 0), (0, add_cols_to_hidden)))

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

        enc_row_channels = []
        for j in range(J):
            enc_col_channels = []
            for k in range(K):
                weight = fc_weight[j * max_len:(j + 1) * max_len, k * max_len:(k + 1) * max_len]
                enc_weight = self.enc_perm_mats(weight, weight.shape[0], left=True)
                enc_col_channels.append(enc_weight)

            bias = fc_bias[j * max_len:(j + 1) * max_len, :]
            enc_bias = self.encrypt_and_send(bias)

            enc_row_channels.append((enc_col_channels, enc_bias))

        return enc_row_channels

    def preprocess_for_fc(self, matrix, d_out, return_bytes=False, add_rows_to_hidden=0):
        if add_rows_to_hidden:
           matrix = np.pad(matrix, ((0, add_rows_to_hidden), (0, 0)))

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
                enc_mat = self.enc_perm_mats(sub_mat, d_out, left=False, return_bytes=return_bytes)
                enc_col_channels.append(enc_mat)
            enc_row_channels.append(enc_col_channels)

        return enc_row_channels

    def sec_fc(self, enc_fc, enc_xs, send_back=True):

        enc_row_channels = []
        for j in range(len(enc_fc)):
            enc_wts, enc_bias = enc_fc[j]

            start = time.process_time()
            if enc_xs is not None:
                enc_y = self.he_matmul(enc_wts[0], enc_xs[j][0])
                for k in range(1, len(enc_wts)):
                    enc_y += self.he_matmul(enc_wts[k], enc_xs[j][k])
                enc_y += enc_bias
            else:
                enc_y = enc_bias
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
        mat_nb = len(mat1_ls)

        enc_y = mat1_ls[0] * mat2_ls[0]
        for i in range(1, mat_nb):
            enc_y += mat1_ls[i] * mat2_ls[i]

        d_in = int(mat1_ls[0].size() ** 0.5)
        d_out = mat_nb
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

    def encrypt_gru(self, gru_weights, gru_biases, add_cols_to_hidden=0):
        hidden_size = int(gru_weights.shape[0] / 3)
        enc_paras = []
        for i in range(3):
            weight = gru_weights[i*hidden_size:(i+1)*hidden_size]
            bias = gru_biases[i*hidden_size:(i+1)*hidden_size]
            enc_para = self.encrypt_fc(weight, bias, add_cols_to_hidden=add_cols_to_hidden)
            enc_paras.append(enc_para)

        return enc_paras

    def predict(self, enc_y, mat_len, output_size):

        y = self.decrypt(enc_y)
        y = y.reshape(mat_len, mat_len)[:output_size, :self.truth_nb]
        pred = np.argmax(y, axis=0)

        enc_pred = self.encrypt(pred)
        enc_pred = self.send_enc_vector(enc_pred)

        return enc_pred

    def encrypt_truth(self, truth):
        enc_truth = self.encrypt(truth)
        enc_truth_bytes = self.send_enc_vector(enc_truth, return_bytes=True)
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
    def __init__(self, input_nb=64):
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

        self.context = None
        self.enc_conv1 = None
        self.enc_fc1 = None
        self.enc_fc2 = None

    def init_model_paras(self, context, model_paras):
        self.context = context
        self.enc_conv1 = self.encrypt_conv(model_paras["conv1.weight"], model_paras["conv1.bias"],
                                           self.conv1_kernel_len, self.conv1_windows_nb)
        self.enc_fc1 = self.encrypt_fc(model_paras["fc1.weight"], model_paras["fc1.bias"])
        self.enc_fc2 = self.encrypt_fc(model_paras["fc2.weight"], model_paras["fc2.bias"])

    def clear_model_paras(self):
        self.context = None
        self.enc_conv1 = None
        self.enc_fc1 = None
        self.enc_fc2 = None

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

        enc_y_channel = self.sec_conv(self.enc_conv1, enc_x)
        x = self.sec_square(enc_y_channel, self.fc1_input_size)

        enc_x = self.preprocess_for_fc(x, self.fc1_output_size)
        enc_y = self.sec_fc(self.enc_fc1, enc_x)
        x = self.sec_square(enc_y, self.fc1_output_size)

        enc_x = self.preprocess_for_fc(x, self.fc2_output_size)
        enc_y = self.sec_fc(self.enc_fc2, enc_x)
        enc_pred = self.predict(enc_y, self.fc2_input_size, self.fc2_output_size)

        return self.sec_compare(enc_pred, enc_truth)


class HE_mRNA_RNN(HEModel):
    def __init__(self, input_nb=32):
        super(HE_mRNA_RNN, self).__init__()
        self.input_nb = input_nb
        self.truth_nb = input_nb
        self.input_shape = (-1, 10, 64)
        self.n_processes = 5

        self.gru_input_size = self.input_shape[2]
        self.gru_output_size = 32
        self.seq_len = self.input_shape[1]

        self.fc_input_size = 32
        self.fc_output_size = 2
        self.fc_channel_nb = 1
        self.fc_residual_nb = 0

        self.context = None
        self.enc_gru_ir = None
        self.enc_gru_hr = None
        self.enc_gru_iz = None
        self.enc_gru_hz = None
        self.enc_gru_in = None
        self.enc_gru_hn = None
        self.enc_fc = None

    def init_model_paras(self, context, model_paras):
        self.context = context
        self.enc_gru_ir, self.enc_gru_iz, self.enc_gru_in = self.encrypt_gru(model_paras["rnn.weight_ih_l0"],
                                                            model_paras["rnn.bias_ih_l0"])
        self.enc_gru_hr, self.enc_gru_hz, self.enc_gru_hn = self.encrypt_gru(model_paras["rnn.weight_hh_l0"],
                                                            model_paras["rnn.bias_hh_l0"],
                                                            add_cols_to_hidden=self.gru_input_size - self.gru_output_size)
        self.enc_fc = self.encrypt_fc(model_paras["fc.weight"], model_paras["fc.bias"])

    def clear_model_paras(self):
        self.context = None
        self.enc_gru_ir = None
        self.enc_gru_hr = None
        self.enc_gru_iz = None
        self.enc_gru_hz = None
        self.enc_gru_in = None
        self.enc_gru_hn = None
        self.enc_fc = None

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
        enc_r = self.sec_fc(self.enc_gru_ir, x, send_back=False)[0]
        enc_r += self.sec_fc(self.enc_gru_hr, h, send_back=False)[0]
        enc_r = self.send_enc_vector(enc_r)
        r = self.sec_sigmoid(enc_r)
        enc_r = self.encrypt(r)
        enc_r = self.send_enc_vector(enc_r)
        return enc_r

    def compute_enc_gru_z(self, x, h):
        enc_z = self.sec_fc(self.enc_gru_iz, x, send_back=False)[0]
        enc_z += self.sec_fc(self.enc_gru_hz, h, send_back=False)[0]
        enc_z = self.send_enc_vector(enc_z)
        z = self.sec_sigmoid(enc_z)
        enc_z = self.encrypt(z)
        enc_z = self.send_enc_vector(enc_z)
        return enc_z

    def compute_enc_gru_n(self, x, h, r):
        enc_n = self.sec_fc(self.enc_gru_in, x, send_back=False)[0]
        enc_n += self.sec_fc(self.enc_gru_hn, h, send_back=False)[0] * r
        enc_n = self.send_enc_vector(enc_n)
        n = self.sec_tanh(enc_n)
        enc_n = self.encrypt(n)
        enc_n = self.send_enc_vector(enc_n)

        return enc_n

    def compute_enc_gru_h(self, z, n, h):
        one_minus_z = self.plaintext_(np.ones(z.size())) - z
        enc_h = one_minus_z * n
        if not (h is None):
            enc_h += z * h
        enc_h = self.send_enc_vector(enc_h)
        return enc_h

    def sec_rnn_gru(self, enc_x_seq, input_size, hidden_size, enc_h_fc=None, enc_h_hardmard=None):
        shape = [input_size, input_size]
        for i in range(self.seq_len):
            enc_x = enc_x_seq[i]
            enc_r = self.compute_enc_gru_r(enc_x, enc_h_fc)
            enc_z = self.compute_enc_gru_z(enc_x, enc_h_fc)
            enc_n = self.compute_enc_gru_n(enc_x, enc_h_fc, enc_r)
            enc_h_hardmard = self.compute_enc_gru_h(enc_z, enc_n, enc_h_hardmard)

            if i < self.seq_len - 1:
                h = self.decrypt(enc_h_hardmard)
                enc_h_hardmard = self.encrypt(h)
                enc_h_hardmard = self.send_enc_vector(enc_h_hardmard)
                h = h.reshape(shape)[:hidden_size, :self.input_nb]
                enc_h_fc = self.preprocess_for_fc(h, h.shape[0], add_rows_to_hidden=input_size-hidden_size)

        return self.decrypt(enc_h_hardmard).reshape(shape)[:hidden_size, :self.input_nb]

    def forward(self, enc_x_seq, enc_truth):

        h = self.sec_rnn_gru(enc_x_seq, self.gru_input_size, self.gru_output_size)

        enc_x = self.preprocess_for_fc(h, self.fc_output_size)
        enc_y = self.sec_fc(self.enc_fc, enc_x)
        enc_pred = self.predict(enc_y, self.fc_input_size, self.fc_output_size)

        return self.sec_compare(enc_pred, enc_truth)


class HE_AGNEWS_Logi(HEModel):
    def __init__(self, input_nb=64):
        super(HE_AGNEWS_Logi, self).__init__()
        self.input_nb = input_nb
        self.input_shape = (-1, 300)
        self.n_processes = 10

        self.fc_input_size = 300
        self.fc_output_size = 4

        self.context = None
        self.enc_fc = None

    def init_model_paras(self, context, model_paras):
        self.context = context
        self.enc_fc = self.encrypt_fc(model_paras["fc.weight"], model_paras["fc.bias"])

    def clear_model_paras(self):
        self.context = None
        self.enc_fc = None

    def encrypt_input(self, x):
        x = x.reshape(self.input_shape)
        x = np.transpose(x, (1, 0))
        return self.preprocess_for_fc(x, self.fc_output_size, return_bytes=True)

    def forward(self, enc_x, enc_truth):
        enc_y = self.sec_fc(self.enc_fc, enc_x)
        enc_pred = self.predict(enc_y, self.max_len, self.fc_output_size)

        return self.sec_compare(enc_pred, enc_truth)

class HE_BANK_Logi(HEModel):
    def __init__(self, input_nb=48):
        super(HE_BANK_Logi, self).__init__()
        self.input_nb = input_nb
        self.input_shape = (-1, 48)
        self.n_processes = 10

        self.fc_input_size = 48
        self.fc_output_size = 2

        self.context = None
        self.enc_fc = None

    def init_model_paras(self, context, model_paras):
        self.context = context
        self.enc_fc = self.encrypt_fc(model_paras["fc.weight"], model_paras["fc.bias"])

    def clear_model_paras(self):
        self.context = None
        self.enc_fc = None

    def encrypt_input(self, x):
        x = x.reshape(self.input_shape)
        x = np.transpose(x, (1, 0))
        return self.preprocess_for_fc(x, self.fc_output_size, return_bytes=True)

    def forward(self, enc_x, enc_truth):
        enc_y = self.sec_fc(self.enc_fc, enc_x)
        enc_pred = self.predict(enc_y, self.fc_input_size, self.fc_output_size)

        return self.sec_compare(enc_pred, enc_truth)