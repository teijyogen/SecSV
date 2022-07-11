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

class EncModel:
    def __init__(self):
        self.context = None
        self.time_dict = {}

    def encrypt(self, vector):
        start = time.process_time()
        enc_vec = ts.ckks_vector(self.context, vector.reshape(-1))
        self.time_dict["encryption"] += time.process_time() - start
        return enc_vec

    def decrypt(self, enc_vec):
        start = time.process_time()
        vec = np.array(enc_vec.decrypt())
        self.time_dict["decryption"] += time.process_time() - start
        return vec

    def plaintext(self, mat):
        return ts.plain_tensor(mat.reshape(-1), dtype='float')

    def send_enc_vector(self, enc_vec, return_bytes=False, nb_receviers=1):
        enc_vec_bytes = enc_vec.serialize()
        self.time_dict["communication"] += communicate(enc_vec_bytes) * nb_receviers
        if return_bytes:
            return enc_vec_bytes
        else:
            enc_vec = ts.CKKSVector.load(self.context, enc_vec_bytes)
            return enc_vec

    def send_big_enc_vector(self, enc_vec, return_bytes=False):
        enc_vec_ls = []
        for ele in enc_vec:
            new_ele = self.send_enc_vector(ele, return_bytes=return_bytes)
            enc_vec_ls.append(new_ele)
        return enc_vec_ls

    def big_mat_dotmul(self, mat1, mat2):
        mat_nb = len(mat1)
        enc_y = mat1[0] * mat2[0]
        for i in range(1, mat_nb):
            enc_y += mat1[i] * mat2[i]

        return enc_y

    def encrypt_conv(self, conv_weight, conv_bias, kernel_len, conv_windows_nb, in_channels_nb=1, vir_channels_nb=1):
        conv_weight = conv_weight.numpy()
        conv_bias = conv_bias.numpy()
        repeated_times = math.ceil(conv_windows_nb / vir_channels_nb) * self.input_nb
        enc_channels = []
        for weight, bias in zip(conv_weight, conv_bias):
            enc_weights_ic = []
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

    # def shift_byrows(self, enc_vec, n_cols, steps=1):
    #     return enc_vec.rotate_vector(steps*n_cols)
    #
    # def shift_bycols(self, enc_vec, n_cols, step=1):
    #     n_rows = enc_vec.shape[0] // n_cols
    #     mask = np.zeros(enc_vec.shape[0]).reshape(n_rows, n_cols)
    #     mask[:, :1] = 1.0
    #     mask = mask.reshape(-1)
    #
    #     temp = enc_vec * mask
    #     res = enc_vec - temp
    #
    #     res.rotate_vector_inplace(step)
    #     temp.rotate_vector_inplace(step-n_cols)
    #
    #     return res + temp

    def preprocess_for_conv(self, x, windows_nb, kernel_len, stride, pad_width=((0, 0), (0, 0), (0, 0)), in_channels_nb=1, vir_channels_nb=1, return_bypes=False):

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
                    chunk_size = int(math.ceil(feature.shape[0] / vir_channels_nb))
                    added_rows_nb = chunk_size * vir_channels_nb - feature.shape[0]
                    padded_feature = np.pad(feature, ((0, added_rows_nb), (0, 0)))

                    enc_features_vc = []
                    for vc in range(vir_channels_nb):
                        enc_feature_vc = self.encrypt(padded_feature[vc*chunk_size:(vc+1)*chunk_size, :])
                        enc_feature_vc = self.send_enc_vector(enc_feature_vc, return_bytes=return_bypes)
                        enc_features_vc.append(enc_feature_vc)

                    enc_features.append(enc_features_vc)

            enc_features_ic.append(enc_features)

        enc_features_ic = np.array(enc_features_ic)
        enc_features_vc = np.transpose(enc_features_ic, (2, 0, 1))

        return enc_features_vc.tolist()

    def sec_conv(self, enc_conv, enc_features):
        enc_y_channel = []
        for (enc_wt, enc_bias) in enc_conv:
            enc_y_vc = []
            for vc in range(len(enc_features)):
                start = time.process_time()
                enc_y = self.big_mat_dotmul(enc_wt[0], enc_features[vc][0])
                for ic in range(1, len(enc_wt)):
                    enc_y += self.big_mat_dotmul(enc_wt[ic], enc_features[vc][ic])
                enc_y += enc_bias
                self.time_dict["HE computation"] += time.process_time() - start

                enc_y = self.send_enc_vector(enc_y)
                enc_y_vc.append(enc_y)
            enc_y_channel.append(enc_y_vc)

        return enc_y_channel

    def encrypt_fc(self, fc_weight, fc_bias, vir_channels_nb=1, add_cols_to_hidden=0):
        fc_weight = fc_weight.numpy()
        fc_bias = fc_bias.numpy()

        if add_cols_to_hidden:
            fc_weight = np.pad(fc_weight, ((0, 0), (0, add_cols_to_hidden)))

        chunk_size = math.ceil(fc_weight.shape[1] / vir_channels_nb)
        added_cols_nb = chunk_size*vir_channels_nb - fc_weight.shape[1]
        padded_weight = np.pad(fc_weight, ((0, 0), (0, added_cols_nb)))
        mat_len = max(chunk_size, fc_weight.shape[0])

        enc_vc = []
        for vc in range(vir_channels_nb):
            weight = padded_weight[:, vc * chunk_size:(vc + 1) * chunk_size]
            enc_vc.append(self.enc_perm_mats(weight, weight.shape[0], left=True))

        fc_bias = fc_bias.reshape(-1, 1)
        subdim = fc_bias.shape[0]
        rep_bias = fc_bias.repeat(self.input_nb, axis=1)
        padded_bias = np.pad(rep_bias, ((0, mat_len-subdim), (0, mat_len-self.input_nb)))

        if mat_len ** 2 > self.n_slots:
            enc_bias = self.encrypt_and_send(padded_bias, n_rows_left=subdim, divide=True, residual=True)
        else:
            enc_bias = self.encrypt_and_send(padded_bias)

        return (enc_vc, enc_bias)

    def preprocess_for_fc(self, matrix, n_rows_left, vir_channels_nb=1, return_bytes=False, add_rows_to_hidden=0):
        if add_rows_to_hidden:
           matrix = np.pad(matrix, ((0, add_rows_to_hidden), (0, 0)))

        chunk_size = math.ceil(matrix.shape[0] / vir_channels_nb)
        added_rows_nb = chunk_size*vir_channels_nb - matrix.shape[0]
        padded_mat = np.pad(matrix, ((0, added_rows_nb), (0, 0)))

        enc_mats_vc = []
        for vc in range(vir_channels_nb):
            sub_mat = padded_mat[chunk_size * vc:chunk_size * (vc + 1), :]
            enc_mats = self.enc_perm_mats(sub_mat, n_rows_left, left=False, return_bytes=return_bytes)
            enc_mats_vc.append(enc_mats)

        return enc_mats_vc

    def sec_fc(self, enc_fc, enc_x, residual_nb=0, send_back=True):
        start = time.process_time()
        enc_wts, enc_bias = enc_fc

        if enc_x is None:
            enc_y = enc_bias
        else:
            enc_y = self.he_matmul(enc_wts[0], enc_x[0], residual_nb)
            for i in range(1, len(enc_wts)):
                enc_y += self.he_matmul(enc_wts[i], enc_x[i], residual_nb)

            if type(enc_bias) == list:
                enc_y += enc_bias[0]
            else:
                enc_y += enc_bias
            self.time_dict["HE computation"] += time.process_time() - start

        if send_back:
            enc_y = self.send_enc_vector(enc_y)

        return enc_y

    def enc_perm_mats(self, matrix, n_rows_left, left=True, return_bytes=False):

        nrows = matrix.shape[0]
        ncols = matrix.shape[1]

        if nrows == ncols:
            square_mat = matrix
        else:
            mat_len = max([nrows, ncols, n_rows_left])
            square_mat = np.zeros((mat_len, mat_len))

            if left:
                if nrows < ncols:
                    for i in range(mat_len // nrows):
                        square_mat[nrows * i: nrows * (i + 1), :] = matrix
                else:
                    square_mat[:, :ncols] = matrix
            else:
                square_mat[:nrows, :ncols] = matrix

        enc_mats = self.generate_perms(square_mat, n_rows_left, left=left, return_bytes=return_bytes)

        return enc_mats

    def encrypt_big_message(self, big_mat, subdim, residual=False):
        start = time.process_time()

        enc_vecs = []

        if residual:
            nb_mat = 1
        else:
            nb_mat = big_mat.shape[0] // subdim

        for i in range(nb_mat):
            vec = big_mat[i*subdim:(i+1)*subdim, :].reshape(-1)
            enc_vecs.append(ts.ckks_vector(self.context, vec))


        self.time_dict["encryption"] += time.process_time() - start

        return enc_vecs

    def encrypt_and_send(self, mat, divide=False, n_rows_left=-1, return_bytes=False, residual=False):
        if divide:
            enc_mat = self.encrypt_big_message(mat, n_rows_left, residual=residual)
            enc_mat = self.send_big_enc_vector(enc_mat, return_bytes=return_bytes)
        else:
            enc_mat = self.encrypt(mat)
            enc_mat = self.send_enc_vector(enc_mat, return_bytes=return_bytes)

        return enc_mat

    def generate_perms(self, mat, n_rows_left, left=True, return_bytes=False):
        mat_len = mat.shape[0]
        parallel_mat_nb = n_rows_left
        residual_mat_nb = mat_len % n_rows_left
        divide = mat_len ** 2 > self.n_slots

        if left:
            mat = sigma(mat)
            enc_mats = [self.encrypt_and_send(mat, divide=divide, n_rows_left=n_rows_left, return_bytes=return_bytes)]

            for i in range(1, parallel_mat_nb):
                # mat = np.roll(mat, -1, axis=1)
                # enc_mats.append(
                #     self.encrypt_and_send(mat, divide=divide, n_rows_left=n_rows_left, return_bytes=return_bytes))
                new_mat = np.roll(mat, -i, axis=1)
                enc_mats.append(
                    self.encrypt_and_send(new_mat, divide=divide, n_rows_left=n_rows_left, return_bytes=return_bytes))

            # mat = np.roll(mat, -(mat_len - residual_mat_nb - parallel_mat_nb), axis=1)
            for i in range(mat_len - residual_mat_nb, mat_len):
                # mat = np.roll(mat, -1, axis=1)
                # enc_mats.append(self.encrypt_and_send(mat, divide=divide, residual=True, return_bytes=return_bytes))
                new_mat = np.roll(mat, -i, axis=1)
                enc_mats.append(self.encrypt_and_send(new_mat, divide=divide, n_rows_left=n_rows_left, residual=True, return_bytes=return_bytes))
        else:
            mat = tau(mat)
            enc_mats = [self.encrypt_and_send(mat, divide=divide, n_rows_left=n_rows_left, return_bytes=return_bytes)]
            for i in range(1, parallel_mat_nb):
                # mat = np.roll(mat, -1, axis=0)
                # enc_mats.append(
                #     self.encrypt_and_send(mat, divide=divide, n_rows_left=n_rows_left, return_bytes=return_bytes))
                new_mat = np.roll(mat, -i, axis=0)
                enc_mats.append(
                    self.encrypt_and_send(new_mat, divide=divide, n_rows_left=n_rows_left, return_bytes=return_bytes))

            # mat = np.roll(mat, -(mat_len - residual_mat_nb - parallel_mat_nb), axis=0)
            for i in range(mat_len - residual_mat_nb, mat_len):
                # mat = np.roll(mat, -1, axis=0)
                # enc_mats.append(self.encrypt_and_send(mat, divide=divide, residual=True, return_bytes=return_bytes))
                new_mat = np.roll(mat, -i, axis=0)
                enc_mats.append(self.encrypt_and_send(new_mat, divide=divide, n_rows_left=n_rows_left, residual=True, return_bytes=return_bytes))

        return enc_mats

    def he_matmul(self, mat1_ls, mat2_ls, nb_residual_mat=0):
        mat_nb = len(mat1_ls) - nb_residual_mat

        if type(mat1_ls[0]) == list:
            enc_y = self.big_mat_dotmul(mat1_ls[0], mat2_ls[0])
            for i in range(1, mat_nb):
                res = self.big_mat_dotmul(mat1_ls[i], mat2_ls[i])
                enc_y += res
            return enc_y
        else:
            mat_len = int(mat1_ls[0].size() ** 0.5)
            subdim = mat_nb

            enc_y = mat1_ls[0] * mat2_ls[0]

            for i in range(1, mat_nb):
                enc_y += mat1_ls[i] * mat2_ls[i]

            if subdim < mat_len:
                double_times = int(math.log(mat_len / subdim, 2))

                if double_times > 0:
                    result = enc_y + enc_y.rotate_vector(mat_len * subdim * 2 ** 0)
                    for k in range(1, double_times):
                        result += result.rotate_vector(mat_len * subdim * 2 ** k)
                else:
                    result = copy.deepcopy(enc_y)

                if 2 ** double_times < int(mat_len / subdim):
                    result += enc_y.rotate_vector_inplace(2 ** double_times * mat_len * subdim)
                    for j in range(2 ** double_times + 1, int(mat_len / subdim)):
                        result += enc_y.rotate_vector_inplace(mat_len * subdim)

                if nb_residual_mat:
                    for i in range(mat_nb, mat_nb+nb_residual_mat):
                        result += mat1_ls[i] * mat2_ls[i]

                return result
            else:
                return enc_y

    def encrypt_gru(self, gru_weights, gru_biases, vir_channel_nb, add_cols_to_hidden=0):
        hidden_size = int(gru_weights.shape[0] / 3)
        enc_paras = []
        for i in range(3):
            weight = gru_weights[i*hidden_size:(i+1)*hidden_size]
            bias = gru_biases[i*hidden_size:(i+1)*hidden_size]
            enc_para = self.encrypt_fc(weight, bias, vir_channel_nb, add_cols_to_hidden=add_cols_to_hidden)
            enc_paras.append(enc_para)

        return enc_paras

    def predict(self, enc_y, shape, output_size):

        y = self.decrypt(enc_y)
        y = y.reshape(shape)[:output_size, :self.truth_nb]
        pred = np.argmax(y, axis=0)

        enc_pred = self.encrypt(pred.reshape(-1))
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

        self.conv1_stride = 3
        self.conv1_kernel_len = 7
        self.conv1_in_channel_nb = 1
        self.conv1_out_channel_nb = 4
        self.conv1_windows_nb = 8 * 8
        self.conv1_padding = ((0, 0), (0, 0), (0, 0), (0, 0))
        self.conv1_vir_channel_nb = 1

        self.fc1_input_size = 256
        self.fc1_output_size = 64
        self.fc1_channel_nb = 4
        self.fc1_residual_nb = max(math.ceil(self.fc1_input_size / self.fc1_channel_nb), self.fc1_output_size) % self.fc1_output_size

        self.fc2_input_size = 64
        self.fc2_output_size = 10
        self.fc2_channel_nb = 1
        self.fc2_residual_nb = max(math.ceil(self.fc2_input_size / self.fc2_channel_nb), self.fc2_output_size) % self.fc2_output_size

        self.context = None
        self.enc_conv1 = None
        self.enc_fc1 = None
        self.enc_fc2 = None

    def init_model_paras(self, context, model_paras):
        self.context = context
        self.enc_conv1 = self.encrypt_conv(model_paras["conv1.weight"], model_paras["conv1.bias"],
                                           self.conv1_kernel_len, self.conv1_windows_nb, self.conv1_in_channel_nb,
                                           self.conv1_vir_channel_nb)

        self.enc_fc1 = self.encrypt_fc(model_paras["fc1.weight"], model_paras["fc1.bias"], self.fc1_channel_nb)
        self.enc_fc2 = self.encrypt_fc(model_paras["fc2.weight"], model_paras["fc2.bias"], self.fc2_channel_nb)

    def clear_model_paras(self):
        self.context = None
        self.enc_conv1 = None
        self.enc_fc1 = None
        self.enc_fc2 = None

    def encrypt_input(self, x):
        x = x.reshape(self.input_nb, self.conv1_in_channel_nb, self.image_len, self.image_len)
        x = np.transpose(x, (1, 2, 3, 0))
        return self.preprocess_for_conv(x, self.conv1_windows_nb, self.conv1_kernel_len, self.conv1_stride,
                                        self.conv1_padding, self.conv1_in_channel_nb, self.conv1_vir_channel_nb, True)

    def sec_square1(self, enc_y):
        y_oc = []
        for oc in range(len(enc_y)):
            y_vc = []
            for vc in range(len(enc_y[0])):
                y = self.decrypt(enc_y[oc][vc])
                y_vc.append(y)
            y = np.array(y_vc).reshape(-1, self.input_nb)[:self.conv1_windows_nb, :]
            y = self.square(y)
            y_oc.append(y)
        x = np.array(y_oc).reshape(-1, self.input_nb)
        return x

    def sec_square2(self, enc_y, shape, n_rows):
        y = self.decrypt(enc_y).reshape(shape)[:n_rows, :self.input_nb]
        x = self.square(y)
        return x

    def forward(self, enc_x, enc_truth):

        enc_y_channel = self.sec_conv(self.enc_conv1, enc_x)
        x = self.sec_square1(enc_y_channel)

        enc_x = self.preprocess_for_fc(x, self.fc1_output_size, vir_channels_nb=self.fc1_channel_nb)
        enc_y = self.sec_fc(self.enc_fc1, enc_x, residual_nb=self.fc1_residual_nb)
        mat_len = self.fc1_output_size
        x = self.sec_square2(enc_y, [mat_len, mat_len], self.fc1_output_size)

        enc_x = self.preprocess_for_fc(x, self.fc2_output_size, vir_channels_nb=self.fc2_channel_nb)
        enc_y = self.sec_fc(self.enc_fc2, enc_x, residual_nb=self.fc2_residual_nb)
        mat_len = self.fc2_input_size
        enc_pred = self.predict(enc_y, [mat_len, mat_len], self.fc2_output_size)

        return self.sec_compare(enc_pred, enc_truth)

# class HE_CNN2_CIFAR(HEModel):
#     def __init__(self, input_nb=34):
#         super(HE_CNN2_CIFAR, self).__init__()
#         self.input_nb = input_nb
#         self.input_shape = (-1, 3, 32, 32)
#         self.image_len = 32
#
#         self.conv1_stride = 1
#         self.conv1_kernel_len = 5
#         self.conv1_in_channel_nb = 3
#         self.conv1_out_channel_nb = 6
#         self.conv1_windows_nb = 28 * 28
#         self.conv1_padding = ((0, 0), (0, 0), (0, 0), (0, 0))
#         self.conv1_vir_channel_nb = 7
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
#         self.fc1_channel_nb = 4
#         self.fc1_residual_nb = max(math.ceil(self.fc1_input_size / self.fc1_channel_nb), self.fc1_output_size) % self.fc1_output_size
#
#         self.fc2_input_size = 120
#         self.fc2_output_size = 84
#         self.fc2_channel_nb = 1
#         self.fc2_residual_nb = max(math.ceil(self.fc2_input_size / self.fc2_channel_nb), self.fc2_output_size) % self.fc2_output_size
#
#         self.fc3_input_size = 84
#         self.fc3_output_size = 10
#         self.fc3_channel_nb = 1
#         self.fc3_residual_nb = max(math.ceil(self.fc3_input_size / self.fc3_channel_nb), self.fc3_output_size) % self.fc3_output_size
#
#         self.context = None
#         self.enc_conv1 = None
#         self.enc_conv2 = None
#         self.enc_fc1 = None
#         self.enc_fc2 = None
#         self.enc_fc3 = None
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
#     def encrypt_input(self, x):
#         x = x.reshape(self.input_nb, self.conv1_in_channel_nb, self.image_len, self.image_len)
#         x = np.transpose(x, (1, 2, 3, 0))
#         return self.preprocess_for_conv(x, self.conv1_windows_nb, self.conv1_kernel_len, self.conv1_stride,
#                                         self.conv1_padding, self.conv1_in_channel_nb, self.conv1_vir_channel_nb, True)
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
#             y = self.relu(y)
#             y = self.maxpool(y)
#             y_oc.append(y)
#         x = np.array(y_oc)
#         return x
#
#     def sec_relu(self, enc_y, shape, n_rows):
#         y = self.decrypt(enc_y).reshape(shape)[:n_rows, :self.input_nb]
#         x = self.relu(y)
#         return x
#
#     def forward(self, enc_x, enc_truth):
#
#         enc_y_channel = self.sec_conv(self.enc_conv1, enc_x)
#         x = self.sec_relu_maxpool(enc_y_channel, [int(self.conv1_windows_nb ** 0.5), int(self.conv1_windows_nb ** 0.5), self.input_nb], self.conv1_windows_nb)
#
#         enc_x = self.preprocess_for_conv(x, self.conv2_windows_nb, self.conv2_kernel_len, self.conv2_stride,
#                                          self.conv2_padding, self.conv2_in_channel_nb, self.conv2_vir_channel_nb)
#         enc_y_channel = self.sec_conv(self.enc_conv2, enc_x)
#         x = self.sec_relu_maxpool(enc_y_channel, [int(self.conv2_windows_nb ** 0.5), int(self.conv2_windows_nb ** 0.5), self.input_nb], self.conv2_windows_nb).reshape(self.fc1_input_size, self.input_nb)
#
#         enc_x = self.preprocess_for_fc(x, self.fc1_output_size, vir_channels_nb=self.fc1_channel_nb)
#         enc_y = self.sec_fc(self.enc_fc1, enc_x, residual_nb=self.fc1_residual_nb)
#         x = self.sec_relu(enc_y, [self.fc1_output_size, self.fc1_output_size], self.fc1_output_size).reshape(self.fc2_input_size, self.input_nb)
#
#         enc_x = self.preprocess_for_fc(x, self.fc2_output_size, vir_channels_nb=self.fc2_channel_nb)
#         enc_y = self.sec_fc(self.enc_fc2, enc_x, residual_nb=self.fc2_residual_nb)
#         x = self.sec_relu(enc_y, [self.fc2_output_size, self.fc2_input_size], self.fc2_output_size).reshape(self.fc3_input_size, self.input_nb)
#
#         enc_x = self.preprocess_for_fc(x, self.fc3_output_size, vir_channels_nb=self.fc3_channel_nb)
#         enc_y = self.sec_fc(self.enc_fc3, enc_x, residual_nb=self.fc3_residual_nb)
#         enc_pred = self.predict(enc_y, [self.fc3_output_size, self.fc3_input_size], self.fc3_output_size)
#
#         return self.sec_compare(enc_pred, enc_truth)
#
#     def __call__(self, *args, **kwargs):
#         return self.forward(*args, **kwargs)


class HE_mRNA_RNN(HEModel):
    def __init__(self, input_nb=32):
        super(HE_mRNA_RNN, self).__init__()
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
                                                            model_paras["rnn.bias_ih_l0"], self.gru_channel_nb)
        self.enc_gru_hr, self.enc_gru_hz, self.enc_gru_hn = self.encrypt_gru(model_paras["rnn.weight_hh_l0"],
                                                            model_paras["rnn.bias_hh_l0"], self.gru_channel_nb,
                                                            add_cols_to_hidden=self.gru_input_size - self.gru_output_size)
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

    def encrypt_input(self, x):
        x = x.reshape(self.input_nb, self.seq_len, self.gru_input_size)
        x = np.transpose(x, (1, 2, 0))
        inputs = []
        for i in range(self.seq_len):
            input = x[i]
            input = self.preprocess_for_fc(input, self.gru_output_size, self.gru_channel_nb, return_bytes=True)
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
        enc_r = self.sec_fc(self.enc_gru_ir, x, send_back=False)
        enc_r += self.sec_fc(self.enc_gru_hr, h, send_back=False)
        enc_r = self.send_enc_vector(enc_r)
        r = self.sec_sigmoid(enc_r)
        enc_r = self.encrypt(r)
        enc_r = self.send_enc_vector(enc_r)
        return enc_r

    def compute_enc_gru_z(self, x, h):
        enc_z = self.sec_fc(self.enc_gru_iz, x, send_back=False)
        enc_z += self.sec_fc(self.enc_gru_hz, h, send_back=False)
        enc_z = self.send_enc_vector(enc_z)
        z = self.sec_sigmoid(enc_z)
        enc_z = self.encrypt(z)
        enc_z = self.send_enc_vector(enc_z)
        return enc_z

    def compute_enc_gru_n(self, x, h, r):
        enc_n = self.sec_fc(self.enc_gru_in, x, send_back=False)
        enc_n += self.sec_fc(self.enc_gru_hn, h, send_back=False) * r
        enc_n = self.send_enc_vector(enc_n)
        n = self.sec_tanh(enc_n)
        enc_n = self.encrypt(n)
        enc_n = self.send_enc_vector(enc_n)

        return enc_n

    def compute_enc_gru_h(self, z, n, h):
        one_minus_z = self.plaintext(np.ones(z.size())) - z
        enc_h = one_minus_z * n
        if not (h is None):
            enc_h += z * h
        enc_h = self.send_enc_vector(enc_h)
        return enc_h

    def sec_rnn_gru(self, enc_x_seq, input_size, hidden_size, channels_nb, enc_h_fc=None, enc_h_hardmard=None):
        mat_len = max(math.ceil(input_size / channels_nb), hidden_size)
        shape = [mat_len, mat_len]
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

        h = self.sec_rnn_gru(enc_x_seq, self.gru_input_size, self.gru_output_size, self.gru_channel_nb)

        enc_x = self.preprocess_for_fc(h, self.fc_output_size, vir_channels_nb=self.fc_channel_nb)
        enc_y = self.sec_fc(self.enc_fc, enc_x, residual_nb=self.fc_residual_nb)
        enc_pred = self.predict(enc_y, [self.fc_input_size, self.fc_input_size], self.fc_output_size)

        return self.sec_compare(enc_pred, enc_truth)


class HE_AGNEWS_Logi(HEModel):
    def __init__(self, input_nb=300):
        super(HE_AGNEWS_Logi, self).__init__()
        self.input_nb = input_nb
        self.input_shape = (-1, 300)

        self.fc_input_size = 300
        self.fc_output_size = 4
        self.fc_channel_nb = 1
        self.fc_residual_nb = max(math.ceil(self.fc_input_size / self.fc_channel_nb), self.fc_output_size) % self.fc_output_size

        self.context = None
        self.enc_fc = None

    def init_model_paras(self, context, model_paras):
        self.context = context
        self.enc_fc = self.encrypt_fc(model_paras["fc.weight"], model_paras["fc.bias"], self.fc_channel_nb)

    def clear_model_paras(self):
        self.context = None
        self.enc_fc = None

    def encrypt_input(self, x):
        x = x.reshape(self.input_shape)
        x = np.transpose(x, (1, 0))
        return self.preprocess_for_fc(x, self.fc_output_size, self.fc_channel_nb, return_bytes=True)

    def forward(self, enc_x, enc_truth):
        enc_y = self.sec_fc(self.enc_fc, enc_x, residual_nb=self.fc_residual_nb)
        mat_len = self.fc_input_size
        enc_pred = self.predict(enc_y, [mat_len, mat_len], self.fc_output_size)

        return self.sec_compare(enc_pred, enc_truth)

class HE_BANK_Logi(HEModel):
    def __init__(self, input_nb=48):
        super(HE_BANK_Logi, self).__init__()
        self.input_nb = input_nb
        self.input_shape = (-1, 48)

        self.fc_input_size = 48
        self.fc_output_size = 2
        self.fc_channel_nb = 1
        self.fc_residual_nb = max(math.ceil(self.fc_input_size / self.fc_channel_nb), self.fc_output_size) % self.fc_output_size

        self.context = None
        self.enc_fc = None

    def init_model_paras(self, context, model_paras):
        self.context = context
        self.enc_fc = self.encrypt_fc(model_paras["fc.weight"], model_paras["fc.bias"], self.fc_channel_nb)

    def clear_model_paras(self):
        self.context = None
        self.enc_fc = None

    def encrypt_input(self, x):
        x = x.reshape(self.input_shape)
        x = np.transpose(x, (1, 0))
        return self.preprocess_for_fc(x, self.fc_output_size, self.fc_channel_nb, return_bytes=True)

    def forward(self, enc_x, enc_truth):
        enc_y = self.sec_fc(self.enc_fc, enc_x, residual_nb=self.fc_residual_nb)
        mat_len = self.fc_input_size
        enc_pred = self.predict(enc_y, [mat_len, mat_len], self.fc_output_size)

        return self.sec_compare(enc_pred, enc_truth)