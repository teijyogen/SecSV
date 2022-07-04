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
        self.measure_time = True
        self.context = None
        self.time_dict = {}

    def plaintext(self, mat):
        return ts.plain_tensor(mat.reshape(-1), dtype='float')

    def encrypt(self, vector):
        if self.measure_time:
            start = time.process_time()
        enc_vec = ts.ckks_vector(self.context, vector.reshape(-1))

        if self.measure_time:
            self.time_dict["encryption"] += time.process_time() - start
        return enc_vec

    def decrypt(self, enc_vec):
        if self.measure_time:
            start = time.process_time()
        vec = np.array(enc_vec.decrypt())
        if self.measure_time:
            self.time_dict["decryption"] += time.process_time() - start
        return vec

    def send_enc_vector(self, enc_vec, return_bytes=False, nb_receviers=1):
        enc_vec_bytes = enc_vec.serialize()
        if self.measure_time:
            self.time_dict["communication"] += communicate(enc_vec_bytes) * nb_receviers
        if return_bytes:
            return enc_vec_bytes
        enc_vec = ts.CKKSVector.load(self.context, enc_vec_bytes)
        return enc_vec

    def big_mat_dotmul(self, mat1, mat2):
        mat_nb = len(mat1)
        enc_y = mat1[0] * mat2[0]
        for i in range(1, mat_nb):
            enc_y += mat1[i] * mat2[i]

        return enc_y

    def encrypt_conv(self, conv_weight, conv_bias, kernel_len, conv_windows_nb, in_channels_nb=1):
        repeated_times = conv_windows_nb * self.input_nb
        enc_channels = []
        for weight, bias in zip(conv_weight, conv_bias):
            enc_weights_ic = []
            for ic in range(in_channels_nb):
                ic_wt = weight[ic].view(-1)
                enc_weights = []
                # print(flat_wt.shape)
                for i in range(kernel_len ** 2):
                    rep_wt = ic_wt[i].repeat(repeated_times)
                    enc_weight = self.encrypt(rep_wt.view(-1))
                    # enc_weight = self.encrypt(flat_wt[i].view(-1))
                    enc_weight = self.send_enc_vector(enc_weight)
                    enc_weights.append(enc_weight)

                enc_weights_ic.append(enc_weights)

            rep_bias = bias.view(-1).repeat(repeated_times)
            enc_bias = self.encrypt(rep_bias.view(-1))
            # enc_bias = self.encrypt(bias.view(-1))
            enc_bias = self.send_enc_vector(enc_bias)
            enc_channels.append((enc_weights_ic, enc_bias))

        return enc_channels

    def relu(self, vec):
        if self.measure_time:
            start = time.process_time()
        res = np.maximum(0, vec)
        if self.measure_time:
            self.time_dict["activation"] += time.process_time() - start
        return res

    def maxpool(self, vec):
        if self.measure_time:
            start = time.process_time()
        res = skimage.measure.block_reduce(vec, (2, 2, 1), np.max)
        if self.measure_time:
            self.time_dict["activation"] += time.process_time() - start
        return res

    def square(self, vec):
        if self.measure_time:
            start = time.process_time()
        res = vec * vec
        if self.measure_time:
            self.time_dict["activation"] += time.process_time() - start
        return res

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
        sub_dk = dk // vir_channels_nb
        padded_x = np.pad(x, pad_width)

        enc_features_vc = []
        for vc in range(vir_channels_nb):
            enc_features_ic = []
            for ic in range(in_channels_nb):
                x_ic = padded_x[ic]
                enc_features = []
                for i in range(kernel_len):
                    for j in range(kernel_len):
                        feature = np.zeros((sub_dk, dk, x_ic.shape[2]))
                        for i_prime in range(sub_dk):
                            for j_prime in range(dk):
                                i_prime_ = i_prime + sub_dk * vc
                                feature[i_prime, j_prime, :] = x_ic[stride * i_prime_ + i, stride * j_prime + j, :].reshape(1, 1, x_ic.shape[2])

                        enc_feature = self.encrypt(feature)
                        enc_feature = self.send_enc_vector(enc_feature, return_bytes=return_bypes)
                        enc_features.append(enc_feature)

                enc_features_ic.append(enc_features)
            enc_features_vc.append(enc_features_ic)

        return enc_features_vc

    def sec_conv(self, enc_conv, enc_features):
        enc_y_channel = []
        for (enc_wt, enc_bias) in enc_conv:
            enc_y_vc = []
            for vc in range(len(enc_features)):
                if self.measure_time:
                    start = time.process_time()

                enc_y = self.big_mat_dotmul(enc_wt[0], enc_features[vc][0])
                for ic in range(len(enc_wt)):
                    enc_y += self.big_mat_dotmul(enc_wt[ic], enc_features[vc][ic])

                enc_y += enc_bias

                if self.measure_time:
                    self.time_dict["HE computation"] += time.process_time() - start

                enc_y = self.send_enc_vector(enc_y)
                enc_y_vc.append(enc_y)
            enc_y_channel.append(enc_y_vc)

        return enc_y_channel

    def encrypt_fc(self, fc_weight, fc_bias, vir_channels_nb, mat_len):
        enc_channels = []
        chunk_size = int(fc_weight.shape[1] / vir_channels_nb)
        for vc in range(vir_channels_nb):
            weight = fc_weight[:, vc * chunk_size:(vc + 1) * chunk_size]
            enc_channels.append(self.enc_perm_mats(weight, weight.shape[0], left=True))

        # rep_bias = fc_bias.view(-1, 1).repeat(1, self.input_nb)
        fc_bias = fc_bias.view(-1, 1)
        subdim = fc_bias.shape[0]
        rep_bias = torch.zeros([mat_len, mat_len])
        rep_bias[:subdim, :self.input_nb] = fc_bias.repeat(1, self.input_nb)

        if mat_len ** 2 > self.n_slots:
            enc_bias = self.encrypt_and_send(rep_bias, n_rows_left=subdim, divide=True, residual=True)
        else:
            enc_bias = self.encrypt_and_send(rep_bias)
        return (enc_channels, enc_bias)

    def preprocess_for_fc(self, matrix, n_rows_left, vir_channles_nb=1, return_bytes=False):
        nrows = matrix.shape[0]
        n_subrows = nrows // vir_channles_nb

        enc_mats_vc = []
        for vc in range(vir_channles_nb):
            sub_mat = matrix[n_subrows * vc:n_subrows * (vc + 1), :]
            enc_mats = self.enc_perm_mats(sub_mat, n_rows_left, left=False, return_bytes=return_bytes)
            enc_mats_vc.append(enc_mats)

        return enc_mats_vc

    def sec_fc(self, enc_fc, enc_x, residual_nb=0):
        if self.measure_time:
            start = time.process_time()

        enc_wts, enc_bias = enc_fc
        enc_y = self.he_matmul(enc_wts[0], enc_x[0])
        for i in range(1, len(enc_wts)):
            enc_y += self.he_matmul(enc_wts[i], enc_x[i], residual_nb)

        if type(enc_bias) == list:
            enc_y += enc_bias[0]
        else:
            enc_y += enc_bias

        if self.measure_time:
            self.time_dict["HE computation"] += time.process_time() - start

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

    def send_big_enc_vector(self, enc_vec, return_bytes=False):
        enc_vec_ls = []
        for ele in enc_vec:
            new_ele = self.send_enc_vector(ele, return_bytes=return_bytes)
            enc_vec_ls.append(new_ele)
        return enc_vec_ls

    def encrypt_big_message(self, big_mat, subdim, residual=False):
        if self.measure_time:
            start = time.process_time()

        enc_vecs = []

        if residual:
            nb_mat = 1
        else:
            nb_mat = big_mat.shape[0] // subdim

        for i in range(nb_mat):
            vec = big_mat[i*subdim:(i+1)*subdim, :].reshape(-1)
            enc_vecs.append(ts.ckks_vector(self.context, vec))

        if self.measure_time:
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

            # print("Mat Number:", mat_nb)
            for i in range(1, mat_nb):
                enc_y += mat1_ls[i] * mat2_ls[i]
                # print(np.array(enc_y.decrypt()).reshape(mat_len, mat_len))

            if subdim < mat_len:
                double_times = int(math.log(mat_len / subdim, 2))

                if double_times > 0:
                    result = enc_y + enc_y.rotate_vector(mat_len * subdim * 2 ** 0)
                    for k in range(1, double_times):
                        result += result.rotate_vector(mat_len * subdim * 2 ** k)
                else:
                    result = copy.deepcopy(enc_y)

                result += enc_y.rotate_vector_inplace(2 ** double_times * mat_len * subdim)
                for j in range(2 ** double_times + 1, int(mat_len / subdim)):
                    result += enc_y.rotate_vector_inplace(mat_len * subdim)

                if nb_residual_mat:
                    for i in range(mat_nb, mat_nb+nb_residual_mat):
                        result += mat1_ls[i] * mat2_ls[i]

                return result
            else:
                return enc_y

    def predict(self, enc_y, shape, output_size):

        y = self.decrypt(enc_y)
        y = y.reshape(shape)[:output_size, :self.input_nb]
        pred = np.argmax(y, axis=0)

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
    def __init__(self, input_nb=64):
        super(HE_CNN1_MNIST, self).__init__()
        self.input_nb = input_nb
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
        self.enc_fc1 = self.encrypt_fc(model_paras["fc1.weight"], model_paras["fc1.bias"], self.conv1_channel_nb, self.conv1_windows_nb)
        self.enc_fc2 = self.encrypt_fc(model_paras["fc2.weight"], model_paras["fc2.bias"], 1, self.fc1_output_size)

    def clear_model_paras(self):
        self.context = None
        self.enc_conv1 = None
        self.enc_fc1 = None
        self.enc_fc2 = None

    def encrypt_input(self, x):
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
                enc_feature_byte = self.send_enc_vector(enc_feature, return_bytes=True)
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
                y = y.reshape(n_rows, n_cols)
                activated_y = self.square_activate(y)
                enc_x.append(self.enc_perm_mats(activated_y, mat_nb, left=False))
        else:
            y = self.decrypt(enc_y)
            mat_len = int(y.shape[0] ** 0.5)
            y = y.reshape(mat_len, mat_len)[:n_rows, :n_cols]
            activated_y = self.square_activate(y).reshape(n_rows, n_cols)
            enc_x = self.enc_perm_mats(activated_y, mat_nb, left=False)

        return enc_x

    def forward(self, enc_features, enc_truth):

        enc_y_channel = self.sec_conv(self.enc_conv1, enc_features)

        enc_x_channel = self.sec_activate(enc_y_channel, self.conv1_windows_nb, self.input_nb, self.fc1_output_size,
                                          channel=True)

        enc_y = self.sec_fc(self.enc_fc1, enc_x_channel, channel=True)

        enc_x = self.sec_activate(enc_y, self.fc1_output_size, self.input_nb, self.fc2_output_size)

        enc_y = self.sec_fc(self.enc_fc2, enc_x, residual_nb=4)

        enc_pred = self.predict(enc_y, self.fc2_output_size)

        return self.sec_compare(enc_pred, enc_truth)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class HE_CNN2_CIFAR10(HEModel):
    def __init__(self, input_nb=34):
        super(HE_CNN2_CIFAR10, self).__init__()
        self.input_nb = input_nb
        self.image_len = 32

        self.conv1_stride = 1
        self.conv1_kernel_len = 5
        self.conv1_in_channel_nb = 3
        self.conv1_out_channel_nb = 6
        self.conv1_windows_nb = 28 * 28
        self.conv1_padding = ((0, 0), (0, 0), (0, 0), (0, 0))
        self.conv1_vir_channel_nb = 7

        self.conv2_stride = 1
        self.conv2_kernel_len = 5
        self.conv2_in_channel_nb = 6
        self.conv2_out_channel_nb = 16
        self.conv2_windows_nb = 10 * 10
        self.conv2_padding = self.conv1_padding
        self.conv2_vir_channel_nb = 1

        self.fc1_input_size = 400
        self.fc1_output_size = 120
        self.fc1_channel_nb = 4

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

        self.enc_fc1 = self.encrypt_fc(model_paras["fc1.weight"], model_paras["fc1.bias"], self.fc1_channel_nb,
                                       self.fc1_output_size)
        self.enc_fc2 = self.encrypt_fc(model_paras["fc2.weight"], model_paras["fc2.bias"], self.fc2_channel_nb,
                                       self.fc2_input_size)
        self.enc_fc3 = self.encrypt_fc(model_paras["fc3.weight"], model_paras["fc3.bias"], self.fc3_channel_nb,
                                       self.fc3_input_size)

    def clear_model_paras(self):
        self.context = None
        self.enc_conv1 = None
        self.enc_conv2 = None
        self.enc_fc1 = None
        self.enc_fc2 = None
        self.enc_fc3 = None

    def encrypt_input(self, x):
        x = x.view(self.input_nb, -1).T.view(self.conv1_in_channel_nb, self.image_len, self.image_len, self.input_nb)
        return self.preprocess_for_conv(x, self.conv1_windows_nb, self.conv1_kernel_len, self.conv1_stride,
                                        self.conv1_padding, self.conv1_in_channel_nb, self.conv1_vir_channel_nb, True)

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

    def sec_relu(self, enc_y, shape, n_rows):
        y = self.decrypt(enc_y).reshape(shape)[:n_rows, :self.input_nb]
        x = self.relu(y)
        return x

    def forward(self, enc_x, enc_truth):

        enc_y_channel = self.sec_conv(self.enc_conv1, enc_x)
        x = self.sec_relu_maxpool(enc_y_channel, [int(self.conv1_windows_nb ** 0.5), int(self.conv1_windows_nb ** 0.5), self.input_nb])

        enc_x = self.preprocess_for_conv(x, self.conv2_windows_nb, self.conv2_kernel_len, self.conv2_stride,
                                         self.conv2_padding, self.conv2_in_channel_nb, self.conv2_vir_channel_nb)
        enc_y_channel = self.sec_conv(self.enc_conv2, enc_x)
        x = self.sec_relu_maxpool(enc_y_channel, [int(self.conv2_windows_nb ** 0.5), int(self.conv2_windows_nb ** 0.5), self.input_nb]).reshape(self.fc1_input_size, self.input_nb)

        enc_x = self.preprocess_for_fc(x, self.fc1_output_size, vir_channles_nb=self.fc1_channel_nb)
        enc_y = self.sec_fc(self.enc_fc1, enc_x, residual_nb=20)
        x = self.sec_relu(enc_y, [self.fc1_output_size, self.fc1_output_size], self.fc1_output_size).reshape(self.fc2_input_size, self.input_nb)

        enc_x = self.preprocess_for_fc(x, self.fc2_output_size, vir_channles_nb=self.fc2_channel_nb)
        enc_y = self.sec_fc(self.enc_fc2, enc_x, residual_nb=36)
        x = self.sec_relu(enc_y, [self.fc2_output_size, self.fc2_input_size], self.fc2_output_size).reshape(self.fc3_input_size, self.input_nb)

        enc_x = self.preprocess_for_fc(x, self.fc3_output_size, vir_channles_nb=self.fc3_channel_nb)
        enc_y = self.sec_fc(self.enc_fc3, enc_x, residual_nb=4)

        enc_pred = self.predict(enc_y, [self.fc3_output_size, self.fc3_input_size], self.fc3_output_size)

        return self.sec_compare(enc_pred, enc_truth)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

