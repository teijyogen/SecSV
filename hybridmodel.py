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
from random128 import random128

class HybridModel:
    def __init__(self, gamma1=16, gamma2=111):
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.modulus_size = 1 + self.gamma1 + self.gamma2
        self.measure_time = True
        self.image_nb = 64

    def encrypt(self, vector):
        if self.measure_time:
            start = time.process_time()
        enc_vec = ts.ckks_vector(self.context, vector)

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

    def send_enc_vector(self, enc_vec, nb_receviers=1):
        enc_vec_bytes = enc_vec.serialize()
        if self.measure_time:
            self.time_dict["communication"] += communicate(enc_vec_bytes) * nb_receviers
        enc_vec = ts.CKKSVector.load(self.context, enc_vec_bytes)

        return enc_vec

    def enc_perm_mats(self, matrix, mat_nb, left=True):
        nrows = matrix.shape[0]
        ncols = matrix.shape[1]

        if nrows == ncols:
            square_mat = matrix
            mat_nb = nrows
        else:
            mat_len = max(nrows, ncols)
            subdim = min(nrows, ncols)
            square_mat = np.zeros((mat_len, mat_len))

            if nrows < ncols:
                # mat_nb = subdim
                for i in range(int(mat_len / subdim)):
                    for j in range(subdim):
                        for k in range(mat_len):
                            square_mat[i * subdim + j, k] = matrix[j, k]
            else:
                # mat_nb = mat_len
                for i in range(int(mat_len / subdim)):
                    for j in range(subdim):
                        for k in range(mat_len):
                            square_mat[k, i * subdim + j] = matrix[k, j]
        if left:
            mat = sigma(square_mat)
            enc_mat = self.encrypt(mat.reshape(-1))
            enc_mat = self.send_enc_vector(enc_mat, nb_receviers=2)
            enc_mats = [enc_mat]

            for i in range(1, mat_nb):
                mat = phi(mat)
                enc_mat = self.encrypt(mat.reshape(-1))
                enc_mat = self.send_enc_vector(enc_mat, nb_receviers=2)
                enc_mats.append(enc_mat)
        else:
            mat = tau(square_mat)
            enc_mat = self.encrypt(mat.reshape(-1))
            enc_mat = self.send_enc_vector(enc_mat, nb_receviers=2)
            enc_mats = [enc_mat]

            for i in range(1, mat_nb):
                mat = psi(mat)
                enc_mat = self.encrypt(mat.reshape(-1))
                enc_mat = self.send_enc_vector(enc_mat, nb_receviers=2)
                enc_mats.append(enc_mat)

        return enc_mats

    def encrypt_conv(self, conv_weight, conv_bias, kernel_len, conv_windows_nb):
        enc_channels = []
        for weight, bias in zip(conv_weight, conv_bias):
            flat_wt = weight.view(-1)
            enc_weights = []
            # print(flat_wt.shape)
            for i in range(kernel_len**2):
                rep_wt = flat_wt[i].repeat(conv_windows_nb*self.image_nb)
                enc_weight = self.encrypt(rep_wt.view(-1))
                # enc_weight = self.encrypt(flat_wt[i].view(-1))
                enc_weight = self.send_enc_vector(enc_weight, nb_receviers=2)
                enc_weights.append(enc_weight)

            rep_bias = bias.view(-1).repeat(conv_windows_nb*self.image_nb)
            enc_bias = self.encrypt(rep_bias.view(-1))
            # enc_bias = self.encrypt(bias.view(-1))
            enc_bias = self.send_enc_vector(enc_bias, nb_receviers=2)
            enc_channels.append((enc_weights, enc_bias))

        return enc_channels

    def encrypt_fc(self, fc_weight, fc_bias, channel_nb):
        enc_channels = []
        chunk_size = int(fc_weight.shape[1] / channel_nb)
        for c in range(channel_nb):
            weight = fc_weight[:, c*chunk_size:(c+1)*chunk_size]
            # print(weight.shape)
            enc_channels.append(self.enc_perm_mats(weight, weight.shape[0], left=True))

        rep_bias = fc_bias.view(-1, 1).repeat(1, self.image_nb)
        enc_bias = self.encrypt(rep_bias.view(-1))
        enc_bias = self.send_enc_vector(enc_bias, nb_receviers=2)

        return (enc_channels, enc_bias)

    def perm_mats(self, matrix, mat_nb, left=True):
        nrows = matrix.shape[0]
        ncols = matrix.shape[1]

        if nrows == ncols:
            square_mat = matrix
            mat_nb = nrows
        else:
            mat_len = max(nrows, ncols)
            subdim = min(nrows, ncols)
            square_mat = np.zeros((mat_len, mat_len))

            if nrows < ncols:
                # mat_nb = subdim
                for i in range(int(mat_len / subdim)):
                    for j in range(subdim):
                        for k in range(mat_len):
                            square_mat[i * subdim + j, k] = matrix[j, k]
            else:
                # mat_nb = mat_len
                for i in range(int(mat_len / subdim)):
                    for j in range(subdim):
                        for k in range(mat_len):
                            square_mat[k, i * subdim + j] = matrix[k, j]
        if left:
            mat = sigma(square_mat)
            plain_mat = ts.plain_tensor(mat.reshape(-1), dtype='float')
            plain_mats = [plain_mat]
            # plain_mats = [mat.reshape(-1)]
            for i in range(1, mat_nb):
                mat = phi(mat)
                # plain_mats.append(mat.reshape(-1))
                plain_mat = ts.plain_tensor(mat.reshape(-1), dtype='float')
                plain_mats.append(plain_mat)
        else:
            mat = tau(square_mat)
            plain_mat = ts.plain_tensor(mat.reshape(-1), dtype='float')
            plain_mats = [plain_mat]
            # plain_mats = [mat.reshape(-1)]
            for i in range(1, mat_nb):
                mat = psi(mat)
                # plain_mats.append(mat.reshape(-1))
                plain_mat = ts.plain_tensor(mat.reshape(-1), dtype='float')
                plain_mats.append(plain_mat)

        return plain_mats

    def he_matmul(self, mat1_ls, mat2_ls):
        subdim = len(mat1_ls)
        mat_len = int(mat1_ls[0].size() ** 0.5)
        enc_y = mat1_ls[0] * mat2_ls[0]

        for i in range(1, subdim):
            enc_y += mat1_ls[i] * mat2_ls[i]

        if subdim == mat_len and mat_len == self.image_nb:
            return enc_y

        if subdim < mat_len:
            # enc_y = self.send_enc_vector(enc_y)
            # y = self.decrypt(enc_y).reshape(mat_len, mat_len)
            y = np.array(enc_y.decrypt()).reshape(mat_len, mat_len)
            true_y = copy.deepcopy(y)[0*subdim:(0+1)*subdim, :]
            for j in range(1, int(mat_len/subdim)):
                true_y += y[j*subdim:(j+1)*subdim, :]
            true_y = true_y[:, :self.image_nb]
            # for k in range(int(math.log(mat_len / subdim))):
            #     true_y = true_y + torch.roll(true_y, -subdim * 2 ** k, 0)
            # true_y = true_y[:subdim, :self.image_nb]
            # enc_y = self.encrypt(true_y.reshape(-1))
            # enc_y = self.send_enc_vector(enc_y)
            enc_y = ts.ckks_vector(self.context, true_y.reshape(-1))
        else:
            # enc_y = self.send_enc_vector(enc_y)
            # true_y = self.decrypt(enc_y).reshape(mat_len, mat_len)
            true_y = np.array(enc_y.decrypt()).reshape(mat_len, mat_len)
            true_y = true_y[:, :self.image_nb]
            # enc_y = self.encrypt(true_y.reshape(-1))
            # enc_y = self.send_enc_vector(enc_y)
            enc_y = ts.ckks_vector(self.context, true_y.reshape(-1))

        return enc_y

    def sec_conv(self, enc_conv, shares):
        enc_y_channel = []
        for (enc_wt, enc_bias) in enc_conv:

            if self.measure_time:
                start = time.process_time()
            enc_y_share1 = enc_wt[0] * shares[0][0]
            # print(len(shares[0]))
            for i in range(1, len(shares[0])):
                enc_y_share1 += enc_wt[i] * shares[0][i]
            if self.measure_time:
                self.time_dict["total"] -= time.process_time() - start

            enc_y_share1 = self.send_enc_vector(enc_y_share1)

            if self.measure_time:
                start = time.process_time()
            enc_y_share2 = enc_wt[0] * shares[1][0]
            for i in range(1, len(shares[0])):
                enc_y_share2 += enc_wt[i] * shares[1][i]
            enc_y = enc_y_share1 + enc_y_share2 + enc_bias
            if self.measure_time:
                self.time_dict["HE computation"] += time.process_time() - start

            enc_y = self.send_enc_vector(enc_y)

            enc_y_channel.append(enc_y)

        return enc_y_channel

    def sec_fc(self, enc_fc, x_shares, channel=False):
        if channel:
            enc_wts, enc_bias = enc_fc

            if self.measure_time:
                start = time.process_time()
            enc_y_share1 = self.he_matmul(enc_wts[0], x_shares[0][0])
            for i in range(1, len(enc_wts)):
                enc_y_share1 += self.he_matmul(enc_wts[i], x_shares[i][0])
            if self.measure_time:
                self.time_dict["total"] -= time.process_time() - start

            enc_y_share1 = self.send_enc_vector(enc_y_share1)

            if self.measure_time:
                start = time.process_time()
            enc_y_share2 = self.he_matmul(enc_wts[0], x_shares[0][1])
            for i in range(1, len(enc_wts)):
                enc_y_share2 += self.he_matmul(enc_wts[i], x_shares[i][1])
            enc_y = enc_y_share1 + enc_y_share2 + enc_bias
            if self.measure_time:
                self.time_dict["HE computation"] += time.process_time() - start
        else:
            enc_wt = enc_fc[0][0]
            enc_bias = enc_fc[1]

            if self.measure_time:
                start = time.process_time()
            enc_y_share1 = self.he_matmul(enc_wt, x_shares[0])
            if self.measure_time:
                self.time_dict["total"] -= time.process_time() - start

            enc_y_share1 = self.send_enc_vector(enc_y_share1)

            if self.measure_time:
                start = time.process_time()
            enc_y_share2 = self.he_matmul(enc_wt, x_shares[1])
            enc_y = enc_y_share1 + enc_y_share2 + enc_bias
            if self.measure_time:
                self.time_dict["HE computation"] += time.process_time() - start

        enc_y = self.send_enc_vector(enc_y)

        return enc_y

    def pre_process_for_conv(self, x, windows_nb, kernel_len, stride):
        dk = int(windows_nb ** 0.5)
        plain_features = []
        for i in range(kernel_len):
            for j in range(kernel_len):

                feature = np.zeros((dk, dk, x.shape[0]))
                for i_prime in range(dk):
                    for j_prime in range(dk):
                        feature[i_prime, j_prime, :] = x[:, stride*i_prime+i, stride*j_prime+j].reshape(1, 1, x.shape[0])

                plain_feature = ts.plain_tensor(feature.reshape(-1), dtype='float')
                plain_features.append(plain_feature)

        return plain_features

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

    def predict(self, enc_y, output_size):
        enc_y = self.send_enc_vector(enc_y)
        y = self.decrypt(enc_y).reshape(output_size, self.image_nb)
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

class Sec_CNN1_MNIST(HybridModel):
    def __init__(self, image_nb=64):
        super(Sec_CNN1_MNIST, self).__init__()
        self.measure_time = False

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
        self.enc_conv1 = self.encrypt_conv(model_paras["conv1.weight"], model_paras["conv1.bias"], self.conv1_kernel_len, self.conv1_windows_nb)
        self.enc_fc1 = self.encrypt_fc(model_paras["fc1.weight"], model_paras["fc1.bias"], self.conv1_channel_nb)
        self.enc_fc2 = self.encrypt_fc(model_paras["fc2.weight"], model_paras["fc2.bias"], 1)

    def clear_model_paras(self):
        self.context = None
        self.enc_conv1 = None
        self.enc_fc1 = None
        self.enc_fc2 = None

    def pre_process_input(self, x):
        return self.pre_process_for_conv(x, self.conv1_windows_nb, self.conv1_kernel_len, self.conv1_stride)

    def square_activate(self, vec):
        if self.measure_time:
            start = time.process_time()
        res = vec * vec
        if self.measure_time:
            self.time_dict["activation"] += time.process_time() - start
        return res

    def activate(self, enc_y, n_rows, n_cols, mat_nb, channel=False):
        if channel:
            plain_x = []
            for i in range(len(enc_y)):
                y = self.decrypt(enc_y[i]).reshape(n_rows, n_cols)
                y = self.remainder(y)
                squared_y = self.square_activate(y)
                share1, share2 = self.generate_shares(squared_y)
                if self.measure_time:
                    self.time_dict["communication"] += communicate(share1) + communicate(share2)
                if self.measure_time:
                    start = time.process_time()
                mats_share1 = self.perm_mats(share1, mat_nb, left=False)
                if self.measure_time:
                    self.time_dict["total"] -= time.process_time() - start
                mats_share2 = self.perm_mats(share2, mat_nb, left=False)
                plain_x.append((mats_share1, mats_share2))
        else:
            y = self.decrypt(enc_y).reshape(n_rows, n_cols)
            y = self.remainder(y)
            squared_y = self.square_activate(y)
            share1, share2 = self.generate_shares(squared_y)
            if self.measure_time:
                self.time_dict["communication"] += communicate(share1) + communicate(share2)
            if self.measure_time:
                start = time.process_time()
            mats_share1 = self.perm_mats(share1, mat_nb, left=False)
            if self.measure_time:
                self.time_dict["total"] -= time.process_time() - start
            mats_share2 = self.perm_mats(share2, mat_nb, left=False)
            plain_x = (mats_share1, mats_share2)

        return plain_x

    def forward(self, x_shares, truth_shares):

        enc_y_channel = self.sec_conv(self.enc_conv1, x_shares)

        x_shares_channel = self.activate(enc_y_channel, self.conv1_windows_nb, self.image_nb, self.fc1_output_size, channel=True)

        enc_y = self.sec_fc(self.enc_fc1, x_shares_channel, channel=True)

        x_shares = self.activate(enc_y, self.fc1_output_size, self.image_nb, self.fc2_output_size)

        enc_y = self.sec_fc(self.enc_fc2, x_shares)

        pred_shares = self.predict(enc_y, self.fc2_output_size)

        return self.sec_compare(pred_shares, truth_shares)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Sec_Logi_MNIST(HybridModel):
    def __init__(self, image_nb=5):
        super(Sec_Logi_MNIST, self).__init__()
        self.measure_time = False

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

    def pre_process_input(self, x):
        x = x.T.reshape(self.input_size, self.image_nb)
        result = self.perm_mats(x, self.fc_output_size, left=False)
        return result

    def sigmoid_predict(self, enc_y):
        y = self.decrypt(enc_y).reshape(self.fc_output_size, self.image_nb)
        y = self.remainder(y)

        if self.measure_time:
            start = time.process_time()
        y = 1 / (1 + np.exp(-y))
        if self.measure_time:
            self.time_dict["activation"] += time.process_time() - start

        pred = np.argmax(y, axis=0)

        share1, share2 = self.generate_shares(pred)

        if self.measure_time:
            self.time_dict["communication"] += communicate(share1) + communicate(share2)

        return share1.reshape(-1), share2.reshape(-1)

    def forward(self, x_shares, truth_shares):

        enc_y = self.sec_fc(self.enc_fc, x_shares)
        pred_shares = self.sigmoid_predict(enc_y)

        return self.sec_compare(pred_shares, truth_shares)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)