import torch
from torchvision import datasets
import torchvision.transforms as transforms
import tenseal as ts
import math
import json
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset
import time
from shapley import *
from utils import *
import copy
import random
from collections import Counter
from utils import *
from train import FL
from hemodel import *
from client import *
import pandas as pd
import torch.multiprocessing as mp
import random128

try:
     mp.set_start_method('spawn')
except RuntimeError:
    pass

class HESV:
    def __init__(self, clients, hemodel):
        self.clients = clients
        self.hemodel = hemodel
        self.test_data = torch.utils.data.ConcatDataset([clients.data[id].test_data() for id in clients.data.keys()])
        self.batch_size = 64
        self.T = len(clients.selection_record)
        self.init_accs = {}
        self.ssv_dict = {}
        self.msv_dict = {}
        self.dirs = "data/"
        self.test_size = len(self.test_data)
        self.input_shape = (-1, 28, 28)
        self.measure_time = True
        self.time_dict = {}
        self.init_time_dict()
        self.context = None
        self.context_bytes = None
        self.encrypted_data_list = []

    def init_context(self):
        print("\nGenerate and distribute HE keys")
        PREC = 23
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=2 ** 13,
            coeff_mod_bit_sizes=[PREC + 9, PREC, PREC, PREC + 9]
        )
        context.global_scale = pow(2, PREC)
        context.generate_galois_keys()

        context_bytes_sk = context.serialize(save_secret_key=True)
        context_bytes = context.serialize(save_secret_key=False)

        if self.measure_time:
            self.time_dict["communication"] += communicate(context_bytes_sk) * (self.clients.size - 1) + communicate(
                context_bytes)

        self.context = context
        self.context_bytes = context_bytes_sk

    def init_time_dict(self):
        self.time_dict["total"] = 0.0
        self.time_dict["encryption"] = 0.0
        self.time_dict["decryption"] = 0.0
        self.time_dict["HE computation"] = 0.0
        self.time_dict["communication"] = 0.0
        self.time_dict["activation"] = 0.0
        self.hemodel.time_dict = self.time_dict

    def encrypt_data(self):
        print("\nEncrypt test data")
        self.hemodel.context = self.context

        self.encrypted_data_bytes_list = []
        for id in self.clients.data.keys():
            test_data = self.clients.data[id].test_data()
            data_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

            for (data, target) in data_loader:
                data = data.view(self.input_shape)
                self.hemodel.image_nb = data.shape[0]
                enc_features_bytes = self.hemodel.encrypt_input(data)
                enc_truth_bytes = self.hemodel.encrypt_truth(target)

                self.encrypted_data_bytes_list.append((enc_features_bytes, enc_truth_bytes, data.shape[0]))

        self.encrypted_data_bytes_list.sort(key=lambda x: x[2], reverse=True)
        self.hemodel.context = None

    def update_ssv(self, sv_dict, rnd):
        for id in self.clients.data:
            if not id in sv_dict:
                sv_dict[id] = 0.0
        self.ssv_dict[rnd] = sv_dict

    def cal_msv(self):
        for id in self.clients.data:
            msv = 0.0
            for rnd in range(self.T):
                msv += self.ssv_dict[rnd][id]

            self.msv_dict[id] = msv

    def save_stat(self, filename):
        data = {"ssv": self.ssv_dict, "msv": self.msv_dict, "time": self.time_dict}
        with open(self.dirs+filename, "w") as f:
            json.dump(data, f, indent=4)

        f.close()

    def eval(self, model_paras):
        context = self.context
        model_paras = model_paras
        encrypted_data_list = self.encrypted_data_list
        correct_nb = 0
        sec_model = self.hemodel
        sec_model.image_nb = encrypted_data_list[0][2]
        sec_model.measure_time = self.measure_time
        sec_model.init_model_paras(context, model_paras)

        pbar = tqdm(encrypted_data_list, mininterval=60)
        # pbar = encrypted_data_list
        for (enc_features, enc_truth, size) in pbar:
            if size != sec_model.image_nb:
                sec_model.image_nb = size
                sec_model.init_model_paras(context, model_paras)

            incr_correct_nb = sec_model(enc_features, enc_truth)
            # print(incr_correct_nb / size)
            correct_nb += incr_correct_nb

        sec_model.clear_model_paras()
        return correct_nb

    def recover_serialized_objects(self):
        self.context = ts.Context.load(self.context_bytes)

        if type(self.encrypted_data_bytes_list[0][0][0]) != list:
            for (enc_features_bytes, enc_truth_bytes, size) in self.encrypted_data_bytes_list:
                enc_features = []
                for enc_feature_byte in enc_features_bytes:
                    enc_features.append(ts.CKKSVector.load(self.context, enc_feature_byte))
                enc_truth = ts.CKKSVector.load(self.context, enc_truth_bytes)
                self.encrypted_data_list.append((enc_features, enc_truth, size))
        else:
            for (enc_features_bytes, enc_truth_bytes, size) in self.encrypted_data_bytes_list:
                enc_features = []
                for enc_feature_byte in enc_features_bytes:
                    enc_feature = []
                    for enc_mat_byte in enc_feature_byte:
                        enc_feature.append(ts.CKKSVector.load(self.context, enc_mat_byte))
                    enc_features.append(enc_feature)
                enc_truth = ts.CKKSVector.load(self.context, enc_truth_bytes)
                self.encrypted_data_list.append((enc_features, enc_truth, size))


    def sv_eval_one_rnd_rparallel(self, rnd, rnds_acc_dict):
        self.recover_serialized_objects()
        self.init_time_dict()

        if self.measure_time:
            start = time.process_time()

        clients = self.clients
        acc_dict = {}
        sel_clients = clients.selected_clients(rnd)
        all_subsets = make_all_subsets(list(sel_clients.keys()))


        for client in tqdm(sel_clients.values()):
            subset = frozenset((client.id,))
            local_model = torch.load(client.models[rnd])
            model_paras = local_model.state_dict()
            correct_nb = self.eval(model_paras)
            acc_dict[subset] = correct_nb / self.test_size


        for subset in tqdm(all_subsets):
            if len(subset) < 2:
                continue

            model_ls = clients.get_model_list(list(subset), rnd)
            size_ls = clients.get_train_size_list(list(subset), rnd)

            aggr_model = FL.fedavg(model_ls, size_ls)
            model_paras = aggr_model.state_dict()
            correct_nb = self.eval(model_paras)
            acc_dict[subset] = correct_nb / self.test_size

        rnds_acc_dict[rnd] = acc_dict

        if self.measure_time:
            self.time_dict["total"] += time.process_time() - start

        return self.time_dict

    def sv_eval_mul_rnds_rparallel(self):
        if self.measure_time:
            start = time.process_time()

        self.init_context()
        self.encrypt_data()
        self.recover_serialized_objects()
        init_model = torch.load(self.clients.init_model)
        model_paras = init_model.state_dict()
        print("\nEvaluate the initial model")
        correct_nb = self.eval(model_paras)
        self.init_accs[0] = correct_nb / self.test_size
        # self.init_accs[0] = 0.1

        # model = torch.load("model/mnist/iid/cnn1/rnd9/global.pkl")
        # model_paras = model.state_dict()
        # correct_nb = self.eval(model_paras)

        if self.measure_time:
            self.time_dict["total"] += time.process_time() - start
            print("\nTime dict after evaluating the init model")
            print(self.time_dict)

        print("\nEvaluate each FL round in parallel")

        self.context = None
        self.encrypted_data_list = []
        manager = mp.Manager()
        rnds_acc_dict = manager.dict()
        pool = mp.Pool(10)

        workers = []
        for rnd in range(0, self.T):
            workers.append(pool.apply_async(self.sv_eval_one_rnd_rparallel, args=(rnd, rnds_acc_dict)))

        pool.close()
        pool.join()

        if self.measure_time:
            time_dicts = [worker.get() for worker in workers]
            print("\nTime consumed by each sub-process")
            print(time_dicts)
            incr_time_df = pd.DataFrame(time_dicts).sum()
            print("\nTotal time consumed by sub-processes")
            print(dict(incr_time_df))
            self.time_dict = dict(pd.DataFrame([self.time_dict]).sum() + incr_time_df)
            print("\nTime dict after evaluating all rounds")
            print(self.time_dict)

            start = time.process_time()

        print("\nCalculate SVs")
        for rnd in rnds_acc_dict:
            acc_dict = rnds_acc_dict[rnd]
            if rnd > 0:
                self.init_accs[rnd] = rnds_acc_dict[rnd-1][frozenset(self.clients.selected_ids(rnd-1))]
            acc_dict[frozenset()] = self.init_accs[rnd]
            SV = ShapleyValue(self.clients.selection_record[rnd], acc_dict)
            self.update_ssv(SV.svs, rnd)

        self.cal_msv()

        if self.measure_time:
            self.time_dict["total"] += time.process_time() - start + self.time_dict["communication"]
            print("\nOverall time dict")
            print(self.time_dict)

        print("\nSSVs and MSVs")
        print(self.ssv_dict)
        print(self.msv_dict)



if __name__ == '__main__':
    clients = Clients()
    clients.dirs = "data/mnist_cnn1/iid/0/"
    clients.load("clients.data")

    sveval = HESV(clients, HE_CNN1_MNIST())
    # sveval.input_shape = (-1, 784)
    # sveval.batch_size = 784
    sveval.sv_eval_mul_rnds_rparallel()
    # sveval.save_stat("cnn1_mnist_iid_he.json")