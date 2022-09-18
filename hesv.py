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
        self.batch_size = hemodel.input_nb
        self.input_shape = hemodel.input_shape
        self.T = len(clients.selection_record)
        self.init_accs = {}
        self.ssv_dict = {}
        self.msv_dict = {}
        self.dirs = clients.data_dirs
        self.test_size = len(self.test_data)
        self.time_dict = {}
        self.init_time_dict()
        self.context = None
        self.context_bytes = None
        self.encrypted_data_list = []
        self.poly_modulus_degree = 2 ** 13
        self.n_slots = self.poly_modulus_degree // 2
        self.debug = False
        self.n_processes = hemodel.n_processes

    def init_context(self):
        print("\nGenerate and distribute HE keys")
        PREC = 23
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=[PREC + 9, PREC, PREC, PREC + 9]
        )
        context.global_scale = pow(2, PREC)
        context.generate_galois_keys()

        context_bytes_sk = context.serialize(save_secret_key=True)
        context_bytes = context.serialize(save_secret_key=False)

        self.time_dict["communication"] += communicate(context_bytes_sk) * (self.clients.size - 1) + communicate(context_bytes)

        self.context = context
        self.context_bytes = context_bytes_sk
        self.hemodel.n_slots = self.n_slots

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
            self.hemodel.input_nb = self.batch_size

            for (data, target) in data_loader:
                data = data.numpy().reshape(self.input_shape)
                size = data.shape[0]
                data = np.pad(data.reshape(size, -1), ((0, self.batch_size-size), (0, 0))).reshape(self.batch_size, -1)
                enc_features_bytes = self.hemodel.encrypt_input(data)
                enc_truth_bytes = self.hemodel.encrypt_truth(target)

                self.encrypted_data_bytes_list.append((enc_features_bytes, enc_truth_bytes, size))

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
        # encrypted_data_list = self.encrypted_data_list
        encrypted_data_list = self.encrypted_data_list
        correct_nb = 0
        sec_model = self.hemodel
        # sec_model.input_nb = encrypted_data_list[0][2]
        sec_model.init_model_paras(context, model_paras)

        if self.debug:
            pbar = tqdm(encrypted_data_list)
        else:
            pbar = encrypted_data_list

        for (enc_features, enc_truth, size) in pbar:
            sec_model.truth_nb = size
            incr_correct_nb = sec_model(enc_features, enc_truth)
            correct_nb += incr_correct_nb
            if self.debug:
                print(incr_correct_nb / size)

        sec_model.clear_model_paras()
        return correct_nb

    def eval_init_model(self):
        print("\nEvaluate the initial model")
        self.recover_serialized_objects()

        if self.debug:
            init_model = self.clients.get_global_model(self.T-1)
        else:
            init_model = self.clients.get_init_model()

        model_paras = init_model.state_dict()
        correct_nb = self.eval(model_paras)
        self.init_accs[0] = correct_nb / self.test_size
        if self.debug:
            print(self.init_accs[0])

    def recover_features(self, enc_feature_bytes):
        if type(enc_feature_bytes) == list:
            enc_features = []
            for enc_feature_byte in enc_feature_bytes:
                enc_features.append(self.recover_features(enc_feature_byte))
            return enc_features
        else:
            return ts.CKKSVector.load(self.context, enc_feature_bytes)


    def recover_serialized_objects(self):
        self.context = ts.Context.load(self.context_bytes)

        for (enc_features_bytes, enc_truth_bytes, size) in self.encrypted_data_bytes_list:
            enc_features = self.recover_features(enc_features_bytes)
            enc_truth = ts.CKKSVector.load(self.context, enc_truth_bytes)
            self.encrypted_data_list.append((enc_features, enc_truth, size))

    def sv_eval_one_rnd_rparallel(self, rnd, rnds_acc_dict):
        self.recover_serialized_objects()
        self.init_time_dict()

        start = time.process_time()
        clients = self.clients
        acc_dict = {}
        sel_clients = clients.selected_clients(rnd)
        all_subsets = make_all_subsets(list(sel_clients.keys()))


        for client in tqdm(sel_clients.values()):
            subset = frozenset((client.id,))
            local_model = client.get_model(rnd)
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
        self.time_dict["total"] += time.process_time() - start

        return self.time_dict

    def sv_eval_mul_rnds_rparallel(self):

        start = time.process_time()
        self.init_context()
        self.encrypt_data()
        self.eval_init_model()
        self.time_dict["total"] += time.process_time() - start
        print("\nTime dict after evaluating the init model")
        print(self.time_dict)

        print("\nEvaluate each FL round in parallel")

        self.context = None
        self.encrypted_data_list = []
        manager = mp.Manager()
        rnds_acc_dict = manager.dict()
        pool = mp.Pool(self.n_processes)

        workers = []
        for rnd in range(0, self.T):
            workers.append(pool.apply_async(self.sv_eval_one_rnd_rparallel, args=(rnd, rnds_acc_dict)))

        pool.close()
        pool.join()

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
            SV.calculate_svs()
            self.update_ssv(SV.svs, rnd)

        self.cal_msv()

        self.time_dict["total"] += time.process_time() - start + self.time_dict["communication"]
        print("\nOverall time dict")
        print(self.time_dict)

        print("\nSSVs and MSVs")
        print(self.ssv_dict)
        print(self.msv_dict)



if __name__ == '__main__':
    clients = Clients("mrna_rnn/dirt0.5sr0.2/0/")
    clients.load("clients.data")

    sveval = HESV(clients, HE_mRNA_RNN())
    sveval.debug = True
    sveval.sv_eval_mul_rnds_rparallel()
    # sveval.save_stat("hesv_.json")