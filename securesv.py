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
from hybridmodel import *
from client import *
import pandas as pd
import torch.multiprocessing as mp
import random128

try:
     mp.set_start_method('spawn')
except RuntimeError:
    pass

class SecSV:
    def __init__(self, clients, hybridmodel):
        self.clients = clients
        self.hybridmodel = hybridmodel
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
        self.test_data_shares = []
        self.all_processed_shares = []
        self.remained_processed_shares = []
        self.processed_shares = None
        self.skip = True
        self.idx_set = set([i for i in range(self.test_size)])
        self.context = None
        self.context_bytes = None

    def init_time_dict(self):
        self.time_dict["total"] = 0.0
        self.time_dict["encryption"] = 0.0
        self.time_dict["decryption"] = 0.0
        self.time_dict["HE computation"] = 0.0
        self.time_dict["communication"] = 0.0
        self.time_dict["activation"] = 0.0
        self.time_dict["shares generation"] = 0.0
        self.hybridmodel.time_dict = self.time_dict

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
            self.time_dict["communication"] += communicate(context_bytes_sk) * (self.clients.size - 2) + communicate(context_bytes) * 2

        self.context = context
        self.context_bytes = context_bytes_sk

    def secretly_share_data(self):
        print("\nSecretly share test data")
        feature_share1_list = []
        feature_share2_list = []
        truth_share1_list = []
        truth_share2_list = []

        for id in self.clients.data.keys():
            test_data = self.clients.data[id].test_data()
            data_loader = torch.utils.data.DataLoader(test_data, batch_size=self.test_size, shuffle=False)

            for (data, target) in data_loader:
                feature_share1, feature_share2 = self.hybridmodel.generate_shares(data.numpy().reshape(self.input_shape))
                # print(data_shares[0].shape)
                truth_share1, truth_share2 = self.hybridmodel.generate_shares(target.numpy())

                if self.measure_time:
                    self.time_dict["communication"] += communicate(feature_share1) + communicate(feature_share2) + communicate(truth_share1) + communicate(truth_share2)

                feature_share1_list.append(feature_share1)
                feature_share2_list.append(feature_share2)
                truth_share1_list.append(truth_share1)
                truth_share2_list.append(truth_share2)

        all_feature_share1 = np.concatenate(feature_share1_list, axis=0)
        all_feature_share2 = np.concatenate(feature_share2_list, axis=0)
        all_truth_share1 = np.concatenate(truth_share1_list, axis=0)
        all_truth_share2 = np.concatenate(truth_share2_list, axis=0)
        self.test_data_shares = (all_feature_share1, all_feature_share2, all_truth_share1, all_truth_share2)

    def shares_loader(self, indices="all"):
        if indices == "all":
            indices = [i for i in range(self.test_size)]

        batch_nb = math.ceil(len(indices) / self.batch_size)
        packed_feature_share1_ls = np.array_split(self.test_data_shares[0][indices], batch_nb)
        packed_feature_share2_ls = np.array_split(self.test_data_shares[1][indices], batch_nb)
        packed_truth_share1_ls = np.array_split(self.test_data_shares[2][indices], batch_nb)
        packed_truth_share2_ls = np.array_split(self.test_data_shares[3][indices], batch_nb)

        process_shares = []
        for i in range(batch_nb):
            size = packed_feature_share1_ls[i].shape[0]
            self.hybridmodel.image_nb = size
            if self.measure_time:
                start = time.process_time()
            processed_feature_share1 = self.hybridmodel.pre_process_input(packed_feature_share1_ls[i])
            if self.measure_time:
                self.time_dict["total"] -= time.process_time() - start
            processed_feature_share2 = self.hybridmodel.pre_process_input(packed_feature_share2_ls[i])

            process_shares.append(((processed_feature_share1, processed_feature_share2), (packed_truth_share1_ls[i], packed_truth_share2_ls[i]), size))

        # packed_feature_shares_ls = zip(processed_feature_share1_ls, processed_feature_share2_ls)
        # packed_truth_shares_ls = zip(packed_truth_share1_ls, packed_truth_share2_ls)
        # print(packed_feature_shares_ls)
        # print(packed_truth_shares_ls)
        if len(indices) == self.test_size:
            self.all_processed_shares = process_shares
        else:
            self.remained_processed_shares = process_shares

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
        sec_model = self.hybridmodel
        sec_model.image_nb = self.processed_shares[0][2]
        sec_model.measure_time = self.measure_time
        sec_model.init_model_paras(context, model_paras)

        correct_ids_ls = []
        pbar = tqdm(self.processed_shares, mininterval=60)
        # pbar = self.processed_shares
        total_size = 0
        for x_shares, truth_shares, size in pbar:
            if size != sec_model.image_nb:
                sec_model.image_nb = size
                sec_model.init_model_paras(context, model_paras)

            correct_ids = sec_model(x_shares, truth_shares) + total_size
            total_size += size
            # print(len(correct_ids) / size)
            correct_ids_ls.append(correct_ids)

        correct_ids = np.concatenate(correct_ids_ls)
        sec_model.clear_model_paras()
        # print(len(correct_ids) / self.test_size)

        return set(correct_ids)

    def recover_serialized_objects(self):
        self.context = ts.Context.load(self.context_bytes)

    def sv_eval_one_rnd_rparallel(self, rnd, rnds_acc_dict):
        self.recover_serialized_objects()
        self.init_time_dict()
        self.shares_loader()
        if self.measure_time:
            start = time.process_time()

        clients = self.clients
        acc_dict = {}
        sel_clients = clients.selected_clients(rnd)
        all_subsets = make_all_subsets(list(sel_clients.keys()))

        naive_samples = self.idx_set
        hard_samples = self.idx_set

        self.processed_shares = self.all_processed_shares
        for client in tqdm(sel_clients.values()):
            subset = frozenset((client.id,))
            local_model = torch.load(client.models[rnd])
            model_paras = local_model.state_dict()
            correct_samples = self.eval(model_paras)
            acc_dict[subset] = len(correct_samples) / self.test_size
            if self.skip:
                naive_samples = naive_samples.intersection(correct_samples)
                hard_samples = hard_samples.intersection(self.idx_set - correct_samples)

        indices = sorted(list(self.idx_set))
        if self.skip:
            indices = sorted(list(self.idx_set - naive_samples - hard_samples))
            self.shares_loader(indices)
            self.processed_shares = self.remained_processed_shares

        if len(indices) > 0:
            for subset in tqdm(all_subsets):
                if len(subset) < 2:
                    continue

                model_ls = clients.get_model_list(list(subset), rnd)
                size_ls = clients.get_train_size_list(list(subset), rnd)

                aggr_model = FL.fedavg(model_ls, size_ls)
                model_paras = aggr_model.state_dict()
                correct_samples = self.eval(model_paras)

                if self.skip:
                    acc = (len(naive_samples) + len(correct_samples)) / self.test_size
                else:
                    acc = len(correct_samples) / self.test_size

                acc_dict[subset] = acc
        else:
            for subset in tqdm(all_subsets):
                acc = len(naive_samples) / self.test_size
                acc_dict[subset] = acc

        rnds_acc_dict[rnd] = acc_dict

        if self.measure_time:
            self.time_dict["total"] += time.process_time() - start

        return self.time_dict

    def sv_eval_mul_rnds_rparallel(self):
        if self.measure_time:
            start = time.process_time()

        self.init_context()
        self.secretly_share_data()
        self.shares_loader()
        self.processed_shares = self.all_processed_shares
        init_model = torch.load(self.clients.init_model)
        model_paras = init_model.state_dict()
        print("\nEvaluate the initial model")
        correct_ids = self.eval(model_paras)
        self.init_accs[0] = len(correct_ids) / self.test_size
        # self.init_accs[0] = 0.1

        # model = torch.load("model/mnist_cnn1/iid/0/rnd9/global.pkl")
        # model_paras = model.state_dict()
        # correct_ids = self.eval(model_paras)
        # print(len(correct_ids)/self.test_size)

        if self.measure_time:
            self.time_dict["total"] += time.process_time() - start
            print("\nTime dict after evaluating the init model")
            print(self.time_dict)

        self.context = None
        self.all_processed_shares = []
        self.processed_shares = []
        manager = mp.Manager()
        rnds_acc_dict = manager.dict()
        pool = mp.Pool(10)

        print("\nEvaluate each FL round in parallel")

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

    sveval = SecSV(clients, Sec_CNN1_MNIST())
    # sveval.input_shape = (-1, 784)
    # sveval.batch_size = 784
    sveval.sv_eval_mul_rnds_rparallel()
    # sveval.save_stat("cnn1_mnist_niid_sec.json")