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
        self.T = len(clients.selection_record)
        self.init_accs = {}
        self.ssv_dict = {}
        self.msv_dict = {}
        self.ssv_dict_skip = {}
        self.msv_dict_skip = {}
        self.dirs = clients.data_dirs
        self.test_size = len(self.test_data)
        self.input_shape = hybridmodel.input_shape
        self.time_dict = {}
        self.time_dict_skip = {}
        self.init_time_dict()
        self.acc_dict = {}
        self.acc_dict_skip = {}
        self.naive_dict = {}
        self.test_data_shares = []
        self.all_processed_shares = []
        self.remained_processed_shares = []
        self.processed_shares = None
        self.skip = True
        self.noskip = True
        self.idx_set = set([i for i in range(self.test_size)])
        self.context = None
        self.context_bytes = None
        self.poly_modulus_degree = 2 ** 13
        self.n_slots = self.poly_modulus_degree // 2
        self.hybridmodel.n_slots = self.n_slots
        self.batch_size = self.hybridmodel.cal_optimal_batch_size(self.test_size)
        self.input_nb = None
        self.debug = False

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
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=[PREC + 9, PREC, PREC, PREC + 9]
        )
        context.global_scale = pow(2, PREC)
        context.generate_galois_keys()

        context_bytes_sk = context.serialize(save_secret_key=True)
        context_bytes = context.serialize(save_secret_key=False)

        self.time_dict["communication"] += communicate(context_bytes_sk) * (self.clients.size - 2) + communicate(context_bytes) * 2

        self.context = context
        self.context_bytes = context_bytes_sk
        self.hybridmodel.n_slots = self.n_slots

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

    def shares_loader(self, indices=None):
        if indices is None:
            indices = np.arange(0, self.test_size)
            batch_size = self.batch_size
        else:
            batch_size = self.hybridmodel.cal_optimal_batch_size(len(indices))

        batch_nb = math.ceil(len(indices) / batch_size)
        packed_feature_share1_ls = np.array_split(self.test_data_shares[0][indices], batch_nb)
        packed_feature_share2_ls = np.array_split(self.test_data_shares[1][indices], batch_nb)
        packed_truth_share1_ls = np.array_split(self.test_data_shares[2][indices], batch_nb)
        packed_truth_share2_ls = np.array_split(self.test_data_shares[3][indices], batch_nb)
        indices_ls = np.array_split(indices, batch_nb)
        max_size = packed_feature_share1_ls[0].shape[0]
        self.hybridmodel.set_input_nb(max_size)

        if batch_size == self.batch_size:
            self.input_nb = max_size

        processed_shares = []
        for i in range(batch_nb):
            feature_share1 = packed_feature_share1_ls[i]
            feature_share2 = packed_feature_share2_ls[i]
            size = feature_share1.shape[0]
            idxs = indices_ls[i]
            # self.hybridmodel.input_nb = size
            feature_share1 = feature_share1.reshape(size, -1)
            feature_share1 = np.pad(feature_share1, ((0, max_size-size), (0, 0))).reshape(max_size, -1)
            feature_share2 = feature_share2.reshape(size, -1)
            feature_share2 = np.pad(feature_share2, ((0, max_size-size), (0, 0))).reshape(max_size, -1)


            start = time.process_time()
            processed_feature_share1 = self.hybridmodel.preprocess_input(feature_share1)
            self.time_dict["total"] -= time.process_time() - start

            processed_feature_share2 = self.hybridmodel.preprocess_input(feature_share2)

            processed_shares.append(((processed_feature_share1, processed_feature_share2), (packed_truth_share1_ls[i], packed_truth_share2_ls[i]), idxs))

        if len(indices) == self.test_size:
            self.all_processed_shares = processed_shares
        else:
            self.remained_processed_shares = processed_shares

    def update_ssv(self, sv_dict, rnd, skip=False):
        for id in self.clients.data:
            if not id in sv_dict:
                sv_dict[id] = 0.0
        if skip:
            self.ssv_dict_skip[rnd] = sv_dict
        else:
            self.ssv_dict[rnd] = sv_dict

    def calc_ssv(self, acc_dict, rnd, skip=False):
        self.init_accs[rnd+1] = acc_dict[frozenset(self.clients.selected_ids(rnd))]
        acc_dict[frozenset()] = self.init_accs[rnd]
        SV = ShapleyValue(self.clients.selection_record[rnd], acc_dict)
        SV.calculate_svs()
        self.update_ssv(SV.svs, rnd, skip=skip)

    def update_msv(self, skip=False):
        if skip:
            for id in self.clients.data:
                msv = 0.0
                for rnd in range(self.T):
                    msv += self.ssv_dict_skip[rnd][id]
                self.msv_dict_skip[id] = msv
        else:
            for id in self.clients.data:
                msv = 0.0
                for rnd in range(self.T):
                    msv += self.ssv_dict[rnd][id]
                self.msv_dict[id] = msv

    def calc_msv(self):
        if self.noskip:
            start = time.process_time()
            for rnd in range(self.T):
                self.calc_ssv(self.acc_dict[rnd], rnd, skip=False)
            self.update_msv(skip=False)
            self.time_dict["total"] += time.process_time() - start + self.time_dict["communication"]
            print("\nOverall time dict")
            print(self.time_dict)

        if self.skip:
            start = time.process_time()
            for rnd in range(self.T):
                self.calc_ssv(self.acc_dict_skip[rnd], rnd, skip=True)
            self.update_msv(skip=True)
            self.time_dict_skip["total"] += time.process_time() - start + self.time_dict_skip["communication"]
            print("\nOverall time dict (skip)")
            print(self.time_dict_skip)

    def update_dicts(self, workers):
        time_dicts = []
        time_dicts_skip = []
        for rnd in range(self.T):
            dicts = workers[rnd].get()
            time_dicts.append(dicts[0][0])
            time_dicts_skip.append(dicts[0][1])
            self.acc_dict[rnd] = dicts[1][0]
            self.acc_dict_skip[rnd] = dicts[1][1]
            self.naive_dict[rnd] = dicts[2]

        # time_dicts = [worker.get()[0][0] for worker in workers]
        # time_dicts_skip = [worker.get()[0][1] for worker in workers]
        self.time_dict_skip = self.time_dict.copy()
        if self.noskip:
            print("------------------------------------------")
            print("\nTime consumed by each sub-process")
            print(time_dicts)
            incr_time_df = pd.DataFrame(time_dicts).sum()
            print("\nTotal time consumed by sub-processes")
            print(dict(incr_time_df))
            self.time_dict = dict(pd.DataFrame([self.time_dict]).sum() + incr_time_df)
            print("\nTime dict after evaluating all rounds")
            print(self.time_dict)
            print("------------------------------------------")
        if self.skip:
            print("------------------------------------------")
            print("\nTime consumed by each sub-process (skip)")
            print(time_dicts_skip)
            incr_time_df = pd.DataFrame(time_dicts_skip).sum()
            print("\nTotal time consumed by sub-processes (skip)")
            print(dict(incr_time_df))
            self.time_dict_skip = dict(pd.DataFrame([self.time_dict_skip]).sum() + incr_time_df)
            print("\nTime dict after evaluating all rounds (skip)")
            print(self.time_dict_skip)
            print("------------------------------------------")

    def save_stat(self, filename, skip=False):
        if skip:
            data = {"test size": self.test_size, "time": self.time_dict_skip, "naive": self.naive_dict,
                    "ssv": self.ssv_dict_skip, "msv": self.msv_dict_skip}
            with open(self.dirs+filename, "w") as f:
                json.dump(data, f, indent=4)
            f.close()
        else:
            data = {"test size": self.test_size, "time": self.time_dict, "ssv": self.ssv_dict, "msv": self.msv_dict}
            with open(self.dirs+filename, "w") as f:
                json.dump(data, f, indent=4)
            f.close()

    def eval(self, model_paras):
        context = self.context
        sec_model = self.hybridmodel
        sec_model.init_model_paras(context, model_paras)

        correct_ids_ls = []
        if self.debug:
            pbar = tqdm(self.processed_shares)
        else:
            pbar = self.processed_shares

        for x_shares, truth_shares, idxs in pbar:
            sec_model.truth_nb = idxs.shape[0]
            correct_ids = idxs[sec_model(x_shares, truth_shares)]
            if self.debug:
                print(len(correct_ids) / idxs.shape[0])
            correct_ids_ls.append(correct_ids)

        correct_ids = np.concatenate(correct_ids_ls)
        sec_model.clear_model_paras()

        return set(correct_ids)

    def recover_serialized_objects(self):
        self.context = ts.Context.load(self.context_bytes)

    def eval_init_model(self):
        print("\nEvaluate the initial model")
        self.processed_shares = self.all_processed_shares
        if self.debug:
            init_model = self.clients.get_global_model(self.T-1)
        else:
            init_model = self.clients.get_init_model()
        model_paras = init_model.state_dict()
        correct_ids = self.eval(model_paras)
        self.init_accs[0] = len(correct_ids) / self.test_size

        print(len(correct_ids)/self.test_size)

    def intersection_except(self, samples_dict, except_key):
        intersection_samples = self.idx_set
        for key, samples in samples_dict.items():
            if key == except_key:
                continue
            intersection_samples = intersection_samples.intersection(samples)
        return intersection_samples

    def find_skippable(self, model_ids, correct_samples_dict):
        correct_samples_size = 0
        correct_samples = set([])
        all_subsets = make_all_subsets(list(model_ids))
        for subset in all_subsets:
            if len(subset) > 0 and subset in correct_samples_dict.keys():
                temp_correct_samples = correct_samples_dict[subset]
                sub_model_ids = model_ids - subset
                if len(sub_model_ids) > 0:
                    temp_correct_samples = temp_correct_samples.intersection(self.find_skippable(sub_model_ids, correct_samples_dict))

                if len(temp_correct_samples) > correct_samples_size:
                    correct_samples_size = len(temp_correct_samples)
                    correct_samples = temp_correct_samples

        return correct_samples

    def eval_utilities(self, subsets, correct_samples_dict, acc_dict, rnd, naive_samples_dict={}, skip=False):
        clients = self.clients
        if not skip:
            self.hybridmodel.set_input_nb(self.input_nb)
            self.processed_shares = self.all_processed_shares
        subset_ls = list(subsets)
        subset_ls.sort(key=lambda x:len(x), reverse=False)
        for subset in tqdm(subset_ls):
            naive_samples = set([])
            if skip:
                # sub_correct_samples_dict = {key: value for key, value in correct_samples_dict.items() if key in subset}
                # naive_samples = self.intersection_except(sub_correct_samples_dict, None)
                naive_samples = self.find_skippable(subset, correct_samples_dict)
                naive_samples_dict[str(list(subset))] = len(naive_samples)
                indices = np.array(list(self.idx_set - naive_samples))

                if len(indices) == 0:
                    acc = len(naive_samples) / self.test_size
                    acc_dict[subset] = acc
                    continue
                elif len(indices) == self.test_size:
                    self.hybridmodel.set_input_nb(self.input_nb)
                    self.processed_shares = self.all_processed_shares
                else:
                    self.shares_loader(indices)
                    self.processed_shares = self.remained_processed_shares

            model_ls = clients.get_model_list(list(subset), rnd)
            size_ls = clients.get_train_size_list(list(subset), rnd)

            aggr_model = FL.fedavg(model_ls, size_ls)
            model_paras = aggr_model.state_dict()
            correct_samples = self.eval(model_paras)
            correct_samples_dict[subset] = correct_samples.union(naive_samples)
            acc = (len(naive_samples) + len(correct_samples)) / self.test_size
            acc_dict[subset] = acc

        return correct_samples_dict, acc_dict, naive_samples_dict

    def sv_eval_one_rnd_rparallel(self, rnd):
        self.recover_serialized_objects()
        self.init_time_dict()
        self.shares_loader()
        time_dict = self.time_dict.copy()
        self.hybridmodel.time_dict = time_dict

        start = time.process_time()
        clients = self.clients
        acc_dict = {}
        correct_samples_dict = {}
        sel_clients = clients.selected_clients(rnd)
        all_subsets = make_all_subsets(list(sel_clients.keys()))
        local_subsets = [subset for subset in all_subsets if len(subset) == 1]
        aggr_subsets = [subset for subset in all_subsets if len(subset) > 1]

        self.eval_utilities(local_subsets, correct_samples_dict, acc_dict, rnd, skip=False)
        time_dict["total"] += time.process_time() - start

        acc_dict_skip = acc_dict.copy()
        correct_samples_dict_skip = correct_samples_dict.copy()
        time_dict_skip = time_dict.copy()
        naive_samples_dict = {}

        if self.noskip:
            self.hybridmodel.time_dict = time_dict

            start = time.process_time()
            self.eval_utilities(aggr_subsets, correct_samples_dict, acc_dict, rnd, skip=False)
            time_dict["total"] += time.process_time() - start

        if self.skip:
            self.hybridmodel.time_dict = time_dict_skip

            start = time.process_time()
            self.eval_utilities(aggr_subsets, correct_samples_dict_skip, acc_dict_skip, rnd, naive_samples_dict, skip=True)
            time_dict_skip["total"] += time.process_time() - start

        return (time_dict, time_dict_skip), (acc_dict, acc_dict_skip), naive_samples_dict

    def sv_eval_mul_rnds_rparallel(self):

        start = time.process_time()
        self.init_context()
        self.secretly_share_data()
        self.shares_loader()
        self.eval_init_model()
        self.time_dict["total"] += time.process_time() - start
        self.time_dict_skip = self.time_dict.copy()
        print("\nTime dict after evaluating the init model")
        print(self.time_dict)

        self.context = None
        self.all_processed_shares = []
        self.processed_shares = []
        manager = mp.Manager()
        pool = mp.Pool(10)

        print("\nEvaluate each FL round in parallel")

        workers = []
        for rnd in range(0, self.T):
            workers.append(pool.apply_async(self.sv_eval_one_rnd_rparallel, args=(rnd,)))

        pool.close()
        pool.join()
        self.update_dicts(workers)

        print("\nCalculate SVs")
        self.calc_msv()


class SecSVPerm(SecSV):
    def __init__(self, clients, hybridmodel, epsi, sigma):
        SecSV.__init__(self, clients, hybridmodel)
        self.epsi = epsi
        self.sigma = sigma
        self.N = len(clients.selection_record[0])
        self.esti_times = math.ceil(2 * self.N / (epsi ** 2) * math.log(2 * self.N / sigma))

    def calc_ssv(self, acc_dict, rnd, skip=False):
        self.init_accs[rnd+1] = acc_dict[frozenset(self.clients.selected_ids(rnd))]
        acc_dict[frozenset()] = self.init_accs[rnd]
        SV = ShapleyValue(self.clients.selection_record[rnd], acc_dict)
        SV.calculate_svs_perm(self.perms_dict[rnd])
        self.update_ssv(SV.svs, rnd, skip=skip)

    def permutation_sampling(self, clients_ls):
        perm_ls = []
        model_ids_ls = []
        for t in range(self.esti_times):
            perm = np.random.permutation(clients_ls)
            perm_ls.append(perm)
            for i in range(self.N):
                model_ids = frozenset(perm[:i+1])
                model_ids_ls.append(model_ids)

        model_ids_ls = list(set(model_ids_ls))
        model_ids_ls.sort(key=lambda x:len(x))
        return perm_ls, model_ids_ls

    def sv_eval_one_rnd_rparallel(self, rnd, perms_dict):
        self.recover_serialized_objects()
        self.init_time_dict()
        self.shares_loader()
        time_dict = self.time_dict.copy()
        self.hybridmodel.time_dict = time_dict

        start = time.process_time()
        clients = self.clients
        acc_dict = {}
        correct_samples_dict = {}
        sel_clients = clients.selected_clients(rnd)
        perm_ls, model_subsets = self.permutation_sampling(list(sel_clients.keys()))
        local_subsets = [subset for subset in model_subsets if len(subset) == 1]
        aggr_subsets = [subset for subset in model_subsets if len(subset) > 1]

        self.eval_utilities(local_subsets, correct_samples_dict, acc_dict, rnd, skip=False)
        time_dict["total"] += time.process_time() - start

        acc_dict_skip = acc_dict.copy()
        correct_samples_dict_skip = correct_samples_dict.copy()
        time_dict_skip = time_dict.copy()
        naive_samples_dict = {}

        if self.noskip:
            self.hybridmodel.time_dict = time_dict

            start = time.process_time()
            self.eval_utilities(aggr_subsets, correct_samples_dict, acc_dict, rnd, skip=False)
            perms_dict[rnd] = perm_ls
            time_dict["total"] += time.process_time() - start

        if self.skip:
            self.hybridmodel.time_dict = time_dict_skip

            start = time.process_time()
            self.eval_utilities(aggr_subsets, correct_samples_dict_skip, acc_dict_skip, rnd, naive_samples_dict, skip=True)
            perms_dict[rnd] = perm_ls
            time_dict_skip["total"] += time.process_time() - start

        return (time_dict, time_dict_skip), (acc_dict, acc_dict_skip), naive_samples_dict

    def sv_eval_mul_rnds_rparallel(self):
        start = time.process_time()
        self.init_context()
        self.secretly_share_data()
        self.shares_loader()
        self.eval_init_model()
        self.time_dict["total"] += time.process_time() - start
        print("\nTime dict after evaluating the init model")
        print(self.time_dict)

        self.context = None
        self.all_processed_shares = []
        self.processed_shares = []
        manager = mp.Manager()
        perms_dict = manager.dict()
        pool = mp.Pool(10)

        print("\nEvaluate each FL round in parallel")

        workers = []
        for rnd in range(0, self.T):
            workers.append(pool.apply_async(self.sv_eval_one_rnd_rparallel, args=(rnd, perms_dict)))

        pool.close()
        pool.join()
        self.perms_dict = perms_dict.copy()
        self.update_dicts(workers)

        print("\nCalculate SVs")
        self.calc_msv()


class SecSVGroupTesting(SecSV):
    def __init__(self, clients, hybridmodel, epsi, sigma):
        SecSV.__init__(self, clients, hybridmodel)
        self.epsi = epsi
        self.sigma = sigma
        N = len(clients.selection_record[0])
        Z = 2 * np.array([1 / k for k in range(1, N)]).sum().item()
        self.N = N
        self.Z = Z
        self.q = [1 / Z * (1 / k + 1 / (N - k)) for k in range(1, N)]
        self.cal_esti_times()

    def cal_esti_times(self):
        N = self.N
        q = self.q
        Z = self.Z
        epsi = self.epsi
        sigma = self.sigma
        qtot = (N - 2) / N * q[0] + np.array(
            [q[k - 1] * (1 + 2 * k * (k - N) / N / (N - 1)) for k in range(2, N)]).sum().item()

        def h(u):
            return (1 + u) * math.log(1 + u) - u

        self.esti_times = math.ceil(
            8 * math.log(N * (N - 1) / (2 * sigma)) / ((1 - qtot ** 2) * h(epsi / (Z * N ** 0.5 * (1 - qtot ** 2)))))

    def calc_ssv(self, acc_dict, rnd, skip=False):
        self.init_accs[rnd+1] = acc_dict[frozenset(self.clients.selected_ids(rnd))]
        acc_dict[frozenset()] = self.init_accs[rnd]
        SV = ShapleyValue(self.clients.selection_record[rnd], acc_dict)
        SV.calculate_svs_group_testing(self.beta_mat_dict[rnd],
                                       self.model_subsets_ls_dict[rnd],
                                       {"Z": self.Z, "epsi": self.epsi, "Utot": self.init_accs[rnd+1] - self.init_accs[rnd]})
        self.update_ssv(SV.svs, rnd, skip=skip)

    def group_testing_sampling(self, clients_ls):
        beta_mat = []
        model_ids_ls = []
        for t in range(self.esti_times):
            beta_dict = {client_id: 0 for client_id in clients_ls}
            k = random.choices([i for i in range(1, self.N)], weights=self.q)[0]

            model_ids = []
            # for j in range(k):
            seq = random.sample(clients_ls, k=k)
            model_ids = model_ids + seq
            for client_id in seq:
                beta_dict[client_id] = 1

            beta_mat.append(list(beta_dict.values()))
            model_ids_ls.append(frozenset(model_ids))

        return beta_mat, model_ids_ls

    def sv_eval_one_rnd_rparallel(self, rnd, beta_mat_dict, model_subsets_ls_dict):
        self.recover_serialized_objects()
        self.init_time_dict()
        self.shares_loader()
        time_dict = self.time_dict.copy()
        self.hybridmodel.time_dict = time_dict

        start = time.process_time()
        clients = self.clients
        acc_dict = {}
        correct_samples_dict = {}
        sel_clients = clients.selected_clients(rnd)
        beta_mat, model_subsets_ls = self.group_testing_sampling(list(sel_clients.keys()))
        model_subsets = list(set(model_subsets_ls + [frozenset(sel_clients)]))
        local_subsets = [subset for subset in model_subsets if len(subset) == 1]
        aggr_subsets = [subset for subset in model_subsets if len(subset) > 1]

        self.eval_utilities(local_subsets, correct_samples_dict, acc_dict, rnd, skip=False)
        time_dict["total"] += time.process_time() - start

        acc_dict_skip = acc_dict.copy()
        correct_samples_dict_skip = correct_samples_dict.copy()
        time_dict_skip = time_dict.copy()
        naive_samples_dict = {}

        if self.noskip:
            self.hybridmodel.time_dict = time_dict

            start = time.process_time()
            self.eval_utilities(aggr_subsets, correct_samples_dict, acc_dict, rnd, skip=False)
            beta_mat_dict[rnd] = beta_mat
            model_subsets_ls_dict[rnd] = model_subsets_ls
            time_dict["total"] += time.process_time() - start

        if self.skip:
            self.hybridmodel.time_dict = time_dict_skip

            start = time.process_time()
            self.eval_utilities(aggr_subsets, correct_samples_dict_skip, acc_dict_skip, rnd, naive_samples_dict, skip=True)
            beta_mat_dict[rnd] = beta_mat
            model_subsets_ls_dict[rnd] = model_subsets_ls
            time_dict_skip["total"] += time.process_time() - start

        return (time_dict, time_dict_skip), (acc_dict, acc_dict_skip), naive_samples_dict

    def sv_eval_mul_rnds_rparallel(self):


        start = time.process_time()
        self.init_context()
        self.secretly_share_data()
        self.shares_loader()
        self.eval_init_model()
        self.time_dict["total"] += time.process_time() - start
        print("\nTime dict after evaluating the init model")
        print(self.time_dict)

        self.context = None
        self.all_processed_shares = []
        self.processed_shares = []
        manager = mp.Manager()
        beta_mat_dict = manager.dict()
        model_subsets_ls_dict = manager.dict()
        pool = mp.Pool(10)

        print("\nEvaluate each FL round in parallel")

        workers = []
        for rnd in range(0, self.T):
            workers.append(pool.apply_async(self.sv_eval_one_rnd_rparallel, args=(rnd, beta_mat_dict, model_subsets_ls_dict)))

        pool.close()
        pool.join()
        self.beta_mat_dict = beta_mat_dict.copy()
        self.model_subsets_ls_dict = model_subsets_ls_dict.copy()
        self.update_dicts(workers)

        print("\nCalculate SVs")
        self.calc_msv()



if __name__ == '__main__':
    clients = Clients("agnews_logi/dirt0.5sr1.0/0/")
    clients.load("clients.data")

    sveval = SecSV(clients, Sec_AGNEWS_Logi())
    # sveval = SecSVPerm(clients, Sec_Logi_MNIST(), 0.25, 0.1)
    sveval.skip = True
    sveval.debug = True
    sveval.sv_eval_mul_rnds_rparallel()

    # sveval = SecSVGroupTesting(clients, Sec_Logi_MNIST(), 0.25, 0.1)
    # # sveval.skip = False
    # sveval.input_shape = (-1, 784)
    # sveval.batch_size = 392
    # sveval.sv_eval_mul_rnds_rparallel()
    # # sveval.dirs = clients.dirs
    # sveval.save_stat("secsv_grouptesting.json", skip=False)
    # sveval.save_stat("secsv_grouptesting_skip.json", skip=True)