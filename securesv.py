import itertools
import torch
import tenseal as ts
import math
import json
import numpy as np
import random
from tqdm import tqdm
import time
from shapley import *
from utils import *
import copy
from hybridmodel import *
from client import *
import torch.multiprocessing as mp
from scipy.special import comb
import os

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass


sec_model_func_dict = {
    "mnist_cnn": Sec_MNIST_CNN,
    "bank_logi": Sec_BANK_Logi,
    "agnews_logi": Sec_AGNEWS_Logi,
    "mrna_rnn": Sec_mRNA_RNN,

    "mnist_dnn": Sec_MNIST_DNN,
    "bank_dnn": Sec_BANK_DNN,
    "agnews_dnn": Sec_AGNEWS_DNN,
    "mrna_dnn": Sec_mRNA_DNN
}


class SecSV:
    def __init__(self, clients, hybridmodel, test_data_usage=1.0):
        self.clients = clients
        self.cids = list(self.clients.data.keys())
        self.hybridmodel = hybridmodel
        self.test_data_usage = test_data_usage
        self.test_size = len(self.clients.return_test_data(self.test_data_usage))
        self.T = len(clients.selection_record)
        self.init_acc = 1 / self.hybridmodel.output_size
        self.ssv_dict = {}
        self.fsv_dict = {}
        self.ssv_dict_skip = {}
        self.fsv_dict_skip = {}
        self.dir = clients.dir
        self.input_shape = hybridmodel.input_shape
        self.time_dict = {}
        self.time_dict_skip = {}
        self.init_time_dict(self.time_dict)
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
        self.poly_modulus_degree = 2 ** 12
        self.n_slots = self.poly_modulus_degree // 2
        self.hybridmodel.n_slots = self.n_slots
        self.batch_size = self.hybridmodel.input_nb
        self.debug = False
        self.n_processes = hybridmodel.n_processes
        self.skip_init_model = True
        self.rate_of_skipping = 1.0
        self.int_prec = 8
        self.frac_prec = 23

    def init_time_dict(self, time_dict):
        time_dict["sequential"] = 0.0
        time_dict["parallel"] = 0.0
        time_dict["encryption"] = 0.0
        time_dict["decryption"] = 0.0
        time_dict["HE computation"] = 0.0
        time_dict["communication"] = 0.0
        time_dict["activation"] = 0.0
        time_dict["shares generation"] = 0.0
        time_dict["repeated"] = 0.0
        self.hybridmodel.time_dict = time_dict

    def init_context(self):
        print("\nGenerate and distribute HE keys")
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=[self.int_prec + self.frac_prec, self.frac_prec, self.frac_prec,
                                 self.int_prec + self.frac_prec]
        )
        context.global_scale = pow(2, self.frac_prec)
        context.generate_galois_keys()

        context_bytes_sk = share_context(context, name="context_sk", save_secret_key=True)
        context_bytes = share_context(context, name="context", save_secret_key=False)

        self.time_dict["communication"] += communicate(context_bytes_sk) * (self.clients.size - 1) + communicate(
            context_bytes) * 2

    def encrypt_models_and_share_data_parallel(self):
        print("\nEncrypt local models and secretly share test data")
        self.parallelize(self.encrypt_models_and_share_data, [(cid,) for cid in self.clients.data.keys()])

        all_feature_share1 = np.concatenate([self.feature_share1_dict[cid] for cid in self.cids], axis=0)
        all_feature_share2 = np.concatenate([self.feature_share2_dict[cid] for cid in self.cids], axis=0)
        all_truth_share1 = np.concatenate([self.truth_share1_dict[cid] for cid in self.cids], axis=0)
        all_truth_share2 = np.concatenate([self.truth_share2_dict[cid] for cid in self.cids], axis=0)
        self.test_data_shares = (all_feature_share1, all_feature_share2, all_truth_share1, all_truth_share2)
        del (self.feature_share1_dict)
        del (self.feature_share2_dict)
        del (self.truth_share1_dict)
        del (self.truth_share2_dict)

    def encrypt_models_and_share_data(self, cid):
        self.encrypt_local_models(cid)
        self.secretly_share_data(cid)

    def encrypt_local_models(self, cid):
        time_dict = {}
        self.init_time_dict(time_dict)

        start = time.process_time()
        self.hybridmodel.context = get_shared_context("context")

        client = self.clients.find_client(cid)

        encrpted_model_info_dicts = {}
        time_for_memory_sharing = .0
        for rnd in tqdm(range(self.T)):
            local_model = client.get_model(rnd)
            model_param = local_model.state_dict()
            self.hybridmodel.init_model_param(model_param, return_bytes=True)
            enc_param = self.hybridmodel.enc_param

            encrpted_model_info_dict = {}
            substart = time.process_time()
            for key, param in enc_param.items():
                shm_name = f"{cid}_{rnd}_{key}"
                param_shape = share_data(param, shm_name)
                encrpted_model_info_dict[key] = (shm_name, param_shape)
            encrpted_model_info_dicts[rnd] = encrpted_model_info_dict
            time_for_memory_sharing += time.process_time() - substart

        time_dict["parallel"] += time.process_time() - start + time_dict["communication"]
        time_dict["sequential"] = time_dict["parallel"] - time_for_memory_sharing

        self.encrypted_local_model_dict[cid] = encrpted_model_info_dicts
        self.time_dict_model_preparation[cid] = time_dict

    def secretly_share_data(self, cid):
        time_dict = {}
        self.init_time_dict(time_dict)
        start = time.process_time()

        feature_share1_list = []
        feature_share2_list = []
        truth_share1_list = []
        truth_share2_list = []

        test_data = self.clients.data[cid].test_data(self.test_data_usage)
        test_size = len(test_data)
        self.test_size_dict[cid] = test_size
        data_loader = torch.utils.data.DataLoader(test_data, batch_size=test_size, shuffle=False)

        for (data, target) in data_loader:
            feature_share1, feature_share2 = self.hybridmodel.generate_shares(data.numpy().reshape(self.input_shape))
            # print(data_shares[0].shape)
            truth_share1, truth_share2 = self.hybridmodel.generate_shares(target.numpy())

            time_dict["communication"] += communicate(feature_share1) + communicate(feature_share2) \
                                          + communicate(truth_share1) + communicate(truth_share2)

            feature_share1_list.append(feature_share1)
            feature_share2_list.append(feature_share2)
            truth_share1_list.append(truth_share1)
            truth_share2_list.append(truth_share2)

        self.feature_share1_dict[cid] = np.vstack(feature_share1_list).reshape(self.input_shape)
        self.feature_share2_dict[cid] = np.vstack(feature_share2_list).reshape(self.input_shape)
        self.truth_share1_dict[cid] = np.vstack(truth_share1_list).reshape(-1)
        self.truth_share2_dict[cid] = np.vstack(truth_share2_list).reshape(-1)

        time_dict["parallel"] += time.process_time() - start + time_dict["communication"] - time_dict["repeated"]
        time_dict["sequential"] = time_dict["parallel"]
        self.time_dict_data_preparation[cid] = time_dict

    def preprocess_test_data(self, feature_share1, feature_share2):
        start = time.process_time()
        size = feature_share1.shape[0]
        feature_share1 = feature_share1.reshape(size, -1)
        feature_share1 = np.pad(feature_share1, ((0, self.batch_size - size), (0, 0))).reshape(self.batch_size, -1)
        processed_feature_share1 = self.hybridmodel.preprocess_input(feature_share1)
        self.time_dict["repeated"] += time.process_time() - start

        feature_share2 = feature_share2.reshape(size, -1)
        feature_share2 = np.pad(feature_share2, ((0, self.batch_size - size), (0, 0))).reshape(self.batch_size, -1)
        processed_feature_share2 = self.hybridmodel.preprocess_input(feature_share2)

        return processed_feature_share1, processed_feature_share2

    def find_mixed_batches(self, mixed_indices, local):
        max_n_clients_in_batch = len(self.cids) - 1 - 1 * (not self.hybridmodel.linear) - 1 * local
        mixed_batches = []
        while len(mixed_indices) > 0:
            n_clients_in_batch = 0
            batch = np.array([], dtype=int)
            reverse = True
            remained_size = self.batch_size
            while n_clients_in_batch < max_n_clients_in_batch and remained_size > 0 and len(mixed_indices) > 0:
                mixed_indices = sorted(mixed_indices, key=len, reverse=reverse)
                indices = mixed_indices[0]
                batch = np.concatenate([batch, indices[:remained_size]], axis=0)

                if indices.shape[0] <= remained_size:
                    del (mixed_indices[0])
                else:
                    mixed_indices[0] = indices[remained_size:]

                remained_size = self.batch_size - batch.shape[0]
                reverse = not reverse
                n_clients_in_batch += 1

            if batch.shape[0] > 0:
                mixed_batches.append(batch)
        return mixed_batches

    def shares_loader(self, target_indices=None, local=True):
        if target_indices is None:
            target_indices = np.arange(0, self.test_size)

        batch_size = self.batch_size
        test_data_count = 0
        processed_shares = []
        mixed_indices = []
        indices_batches = []
        for cid in self.cids:
            test_size = self.test_size_dict[cid]
            indices = target_indices[
                np.where((target_indices >= test_data_count) & (target_indices < test_data_count + test_size))]
            full_batch_nb = len(indices) // batch_size

            if indices[full_batch_nb * batch_size:].shape[0] > 0:
                mixed_indices.append(indices[full_batch_nb * batch_size:])
            test_data_count += test_size

            if full_batch_nb > 0:
                indices_batches += np.array_split(indices[:full_batch_nb * batch_size], full_batch_nb)

        indices_batches += self.find_mixed_batches(mixed_indices, local)
        # print(np.concatenate(indices_batches, axis=0).shape)

        for indices in indices_batches:
            processed_feature_share1, processed_feature_share2 = self.preprocess_test_data(
                self.test_data_shares[0][indices], self.test_data_shares[1][indices])
            processed_shares.append(((processed_feature_share1, processed_feature_share2),
                                     (self.test_data_shares[2][indices], self.test_data_shares[3][indices]), indices))

        if len(target_indices) == self.test_size:
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

    def update_fsv(self, skip=False):
        if skip:
            for id in self.clients.data:
                fsv = 0.0
                for rnd in range(self.T):
                    fsv += self.ssv_dict_skip[rnd][id]
                self.fsv_dict_skip[id] = fsv
        else:
            for id in self.clients.data:
                fsv = 0.0
                for rnd in range(self.T):
                    fsv += self.ssv_dict[rnd][id]
                self.fsv_dict[id] = fsv

    def calc_ssv(self, rnd, skip=False):
        acc_dict = self.acc_dict_skip[rnd] if skip else self.acc_dict[rnd]
        SV = ShapleyValue(self.clients.selection_record[rnd], acc_dict)
        SV.calculate_svs()
        self.update_ssv(SV.svs, rnd, skip=skip)

    def calc_fsv(self):
        self.time_dict_skip = self.time_dict.copy()

        if self.noskip:
            start = time.process_time()
            for rnd in range(self.T):
                acc_dict = self.acc_dict[rnd]
                if rnd == 0:
                    acc_dict[frozenset()] = self.init_acc
                else:
                    acc_dict[frozenset()] = self.acc_dict[rnd - 1][frozenset(self.clients.selected_ids(rnd))]
                self.acc_dict[rnd] = acc_dict
                self.calc_ssv(rnd, skip=False)
            self.update_fsv(skip=False)
            time_taken = time.process_time() - start
            self.time_dict["parallel"] += time_taken
            self.time_dict["sequential"] += time_taken

            print("\nSSVs and FSVs")
            print(self.ssv_dict)
            print(self.fsv_dict)

        if self.skip:
            start = time.process_time()
            for rnd in range(self.T):
                acc_dict = self.acc_dict_skip[rnd]
                if rnd == 0:
                    acc_dict[frozenset()] = self.init_acc
                else:
                    acc_dict[frozenset()] = self.acc_dict_skip[rnd - 1][frozenset(self.clients.selected_ids(rnd))]
                self.acc_dict_skip[rnd] = acc_dict
                self.calc_ssv(rnd, skip=True)
            self.update_fsv(skip=True)
            time_taken = time.process_time() - start
            self.time_dict_skip["parallel"] += time_taken
            self.time_dict_skip["sequential"] += time_taken

            print("\nSSVs and FSVs (skip)")
            print(self.ssv_dict_skip)
            print(self.fsv_dict_skip)

    def save_stat(self, filename, skip=False, suffix=None):
        if skip:
            data = {"test size": self.test_size,
                    "ssv": self.ssv_dict_skip, "fsv": self.fsv_dict_skip,
                    "time for main process": self.time_dict_skip,
                    "time for data preparation": self.time_dict_data_preparation.copy(),
                    "time for model preparation": self.time_dict_model_preparation.copy(),
                    "time for secure testing": self.time_dict_secure_testing_skip.copy()
                    }
        else:
            data = {"test size": self.test_size,
                    "ssv": self.ssv_dict, "fsv": self.fsv_dict,
                    "time for main process": self.time_dict,
                    "time for data preparation": self.time_dict_data_preparation.copy(),
                    "time for model preparation": self.time_dict_model_preparation.copy(),
                    "time for secure testing": self.time_dict_secure_testing.copy()
                    }

        if suffix:
            folder = "stat/" + self.dir + suffix + "/"
            filename = f"{filename}_{suffix}.json"
        else:
            folder = "stat/" + self.dir
            filename = f"{filename}.json"

        make_dir(folder)

        with open(folder + filename, "w") as f:
            json.dump(data, f, indent=4)
        f.close()

    def save(self, filename, suffix=None):
        import pickle
        self.test_data_shares = []
        self.all_processed_shares = []
        del (self.clients)

        if suffix:
            folder = "pkl/" + self.dir + suffix + "/"
            filename = f"{filename}_{suffix}.pkl"
        else:
            folder = "pkl/" + self.dir
            filename = f"{filename}.pkl"
        make_dir(folder)
        with open(folder + filename, "wb") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def clear_shared_memory(self):
        unlink_shared_data("context")
        unlink_shared_data("context_sk")

        for rnd in range(self.T):
            for cid, model_info_dicts in self.encrypted_local_model_dict.items():
                model_info_dict = model_info_dicts[rnd]
                for key, (shm_name, param_shape) in model_info_dict.items():
                    unlink_shared_data(shm_name)

    def eval(self, model):
        correct_ids_ls = []
        if self.debug:
            pbar = tqdm(self.processed_shares)
        else:
            pbar = self.processed_shares

        for x_shares, truth_shares, idxs in pbar:
            model.truth_nb = idxs.shape[0]
            correct_ids = idxs[model(x_shares, truth_shares)]
            if self.debug:
                print(len(correct_ids) / idxs.shape[0])
            correct_ids_ls.append(correct_ids)

        correct_ids = np.concatenate(correct_ids_ls)
        return set(correct_ids)

    def eval_init_model(self):
        if self.skip_init_model:
            return

        print("\nEvaluate the initial model")
        self.shares_loader()
        self.processed_shares = self.all_processed_shares
        if self.debug:
            init_model = self.clients.get_global_model(self.T - 1)
        else:
            init_model = self.clients.get_init_model()

        model_param = init_model.state_dict()
        enc_model = copy.deepcopy(self.hybridmodel)
        enc_model.time_dict = self.time_dict
        enc_model.init_context()
        enc_model.init_model_param(model_param)

        correct_ids = self.eval(enc_model)
        self.init_acc = len(correct_ids) / self.test_size
        if self.debug:
            print(self.init_acc)

        del enc_model
        self.processed_shares = []

    def find_skippable(self, cids, correct_samples_dict):
        naive_samples = set([])
        all_subsets = make_all_subsets(list(cids))
        for subset in all_subsets:
            if len(subset) in range(1, len(cids)) and subset in correct_samples_dict.keys():
                complement_set = cids - subset
                if complement_set in correct_samples_dict.keys():
                    new_naive_samples = correct_samples_dict[subset].intersection(correct_samples_dict[complement_set])
                    naive_samples = naive_samples.union(new_naive_samples)

        return naive_samples

    def find_subsets_for_eval(self, rnd):
        sel_ids = self.clients.selected_ids(rnd)
        all_subsets = make_all_subsets(sel_ids)
        local_subsets = [subset for subset in all_subsets if len(subset) == 1]
        aggr_subsets = [subset for subset in all_subsets if len(subset) > 1]

        self.local_subsets[rnd] = local_subsets
        self.aggr_subsets[rnd] = aggr_subsets

    def load_encrypted_local_models(self, rnd):
        model_dict = {}
        for cid, model_info_dicts in self.encrypted_local_model_dict.items():
            subset = frozenset((cid,))
            model_info_dict = model_info_dicts[rnd]
            client = self.clients.find_client(cid)
            size = client.train_size
            param_dict = {}
            for key, (shm_name, param_shape) in model_info_dict.items():
                enc_param = get_shared_data(shm_name, self.hybridmodel.context, param_shape)
                param_dict[key] = enc_param

            model_dict[subset] = (param_dict, size)

        return model_dict

    def eval_local_models(self, rnd):
        subsets = self.local_subsets[rnd]
        self.processed_shares = self.all_processed_shares

        correct_samples_dict, acc_dict = {}, {}
        model_dict = self.load_encrypted_local_models(rnd)

        for subset in tqdm(subsets):
            enc_param, _ = model_dict[subset]
            self.hybridmodel.enc_param = enc_param
            self.hybridmodel.aggregated = False

            correct_samples = self.eval(self.hybridmodel)
            correct_samples_dict[subset] = correct_samples
            acc = len(correct_samples) / self.test_size
            acc_dict[subset] = acc

        return model_dict, correct_samples_dict, acc_dict

    def eval_aggr_models(self, model_dict, correct_samples_dict, rnd, skip=False):
        subsets = self.aggr_subsets[rnd]
        acc_dict, naive_samples_dict, skipped_samples_dict = {}, {}, {}
        self.processed_shares = self.all_processed_shares

        subsets.sort(key=lambda x: len(x), reverse=False)
        for subset in tqdm(subsets):
            skipped_naive_samples = set([])
            if skip:
                naive_samples = self.find_skippable(subset, correct_samples_dict)
                # naive_samples_dict[str(list(subset))] = len(naive_samples)
                skipped_naive_samples = set(np.random.choice(list(naive_samples),
                                                             int(len(naive_samples) * self.rate_of_skipping), False))
                naive_samples_dict[subset] = naive_samples
                skipped_samples_dict[subset] = skipped_naive_samples
                indices = np.array(list(self.idx_set - skipped_naive_samples))

                if len(indices) == 0:
                    correct_samples_dict[subset] = self.idx_set
                    acc_dict[subset] = 1.0
                    print("All samples skipped")
                    continue
                elif len(indices) == self.test_size:
                    self.processed_shares = self.all_processed_shares
                else:
                    self.shares_loader(indices, False)
                    self.processed_shares = self.remained_processed_shares

            param_size_pairs = [model_dict[frozenset((cid,))] for cid in list(subset)]
            self.hybridmodel.aggregate(param_size_pairs)
            correct_samples = self.eval(self.hybridmodel)
            correct_samples_dict[subset] = correct_samples.union(skipped_naive_samples)
            acc_dict[subset] = len(correct_samples_dict[subset]) / self.test_size

        return correct_samples_dict, acc_dict, naive_samples_dict, skipped_samples_dict

    def sv_eval_one_rnd(self, rnd):
        set_random_seed((os.getpid() * int(time.time())) % 123456789)
        time_dict = {}
        self.init_time_dict(time_dict)

        start = time.process_time()
        self.hybridmodel.init_context()
        self.shares_loader()
        time_for_load_context_and_data = time.process_time() - start

        self.find_subsets_for_eval(rnd)
        model_dict, correct_samples_dict, acc_dict = self.eval_local_models(rnd)
        time_dict["parallel"] += time.process_time() - start
        time_dict["sequential"] = time_dict["parallel"] - time_for_load_context_and_data

        if self.noskip:
            self.sv_eval_one_rnd_aggr_models(model_dict, acc_dict.copy(), correct_samples_dict.copy(), time_dict.copy(),
                                             rnd, skip=False)
        if self.skip:
            self.sv_eval_one_rnd_aggr_models(model_dict, acc_dict.copy(), correct_samples_dict.copy(), time_dict.copy(),
                                             rnd, skip=True)

    def sv_eval_one_rnd_aggr_models(self, model_dict, acc_dict, correct_samples_dict, time_dict, rnd, skip=False):
        self.hybridmodel.time_dict = time_dict
        start = time.process_time()
        correct_samples_dict, new_acc_dict, naive_samples_dict, skipped_samples_dict = self.eval_aggr_models(
            model_dict, correct_samples_dict, rnd, skip=skip)

        acc_dict.update(new_acc_dict)
        time_taken = time.process_time() - start + time_dict["communication"] - time_dict["repeated"]
        time_dict["parallel"] += time_taken
        time_dict["sequential"] += time_taken

        if skip:
            self.time_dict_secure_testing_skip[rnd] = time_dict
            self.acc_dict_skip[rnd] = acc_dict
            self.correct_samples_dict_skip[rnd] = correct_samples_dict

            self.naive_dict[rnd] = naive_samples_dict
            self.skipped_dict[rnd] = skipped_samples_dict
        else:
            self.time_dict_secure_testing[rnd] = time_dict
            self.acc_dict[rnd] = acc_dict
            self.correct_samples_dict[rnd] = correct_samples_dict

    def init_shared_dict(self):
        manager = mp.Manager()

        self.encrypted_local_model_dict = manager.dict()

        self.time_dict_data_preparation = manager.dict()
        self.time_dict_model_preparation = manager.dict()
        self.feature_share1_dict = manager.dict()
        self.feature_share2_dict = manager.dict()
        self.truth_share1_dict = manager.dict()
        self.truth_share2_dict = manager.dict()
        self.test_size_dict = manager.dict()

        self.time_dict_secure_testing = manager.dict()
        self.time_dict_secure_testing_skip = manager.dict()

        self.acc_dict = manager.dict()
        self.acc_dict_skip = manager.dict()
        self.naive_dict = manager.dict()
        self.skipped_dict = manager.dict()
        self.correct_samples_dict = manager.dict()
        self.correct_samples_dict_skip = manager.dict()

        self.local_subsets = manager.dict()
        self.aggr_subsets = manager.dict()

    def clear_shared_dicts(self):
        self.encrypted_local_model_dict = self.encrypted_local_model_dict.copy()

        self.time_dict_data_preparation = self.time_dict_data_preparation.copy()
        self.time_dict_model_preparation = self.time_dict_model_preparation.copy()
        self.test_size_dict = self.test_size_dict.copy()

        self.time_dict_secure_testing = self.time_dict_secure_testing.copy()
        self.time_dict_secure_testing_skip = self.time_dict_secure_testing_skip.copy()

        self.acc_dict = self.acc_dict.copy()
        self.acc_dict_skip = self.acc_dict_skip.copy()
        self.naive_dict = self.naive_dict.copy()
        self.skipped_dict = self.skipped_dict.copy()
        self.correct_samples_dict = self.correct_samples_dict.copy()
        self.correct_samples_dict_skip = self.correct_samples_dict_skip.copy()

        self.local_subsets = self.local_subsets.copy()
        self.aggr_subsets = self.aggr_subsets.copy()

    def setup(self):
        print("Set up environment")
        start = time.process_time()
        self.init_context()
        self.init_shared_dict()
        self.encrypt_models_and_share_data_parallel()
        self.hybridmodel.init_context()
        self.shares_loader()
        # self.eval_init_model()
        self.all_processed_shares = []
        self.hybridmodel.clear_context()
        self.time_dict["parallel"] += time.process_time() - start + self.time_dict["communication"] - self.time_dict[
            "repeated"]
        self.time_dict["sequential"] = self.time_dict["parallel"]

    def parallelize(self, func, args_list):
        pool = mp.Pool(self.n_processes)
        workers = []
        for args in args_list:
            worker = pool.apply_async(func, args=args)
            workers.append(worker)

        pool.close()
        pool.join()

        for worker in workers:
            worker.get()

    def secure_testing(self):
        print("\nEvaluate each FL round in parallel")
        self.parallelize(self.sv_eval_one_rnd, [(rnd,) for rnd in range(self.T)])

    def print_time_dicts(self):

        if self.noskip:
            print("\ntime for main process")
            print(self.time_dict)
            print("\ntime for data preparation")
            print(self.time_dict_data_preparation)
            print("\ntime for model preparation")
            print(self.time_dict_model_preparation)
            print("\ntime for secure testing")
            print(self.time_dict_secure_testing)

        if self.skip:
            print("\ntime for main process (skip)")
            print(self.time_dict_skip)
            print("\ntime for data preparation (skip)")
            print(self.time_dict_data_preparation)
            print("\ntime for model preparation (skip)")
            print(self.time_dict_model_preparation)
            print("\ntime for secure testing (skip)")
            print(self.time_dict_secure_testing_skip)

    def sv_eval_mul_rnds_rparallel(self):
        self.setup()
        self.secure_testing()
        self.calc_fsv()
        self.print_time_dicts()
        self.clear_shared_memory()
        self.clear_shared_dicts()


class SecSVPerm(SecSV):
    def __init__(self, clients, hybridmodel, epsi, sigma, test_data_usage=1.0):
        SecSV.__init__(self, clients, hybridmodel, test_data_usage)
        self.epsi = epsi
        self.sigma = sigma
        self.N = len(clients.selection_record[0])
        self.esti_times = math.ceil(2 * self.N / (epsi ** 2) * math.log(2 * self.N / sigma))

    def calc_ssv(self, rnd, skip=False):
        acc_dict = self.acc_dict_skip[rnd] if skip else self.acc_dict[rnd]
        SV = ShapleyValue(self.clients.selection_record[rnd], acc_dict)
        SV.calculate_svs_perm(self.perms_dict[rnd])
        self.update_ssv(SV.svs, rnd, skip=skip)

    def permutation_sampling(self, clients_ls):
        perm_ls = []
        model_set = set()
        for t in range(self.esti_times):
            perm = np.random.permutation(clients_ls)
            perm_ls.append(perm)
            for i in range(self.N):
                cids = frozenset(perm[:i + 1])
                model_set.add(cids)

        return perm_ls, model_set

    def find_subsets_for_eval(self, rnd):
        sel_ids = self.clients.selected_ids(rnd)
        perm_ls, model_set = self.permutation_sampling(sel_ids)
        self.perms_dict[rnd] = perm_ls
        local_subsets = [subset for subset in model_set if len(subset) == 1]
        aggr_subsets = [subset for subset in model_set if len(subset) > 1]

        self.local_subsets[rnd] = local_subsets
        self.aggr_subsets[rnd] = aggr_subsets

    def init_shared_dict(self):
        super().init_shared_dict()
        manager = mp.Manager()
        self.perms_dict = manager.dict()

    def clear_shared_dicts(self):
        super().clear_shared_dicts()
        self.perms_dict = self.perms_dict.copy()


class SecSVGroupTesting(SecSV):
    def __init__(self, clients, hybridmodel, epsi, sigma, test_data_usage=1.0):
        SecSV.__init__(self, clients, hybridmodel, test_data_usage)
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

    def calc_ssv(self, rnd, skip=False):
        acc_dict = self.acc_dict_skip[rnd] if skip else self.acc_dict[rnd]
        SV = ShapleyValue(self.clients.selection_record[rnd], acc_dict)
        SV.calculate_svs_group_testing(self.beta_mat_dict[rnd],
                                       self.model_ls_dict[rnd],
                                       {"Z": self.Z, "epsi": self.epsi,
                                        "Utot": self.acc_dict[rnd][frozenset(self.clients.selection_record[rnd])]
                                                - self.acc_dict[rnd][frozenset()]})
        self.update_ssv(SV.svs, rnd, skip=skip)

    def group_testing_sampling(self, clients_ls):
        beta_mat = []
        model_ls = []
        nclients = len(clients_ls)
        id_index_df = pd.DataFrame({
            "Index": np.arange(nclients),
            "ID": sorted(clients_ls)
        })

        for t in range(self.esti_times):
            beta_vec = np.zeros(nclients)
            k = random.choices([i for i in range(1, self.N)], weights=self.q)[0]
            cids = random.sample(clients_ls, k=k)
            indices = id_index_df[id_index_df["ID"].isin(cids)]["Index"].tolist()
            beta_vec[indices] = 1

            beta_mat.append(beta_vec)
            model_ls.append(frozenset(cids))

        beta_mat = np.vstack(beta_mat)

        return beta_mat, model_ls

    def find_subsets_for_eval(self, rnd):
        sel_ids = self.clients.selected_ids(rnd)
        beta_mat, model_ls = self.group_testing_sampling(sel_ids)
        self.beta_mat_dict[rnd] = beta_mat
        self.model_ls_dict[rnd] = model_ls
        model_set = set(model_ls)
        model_set.add(frozenset(sel_ids))
        local_subsets = [subset for subset in model_set if len(subset) == 1]
        aggr_subsets = [subset for subset in model_set if len(subset) > 1]

        self.local_subsets[rnd] = local_subsets
        self.aggr_subsets[rnd] = aggr_subsets

    def init_shared_dict(self):
        super().init_shared_dict()
        manager = mp.Manager()
        self.beta_mat_dict = manager.dict()
        self.model_ls_dict = manager.dict()

    def clear_shared_dicts(self):
        super().clear_shared_dicts()
        self.beta_mat_dict = self.beta_mat_dict.copy()
        self.model_ls_dict = self.model_ls_dict.copy()


class SecSVKernelSHAP(SecSV):
    def __init__(self, clients, hybridmodel, test_data_usage=1.0):
        SecSV.__init__(self, clients, hybridmodel, test_data_usage)
        self.N = len(clients.selection_record[0])
        self.nsamples = 2 * self.N + 16

    def calc_ssv(self, rnd, skip=False):
        acc_dict = self.acc_dict_skip[rnd] if skip else self.acc_dict[rnd]
        SV = ShapleyValue(self.clients.selection_record[rnd], acc_dict)
        SV.calculate_svs_kernel_shap(self.samples[rnd])
        self.update_ssv(SV.svs, rnd, skip=skip)

    def sampling_for_local_exploration(self, clients_ls):
        num_subset_sizes = np.int(np.ceil((self.N - 1) / 2.0))
        num_paired_subset_sizes = np.int(np.floor((self.N - 1) / 2.0))
        weight_vector = np.array([(self.N - 1.0) / (i * (self.N - i)) for i in range(1, num_subset_sizes + 1)])
        weight_vector[:num_paired_subset_sizes] *= 2
        weight_vector /= weight_vector.sum()

        num_full_subsets = 0
        num_samples_left = self.nsamples
        remaining_weight_vector = copy.copy(weight_vector)

        samples = {}
        model_set = set()
        for subset_size in range(1, num_subset_sizes + 1):
            nsubsets = int(comb(self.N, subset_size))
            if subset_size <= num_paired_subset_sizes: nsubsets *= 2

            if num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets >= 1.0 - 1e-8:
                num_full_subsets += 1
                num_samples_left -= nsubsets

                if remaining_weight_vector[subset_size - 1] < 1.0:
                    remaining_weight_vector /= (1 - remaining_weight_vector[subset_size - 1])

                w = weight_vector[subset_size - 1] / comb(self.N, subset_size)
                if subset_size <= num_paired_subset_sizes: w /= 2.0
                for cids in itertools.combinations(clients_ls, subset_size):
                    subset = frozenset(cids)
                    model_set.add(subset)
                    samples[subset] = w
                    if subset_size <= num_paired_subset_sizes:
                        complement_set = frozenset(set(clients_ls) - set(subset))
                        model_set.add(complement_set)
                        samples[complement_set] = w

        new_samples = {}
        if num_full_subsets != num_subset_sizes:
            remaining_weight_vector = copy.copy(weight_vector)
            remaining_weight_vector[:num_paired_subset_sizes] /= 2
            remaining_weight_vector = remaining_weight_vector[num_full_subsets:]
            remaining_weight_vector /= np.sum(remaining_weight_vector)

            ind_set = np.random.choice(len(remaining_weight_vector), 4 * num_samples_left, p=remaining_weight_vector)
            ind_set_pos = 0
            while num_samples_left > 0 and ind_set_pos < len(ind_set):
                ind = ind_set[ind_set_pos]
                ind_set_pos += 1
                subset_size = ind + num_full_subsets + 1
                subset = frozenset(np.random.permutation(clients_ls)[:subset_size].tolist())

                if subset not in new_samples.keys():
                    num_samples_left -= 1
                    new_samples[subset] = 1.0
                    model_set.add(subset)
                else:
                    new_samples[subset] += 1.0

                if num_samples_left > 0 and subset_size <= num_paired_subset_sizes:
                    complement_set = frozenset(set(clients_ls) - set(subset))
                    if complement_set not in samples.keys():
                        num_samples_left -= 1
                        new_samples[complement_set] = 1.0
                        model_set.add(complement_set)
                    else:
                        new_samples[complement_set] += 1.0

        weight_for_new_samples = np.array(list(new_samples.values())).sum().item()
        for key in new_samples.keys():
            new_samples[key] /= weight_for_new_samples

        samples.update(new_samples)
        model_set.add(frozenset(clients_ls))

        return samples, model_set

    def find_subsets_for_eval(self, rnd):
        sel_ids = self.clients.selected_ids(rnd)
        samples, model_set = self.sampling_for_local_exploration(sel_ids)
        self.samples[rnd] = samples
        local_subsets = [subset for subset in model_set if len(subset) == 1]
        aggr_subsets = [subset for subset in model_set if len(subset) > 1]

        self.local_subsets[rnd] = local_subsets
        self.aggr_subsets[rnd] = aggr_subsets

    def init_shared_dict(self):
        super().init_shared_dict()
        manager = mp.Manager()
        self.samples = manager.dict()

    def clear_shared_dicts(self):
        super().clear_shared_dicts()
        self.samples = self.samples.copy()


class SecSV_Skip_Effect(SecSV):
    def __init__(self, clients, hybridmodel, rates_of_skipping=(0., 0.25, 0.5, 0.75, 1.0), test_data_usage=1.0):
        SecSV.__init__(self, clients, hybridmodel, test_data_usage)
        self.rates_of_skipping = rates_of_skipping
        self.ssv_dicts = {rate: {} for rate in self.rates_of_skipping}
        self.fsv_dicts = {rate: {} for rate in self.rates_of_skipping}

    def setup(self):
        print("Set up environment")
        self.init_context()
        self.init_shared_dict()
        self.secretly_share_data_parallel()

    def init_shared_dict(self):
        manager = mp.Manager()

        self.time_dict_data_preparation = manager.dict()
        self.feature_share1_dict = manager.dict()
        self.feature_share2_dict = manager.dict()
        self.truth_share1_dict = manager.dict()
        self.truth_share2_dict = manager.dict()
        self.test_size_dict = manager.dict()

        self.acc_dicts = manager.dict()
        self.naive_dicts = manager.dict()
        self.skipped_dicts = manager.dict()
        self.correct_samples_dicts = manager.dict()

        self.local_subsets = manager.dict()
        self.aggr_subsets = manager.dict()

    def clear_shared_dicts(self):
        self.time_dict_data_preparation = self.time_dict_data_preparation.copy()
        self.test_size_dict = self.test_size_dict.copy()
        self.acc_dicts = self.acc_dicts.copy()
        self.naive_dicts = self.naive_dicts.copy()
        self.skipped_dicts = self.skipped_dicts.copy()
        self.correct_samples_dicts = self.correct_samples_dicts.copy()

        self.local_subsets = self.local_subsets.copy()
        self.aggr_subsets = self.aggr_subsets.copy()

    def sv_eval_one_rnd_aggr_models(self, model_dict, acc_dict, correct_samples_dict, time_dict, rnd, skip=True):
        start = time.process_time()
        correct_samples_dict, new_acc_dict, naive_samples_dict, skipped_samples_dict = self.eval_aggr_models(
            model_dict, correct_samples_dict, rnd, time_dict, skip=True)

        acc_dict.update(new_acc_dict)
        time_taken = time.process_time() - start + time_dict["communication"] - time_dict["repeated"]
        time_dict["parallel"] += time_taken
        time_dict["sequential"] += time_taken

        return time_dict, acc_dict, correct_samples_dict, naive_samples_dict, skipped_samples_dict

    def sv_eval_one_rnd(self, rnd):
        set_random_seed((os.getpid() * int(time.time())) % 123456789)
        time_dict = {}
        self.init_time_dict(time_dict)

        start = time.process_time()
        self.shares_loader()
        time_for_shares_loading = time.process_time() - start
        self.find_subsets_for_eval(rnd)
        model_dict, correct_samples_dict, acc_dict = self.eval_local_models(rnd, time_dict)
        time_dict["parallel"] += time.process_time() - start
        time_dict["sequential"] = time_dict["parallel"] - time_for_shares_loading

        acc_dicts, correct_dicts, naive_dicts, skipped_dicts = {}, {}, {}, {}
        for rate_of_skipping in self.rates_of_skipping:
            self.rate_of_skipping = rate_of_skipping
            _, acc_dict_skip, correct_dict_skip, naive_dict, skipped_dict = \
                self.sv_eval_one_rnd_aggr_models(model_dict.copy(), acc_dict.copy(), correct_samples_dict.copy(),
                                                 time_dict.copy(),
                                                 rnd)
            acc_dicts[rate_of_skipping] = acc_dict_skip
            correct_dicts[rate_of_skipping] = correct_dict_skip
            naive_dicts[rate_of_skipping] = naive_dict
            skipped_dicts[rate_of_skipping] = skipped_dict

        self.acc_dicts[rnd] = acc_dicts
        self.correct_samples_dicts[rnd] = correct_dicts
        self.naive_dicts[rnd] = naive_dicts
        self.skipped_dicts[rnd] = skipped_dicts

    def calc_fsv(self):
        for rate_of_skip in self.rates_of_skipping:
            for rnd in range(self.T):
                acc_dict = self.acc_dicts[rnd][rate_of_skip]
                if rnd == 0:
                    acc_dict[frozenset()] = self.init_acc
                else:
                    acc_dict[frozenset()] = self.acc_dicts[rnd - 1][rate_of_skip][
                        frozenset(self.clients.selected_ids(rnd))]

                SV = ShapleyValue(self.clients.selection_record[rnd], acc_dict)
                SV.calculate_svs()
                sv_dict = SV.svs
                for id in self.clients.data:
                    if not id in sv_dict:
                        sv_dict[id] = 0.0
                self.ssv_dicts[rate_of_skip].update({rnd: sv_dict})

            fsv_dict = {}
            for id in self.clients.data:
                fsv = 0.0
                for rnd in range(self.T):
                    fsv += self.ssv_dicts[rate_of_skip][rnd][id]
                fsv_dict[id] = fsv
            self.fsv_dicts[rate_of_skip] = fsv_dict

    def save_stat(self, filename, skip=None, suffix=None):
        data = {
            "test size": self.test_size,
            "ssv": self.ssv_dicts, "fsv": self.fsv_dicts,
        }

        if suffix:
            folder = "stat/" + self.dir + suffix + "/"
            filename = f"{filename}_{suffix}.json"
        else:
            folder = "stat/" + self.dir
            filename = f"{filename}.json"
        make_dir(folder)
        with open(folder + filename, "w") as f:
            json.dump(data, f, indent=4)
        f.close()

    def sv_eval_mul_rnds_rparallel(self):
        self.setup()
        self.secure_testing()
        self.calc_fsv()
        self.clear_shared_memory()
        self.clear_shared_dicts()


