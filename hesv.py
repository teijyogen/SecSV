import torch
import tenseal as ts
import math
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import time
from shapley import *
from utils import *
import copy
from collections import Counter
from hemodel import *
from client import Clients
import pandas as pd
import torch.multiprocessing as mp
import random128
import os


try:
     mp.set_start_method('spawn')
except RuntimeError:
    pass


he_model_func_dict = {
    "mnist_cnn": HE_MNIST_CNN,
    "bank_logi": HE_BANK_Logi,
    "agnews_logi": HE_AGNEWS_Logi,
    "mrna_rnn": HE_mRNA_RNN,

    "mnist_dnn": HE_MNIST_DNN,
    "bank_dnn": HE_BANK_DNN,
    "agnews_dnn": HE_AGNEWS_DNN,
    "mrna_dnn": HE_mRNA_DNN
}

class HESV:
    def __init__(self, clients, hemodel, test_data_usage=1.0):
        self.clients = clients
        self.hemodel = hemodel
        self.test_data_usage = test_data_usage
        self.test_size = len(clients.return_test_data(self.test_data_usage))
        self.batch_size = hemodel.input_nb
        self.input_shape = hemodel.input_shape
        self.T = len(clients.selection_record)
        self.init_acc = 1 / self.hemodel.output_size
        self.ssv_dict = {}
        self.fsv_dict = {}
        self.dir = clients.dir
        self.time_dict = {}
        self.init_time_dict(self.time_dict)
        self.encrypted_data_list = []
        self.poly_modulus_degree = 2 ** 12
        self.hemodel.set_n_slots(self.poly_modulus_degree // 2)
        self.n_slots = self.poly_modulus_degree // 2
        self.debug = False
        self.n_processes = hemodel.n_processes
        self.across_models = True
        self.skip_init_model = True
        self.int_prec = 8
        self.frac_prec = 23

    def init_context(self):
        print("\nGenerate and distribute HE keys")
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=[self.int_prec + self.frac_prec, self.frac_prec, self.frac_prec, self.int_prec + self.frac_prec]
        )
        context.global_scale = pow(2, self.frac_prec)
        context.generate_galois_keys()

        context_bytes_sk = share_context(context, name="context_sk", save_secret_key=True)
        context_bytes = share_context(context, name="context", save_secret_key=False)

        self.time_dict["communication"] += communicate(context_bytes_sk) * (self.clients.size - 1) + communicate(context_bytes)
        self.hemodel.n_slots = self.n_slots

    def init_time_dict(self, time_dict):
        time_dict["sequential"] = 0.0
        time_dict["parallel"] = 0.0
        time_dict["encryption"] = 0.0
        time_dict["decryption"] = 0.0
        time_dict["HE computation"] = 0.0
        time_dict["communication"] = 0.0
        time_dict["activation"] = 0.0
        self.hemodel.time_dict = time_dict

        return time_dict

    def encrypt_models_and_data_parallel(self):
        print("\nEncrypt local models and test data")
        client_ids = self.clients.data.keys()
        pool = mp.Pool(len(client_ids))

        workers = []
        for cid in client_ids:
            worker = pool.apply_async(self.encrypt_models_and_data, args=(cid,))
            workers.append(worker)

        pool.close()
        pool.join()

        for worker in workers:
            worker.get()

        self.encrypted_data_info_list = list(self.encrypted_data_info_list)
        self.encrypted_data_info_list.sort(key=lambda x: x[2], reverse=True)

    def encrypt_models_and_data(self, cid):
        self.encrypt_local_models(cid)
        self.encrypt_data(cid)

    def encrypt_data(self, cid):
        time_dict = {}
        self.init_time_dict(time_dict)

        start = time.process_time()
        self.hemodel.context = get_shared_context("context")
        test_data = self.clients.data[cid].test_data(self.test_data_usage)
        data_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        encrypted_data_bytes_list = []
        for (data, target) in tqdm(data_loader):
            data = data.numpy().reshape(self.input_shape)
            size = data.shape[0]
            data = np.pad(data.reshape(size, -1), ((0, self.batch_size-size), (0, 0))).reshape(self.batch_size, -1)
            enc_features_bytes = self.hemodel.encrypt_input(data)
            enc_truth_bytes = self.hemodel.encrypt_truth(target)
            encrypted_data_bytes_list.append((enc_features_bytes, enc_truth_bytes, size))

        sub_start = time.process_time()
        batch_id = 0
        encrypted_data_info_list = []
        for enc_features_bytes, enc_truth_bytes, size in encrypted_data_bytes_list:
            shm_name_features = f"{cid}_{batch_id}_features"
            shm_name_truth = f"{cid}_{batch_id}_truth"
            shape_features = share_data(enc_features_bytes, shm_name_features)
            shape_truth = share_data(enc_truth_bytes, shm_name_truth)
            encrypted_data_info_list.append(((shm_name_features, shape_features), (shm_name_truth, shape_truth), size))
            batch_id += 1
        time_for_memory_sharing = time.process_time() - sub_start

        time_dict["parallel"] += time.process_time() - start + time_dict["communication"]
        time_dict["sequential"] = time_dict["parallel"] - time_for_memory_sharing

        self.encrypted_data_info_list += encrypted_data_info_list
        self.time_dict_data_preparation[cid] = time_dict

    def encrypt_local_models(self, cid):
        time_dict = {}
        self.init_time_dict(time_dict)

        start = time.process_time()
        self.hemodel.context = get_shared_context("context")

        client = self.clients.find_client(cid)

        encrpted_model_info_dicts = {}
        time_for_memory_sharing = .0
        for rnd in tqdm(range(self.T)):
            local_model = client.get_model(rnd)
            model_param = local_model.state_dict()
            self.hemodel.init_model_param(model_param, return_bytes=True)
            enc_param = self.hemodel.enc_param

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

    def update_ssv(self, sv_dict, rnd):
        for id in self.clients.data:
            if not id in sv_dict:
                sv_dict[id] = 0.0
        self.ssv_dict[rnd] = sv_dict

    def update_fsv(self):
        for id in self.clients.data:
            fsv = 0.0
            for rnd in range(self.T):
                fsv += self.ssv_dict[rnd][id]

            self.fsv_dict[id] = fsv

    def calc_ssv(self, rnd):
        SV = ShapleyValue(self.clients.selection_record[rnd], self.acc_dict[rnd])
        SV.calculate_svs()
        self.update_ssv(SV.svs, rnd)

    def calc_fsv(self):
        start = time.process_time()
        for rnd in range(self.T):
            acc_dict = self.acc_dict[rnd]
            if rnd == 0:
                acc_dict[frozenset()] = self.init_acc
            else:
                acc_dict[frozenset()] = self.acc_dict[rnd - 1][frozenset(self.clients.selected_ids(rnd))]
            self.acc_dict[rnd] = acc_dict
            self.calc_ssv(rnd)
        self.update_fsv()
        time_taken = time.process_time() - start
        self.time_dict["parallel"] += time_taken
        self.time_dict["sequential"] += time_taken

        print("\nSSVs and FSVs")
        print(self.ssv_dict)
        print(self.fsv_dict)

    def save_stat(self, filename, suffix=None):
        data = {"test size": self.test_size, "ssv": self.ssv_dict, "fsv": self.fsv_dict,
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
        with open(folder+filename, "w") as f:
            json.dump(data, f, indent=4)

        f.close()

    def save(self, filename, suffix=None):
        import pickle

        if suffix:
            folder = "pkl/" + self.dir + suffix + "/"
            filename = f"{filename}_{suffix}.pkl"
        else:
            folder = "pkl/" + self.dir
            filename = f"{filename}.pkl"
        make_dir(folder)
        with open(folder + filename, "wb") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    # def eval_across_data(self, model):
    #     encrypted_data_list = self.encrypted_data_list
    #     correct_nb = 0
    #
    #     if self.debug:
    #         pbar = tqdm(encrypted_data_list)
    #     else:
    #         pbar = encrypted_data_list
    #
    #     for (enc_features, enc_truth, size) in pbar:
    #         model.truth_nb = size
    #         incr_correct_nb = model(enc_features, enc_truth)
    #         correct_nb += incr_correct_nb
    #         if self.debug:
    #             print(incr_correct_nb / size)
    #
    #     return correct_nb

    def eval_across_models(self, enc_data_batch, enc_model_dict, correct_nb_dict):
        enc_features, enc_truth, size = enc_data_batch

        for key, (enc_param, _) in enc_model_dict.items():
            self.hemodel.enc_param = enc_param
            self.hemodel.truth_nb = size
            correct_nb = self.hemodel(enc_features, enc_truth)
            if self.debug:
                print("Model %s: %s" %(str(key), correct_nb / size))

            correct_nb_dict[key] += correct_nb

        return correct_nb_dict

    # def eval_init_model(self):
    #     if self.skip_init_model:
    #         return
    #
    #     print("\nEvaluate the initial model")
    #     self.recover_serialized_objects()
    #
    #     if self.debug:
    #         init_model = self.clients.get_global_model(self.T-1)
    #     else:
    #         init_model = self.clients.get_init_model()
    #
    #     model_param = init_model.state_dict()
    #     enc_model = copy.deepcopy(self.hemodel)
    #     enc_model.init_context()
    #     enc_model.init_model_param(model_param)
    #     correct_nb = self.eval_across_data(enc_model)
    #
    #     self.init_acc = correct_nb / self.test_size
    #     if self.debug:
    #         print(self.init_acc)
    #
    #     del enc_model

    def load_encrypted_data(self):
        context = self.hemodel.context
        for (features, truth, size) in self.encrypted_data_info_list:
            enc_features = get_shared_data(features[0], context, features[1])
            enc_truth = get_shared_data(truth[0], context, truth[1])
            self.encrypted_data_list.append((enc_features, enc_truth, size))

    def clear_shared_memory(self):
        unlink_shared_data("context")
        unlink_shared_data("context_sk")

        for (features, truth, size) in self.encrypted_data_info_list:
            unlink_shared_data(features[0])
            unlink_shared_data(truth[0])

        for rnd in range(self.T):
            for cid, model_info_dicts in self.encrypted_local_model_dict.items():
                model_info_dict = model_info_dicts[rnd]
                for key, (shm_name, param_shape) in model_info_dict.items():
                    unlink_shared_data(shm_name)

    def load_encrypted_local_models(self, rnd):
        model_dict = {}
        for cid, model_info_dicts in self.encrypted_local_model_dict.items():
            subset = frozenset((cid,))
            model_info_dict = model_info_dicts[rnd]
            client = self.clients.find_client(cid)
            size = client.train_size
            param_dict = {}
            for key, (shm_name, param_shape) in model_info_dict.items():
                enc_param = get_shared_data(shm_name, self.hemodel.context, param_shape)
                param_dict[key] = enc_param

            model_dict[subset] = (param_dict, size)

        return model_dict

    # TODO
    # def sv_eval_one_rnd_across_data(self, rnd):
    #     self.recover_serialized_objects()
    #
    #     time_dict = {}
    #     self.init_time_dict(time_dict)
    #
    #     start = time.process_time()
    #     clients = self.clients
    #     acc_dict = {}
    #     sel_clients = clients.selected_clients(rnd)
    #     all_subsets = make_all_subsets(list(sel_clients.keys()))
    #     model_dict = {}
    #
    #     for client in tqdm(sel_clients.values()):
    #         subset = frozenset((client.id,))
    #         local_model = client.get_model(rnd)
    #         model_param = local_model.state_dict()
    #
    #         enc_model = copy.deepcopy(self.hemodel)
    #         enc_model.time_dict = time_dict
    #         enc_model.init_context()
    #         enc_model.init_model_param(model_param)
    #         correct_nb = self.eval_across_data(enc_model)
    #         acc_dict[subset] = correct_nb / self.test_size
    #
    #         size = client.train_size
    #         model_dict[client.id] = (enc_model.enc_param, size)
    #
    #     for subset in tqdm(all_subsets):
    #         if len(subset) < 2:
    #             continue
    #
    #         enc_model = copy.deepcopy(self.hemodel)
    #         enc_model.time_dict = time_dict
    #         enc_model.init_context()
    #         param_size_pairs = [model_dict[cid] for cid in list(subset)]
    #         enc_model.aggregate(param_size_pairs)
    #
    #         correct_nb = self.eval_across_data(enc_model)
    #         acc_dict[subset] = correct_nb / self.test_size
    #
    #     time_dict["parallel"] += time.process_time() - start + time_dict["communication"]
    #     time_dict["sequential"] = time_dict["parallel"]
    #
    #     self.acc_dict[rnd] = acc_dict
    #     self.time_dict_secure_testing[rnd] = time_dict

    def sv_eval_one_rnd_across_models(self, rnd):
        time_dict = {}
        self.init_time_dict(time_dict)

        start = time.process_time()
        self.hemodel.init_context()
        time_for_load_context = time.process_time() - start
        clients = self.clients
        sel_clients = clients.selected_clients(rnd)
        all_subsets = make_all_subsets(list(sel_clients.keys()))
        model_dict = self.load_encrypted_local_models(rnd)
        correct_nb_dict = {subset: 0 for subset in all_subsets}

        for subset in all_subsets:
            if len(subset) < 2:
                continue

            param_size_pairs = [model_dict[frozenset((cid,))] for cid in list(subset)]
            self.hemodel.aggregate(param_size_pairs)
            model_dict[subset] = (self.hemodel.enc_param, None)

        context = self.hemodel.context
        time_for_load_shared_memory = .0
        for (features, truth, size) in tqdm(self.encrypted_data_info_list):
            sub_start = time.process_time()
            enc_features = get_shared_data(features[0], context, features[1])
            enc_truth = get_shared_data(truth[0], context, truth[1])
            enc_data_batch = (enc_features, enc_truth, size)
            time_for_load_shared_memory += time.process_time() - sub_start
            self.eval_across_models(enc_data_batch, model_dict, correct_nb_dict)

        acc_dict = {key: value / self.test_size for key, value in correct_nb_dict.items()}
        time_dict["parallel"] += time.process_time() - start + time_dict["communication"]
        time_dict["sequential"] = time_dict["parallel"] - time_for_load_shared_memory - time_for_load_context

        self.acc_dict[rnd] = acc_dict
        self.time_dict_secure_testing[rnd] = time_dict

    def sv_eval_one_rnd(self, rnd):
        set_random_seed((os.getpid() * int(time.time())) % 123456789)
        self.sv_eval_one_rnd_across_models(rnd)

        # if self.across_models:
        #     self.sv_eval_one_rnd_across_models(rnd)
        # else:
        #     self.sv_eval_one_rnd_across_data(rnd)

    def init_shared_dict(self):
        manager = mp.Manager()
        self.encrypted_data_info_list = manager.list()
        self.encrypted_local_model_dict = manager.dict()
        self.time_dict_data_preparation = manager.dict()
        self.time_dict_model_preparation = manager.dict()
        self.acc_dict = manager.dict()
        self.time_dict_secure_testing = manager.dict()

    def clear_shared_dicts(self):
        self.encrypted_data_info_list = self.encrypted_data_info_list.copy()
        self.encrypted_local_model_dict = self.encrypted_local_model_dict.copy()
        self.time_dict_data_preparation = self.time_dict_data_preparation.copy()
        self.time_dict_model_preparation = self.time_dict_model_preparation.copy()
        self.acc_dict = self.acc_dict.copy()
        self.time_dict_secure_testing = self.time_dict_secure_testing.copy()

    def setup(self):
        print("Set up environment")
        start = time.process_time()
        self.init_context()
        self.init_shared_dict()
        self.encrypt_models_and_data_parallel()
        self.hemodel.init_context()
        self.load_encrypted_data()
        self.encrypted_data_list = []
        self.hemodel.clear_context()
        self.time_dict["parallel"] += time.process_time() - start + self.time_dict["communication"]
        self.time_dict["sequential"] = self.time_dict["parallel"]

    def secure_testing(self):
        # self.eval_init_model()
        print("\nEvaluate each FL round in parallel")
        pool = mp.Pool(self.n_processes)

        workers = []
        for rnd in range(0, self.T):
            worker = pool.apply_async(self.sv_eval_one_rnd, args=(rnd,))
            workers.append(worker)

        pool.close()
        pool.join()

        for worker in workers:
            worker.get()

    def print_time_dicts(self):
        print("\ntime for main process")
        print(self.time_dict)
        print("\ntime for data preparation")
        print(self.time_dict_data_preparation)
        print("\ntime for model preparation")
        print(self.time_dict_model_preparation)
        print("\ntime for secure testing")
        print(self.time_dict_secure_testing)

    def sv_eval_mul_rnds_rparallel(self):
        self.setup()
        self.secure_testing()
        self.calc_fsv()
        self.print_time_dicts()
        self.clear_shared_memory()
        self.clear_shared_dicts()

