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
from collections import Counter
from utils import *
from ssmodel import *
from client import *
import torch.multiprocessing as mp
import os

try:
     mp.set_start_method('spawn')
except RuntimeError:
    pass


ss_model_func_dict = {
    "mnist_cnn": SS_MNIST_CNN,
    "bank_logi": SS_BANK_Logi,
    "agnews_logi": SS_AGNEWS_Logi,
    "mrna_rnn": SS_mRNA_RNN
}

class SecretSV:
    def __init__(self, clients, ssmodel, test_data_usage=1.0):
        self.clients = clients
        self.ssmodel = ssmodel
        self.test_data_usage = test_data_usage
        self.test_size = len(self.clients.return_test_data(self.test_data_usage))
        self.cids = list(self.clients.data.keys())
        self.T = len(clients.selection_record)
        self.init_acc = 1 / self.ssmodel.output_size
        self.ssv_dict = {}
        self.fsv_dict = {}
        self.dir = clients.dir
        self.input_shape = ssmodel.input_shape
        self.time_dict = {}
        self.init_time_dict(self.time_dict)
        self.acc_dict = {}
        self.test_data_shares = []
        self.debug = False
        self.n_processes = ssmodel.n_processes
        self.skip_init_model = True

    def init_time_dict(self, time_dict):
        time_dict["sequential"] = 0.0
        time_dict["parallel"] = 0.0
        time_dict["encryption"] = 0.0
        time_dict["decryption"] = 0.0
        time_dict["Sec computation"] = 0.0
        time_dict["communication"] = 0.0
        time_dict["activation"] = 0.0
        time_dict["shares generation"] = 0.0
        time_dict["repeated"] = 0.0
        self.ssmodel.time_dict = time_dict

    def secretly_share_models_and_data_parallel(self):
        print("\nSecretly share local models and test data")
        self.parallelize(self.secretly_share_models_and_data, [(cid,) for cid in self.cids])
        for cid in self.clients.data.keys():
            self.test_data_shares.extend(self.data_shares_dict[cid])
        del self.data_shares_dict

    def secretly_share_models_and_data(self, cid):
        set_random_seed((os.getpid() * int(time.time())) % 123456789)
        self.secretly_share_models(cid)
        self.secretly_share_data(cid)

    def secretly_share_models(self, cid):
        time_dict = {}
        self.init_time_dict(time_dict)

        start = time.process_time()
        client = self.clients.find_client(cid)
        shared_local_models = {}

        for rnd in range(self.T):
            local_model = client.get_model(rnd)
            model_param = local_model.state_dict()

            # ssmodel = copy.deepcopy(self.ssmodel)
            # ssmodel.time_dict = time_dict
            # ssmodel.init_model_param(model_param)
            # shared_local_models[rnd] = ssmodel.model_shares

            self.ssmodel.init_model_param(model_param)
            shared_local_models[rnd] = self.ssmodel.model_shares
            self.ssmodel.model_shares = {}

        time_dict["parallel"] += time.process_time() - start + time_dict["communication"] - time_dict["repeated"]
        time_dict["sequential"] = time_dict["parallel"]

        self.shared_local_models_dict[cid] = shared_local_models
        self.time_dict_model_preparation[cid] = time_dict


    def secretly_share_data(self, cid):
        time_dict = {}
        self.init_time_dict(time_dict)
        start = time.process_time()

        test_data = self.clients.data[cid].test_data(self.test_data_usage)
        data_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=False)

        data_shares = []
        for (data, truth) in data_loader:
            feature_shares = self.ssmodel.preprocess_input(data.numpy())
            truth_shares = self.ssmodel.preprocess_truth(truth.numpy())
            data_shares.append((feature_shares, truth_shares))

        time_dict["parallel"] += time.process_time() - start + time_dict["communication"] - time_dict["repeated"]
        time_dict["sequential"] = time_dict["parallel"]

        self.data_shares_dict[cid] = data_shares
        self.time_dict_data_preparation[cid] = time_dict

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

    def eval(self, model):
        if self.debug:
            pbar = tqdm(self.test_data_shares)
        else:
            pbar = self.test_data_shares

        total_correct_nb = 0
        for x_shares, truth_shares in pbar:
            correct_nb = model(x_shares, truth_shares)
            if self.debug:
                print(correct_nb / truth_shares[0].shape[0])
            total_correct_nb += correct_nb

        return total_correct_nb

    def eval_init_model(self):
        if self.skip_init_model:
            return

        print("\nEvaluate the initial model")
        if self.debug:
            init_model = self.clients.get_global_model(self.T-1)
        else:
            init_model = self.clients.get_init_model()

        model_param = init_model.state_dict()
        self.ssmodel.init_model_param(model_param)

        correct_nb = self.eval(self.ssmodel)
        self.init_acc = correct_nb / self.test_size

    def sv_eval_one_rnd(self, rnd):
        set_random_seed((os.getpid() * int(time.time())) % 123456789)
        time_dict = {}
        self.init_time_dict(time_dict)

        start = time.process_time()
        clients = self.clients
        acc_dict = {}
        sel_clients = clients.selected_clients(rnd)
        all_subsets = make_all_subsets(list(sel_clients.keys()))
        aggr_subsets = [subset for subset in all_subsets if len(subset) > 1]
        model_dict = {}

        for client in tqdm(sel_clients.values()):
            subset = frozenset((client.id,))
            model_shares = self.shared_local_models_dict[client.id][rnd]

            # ssmodel = copy.deepcopy(self.ssmodel)
            # ssmodel.time_dict = time_dict
            # ssmodel.model_shares = model_shares
            # correct_nb = self.eval(ssmodel)

            self.ssmodel.model_shares = model_shares
            correct_nb = self.eval(self.ssmodel)
            acc_dict[subset] = correct_nb / self.test_size
            model_dict[client.id] = (model_shares, client.train_size)
            self.ssmodel.model_shares = {}

        for subset in tqdm(aggr_subsets):
            param_size_pairs = [model_dict[cid] for cid in list(subset)]
            # ssmodel = copy.deepcopy(self.ssmodel)
            # ssmodel.time_dict = time_dict
            # ssmodel.aggregate(param_size_pairs)
            # correct_nb = self.eval(ssmodel)
            self.ssmodel.aggregate(param_size_pairs)
            correct_nb = self.eval(self.ssmodel)
            acc_dict[subset] = correct_nb / self.test_size
            self.ssmodel.model_shares = {}

        time_dict["parallel"] += time.process_time() - start + time_dict["communication"] - time_dict["repeated"]
        time_dict["sequential"] = time_dict["parallel"]

        self.acc_dict[rnd] = acc_dict
        self.time_dict_secure_testing[rnd] = time_dict

    def init_shared_dict(self):
        manager = mp.Manager()

        self.shared_local_models_dict = manager.dict()
        self.time_dict_data_preparation = manager.dict()
        self.time_dict_model_preparation = manager.dict()
        self.data_shares_dict = manager.dict()
        self.time_dict_secure_testing = manager.dict()
        self.acc_dict = manager.dict()

    def clear_shared_dicts(self):
        self.shared_local_models_dict = self.shared_local_models_dict.copy
        self.time_dict_data_preparation = self.time_dict_data_preparation.copy()
        self.time_dict_model_preparation = self.time_dict_model_preparation.copy()
        self.time_dict_secure_testing = self.time_dict_secure_testing.copy()
        self.acc_dict = self.acc_dict.copy()

    def setup(self):
        print("Set up environment")
        start = time.process_time()
        self.init_shared_dict()
        self.secretly_share_models_and_data_parallel()
        self.eval_init_model()
        self.time_dict["parallel"] += time.process_time() - start + self.time_dict["communication"] - self.time_dict["repeated"]
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
        self.clear_shared_dicts()

