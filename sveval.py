import torch
import math
import numpy as np
import random
from tqdm import tqdm
import time
from shapley import *
from utils import *
import copy
import random
from train import *
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import torch.multiprocessing as mp

try:
     mp.set_start_method('spawn')
except RuntimeError:
    pass


class SVEval:
    def __init__(self, clients, test_data_usage=1.0, no_cuda=False):
        self.clients = clients
        self.cids = list(self.clients.data.keys())
        self.test_data_usage = test_data_usage
        self.n_classes = self.clients.get_init_model().n_classes
        self.test_size = len(self.clients.return_test_data(self.test_data_usage))
        self.batch_size = self.test_size
        self.T = len(clients.selection_record)
        self.init_acc = 1 / self.n_classes
        self.ssv_dict = {}
        self.fsv_dict = {}
        self.ssv_dict_skip = {}
        self.fsv_dict_skip = {}
        self.skip = True
        self.noskip = True
        self.skip_compare = False
        self.dir = clients.dir
        self.indices = np.arange(self.test_size)
        self.no_cuda = no_cuda
        self.use_cuda = not self.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.skip_compare_dict_sum = {}
        self.time_dict = {}
        self.init_time_dict(self.time_dict)
        self.skip_init_model = True
        self.rate_of_skipping = 1.0
        self.n_processes = 10

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

    def submit_data_parallel(self):
        print("\nSubmit test data")
        self.parallelize(self.submit_data, [(cid,) for cid in self.cids])
        test_data = self.clients.return_test_data(self.test_data_usage)
        self.test_data = torch.utils.data.Subset(test_data, self.indices)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

    def submit_data(self, cid):
        time_dict = {}
        self.init_time_dict(time_dict)
        start = time.process_time()

        test_data = self.clients.data[cid].test_data(self.test_data_usage)
        data_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=False)

        for (data, truth) in data_loader:
            time_dict["communication"] += communicate(data.storage()) + communicate(truth.storage())

        time_dict["parallel"] += time.process_time() - start + time_dict["communication"] - time_dict["repeated"]
        time_dict["sequential"] = time_dict["parallel"]
        self.time_dict_data_preparation[cid] = time_dict

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
        if skip and self.skip_compare:
            data = {"test size": self.test_size,
                    "ssv": self.ssv_dict_skip, "fsv": self.fsv_dict_skip,
                    "time for main process": self.time_dict_skip,
                    "time for secure testing": self.time_dict_testing_skip.copy(),
                    "time for data preparation": self.time_dict_data_preparation.copy(),
                    # "skip_rnds": self.skip_compare_dict,
                    "skip_sum": self.skip_compare_dict_sum
                    }
        elif skip:
            data = {"test size": self.test_size,
                    "ssv": self.ssv_dict_skip, "fsv": self.fsv_dict_skip,
                    "time for main process": self.time_dict_skip,
                    "time for secure testing": self.time_dict_testing_skip.copy(),
                    "time for data preparation": self.time_dict_data_preparation.copy()
                    }
        else:
            data = {"test size": self.test_size,
                    "ssv": self.ssv_dict, "fsv": self.fsv_dict,
                    "time for main process": self.time_dict,
                    "time for secure testing": self.time_dict_testing.copy(),
                    "time for data preparation": self.time_dict_data_preparation.copy()}

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

    def summarize_skip_compare_dict(self):
        skip_compare_dict = self.skip_compare_dict
        total_corrects, total_size, corrects, naive, wrong_naive = 0, 0, 0, 0, 0

        max_wrong_perc, min_wrong_perc = 0.0, 0.0
        for sub_dict in skip_compare_dict.values():
            if type(sub_dict) == dict:
                for subsub_dict in sub_dict.values():
                    total_size += self.test_size
                    total_corrects += subsub_dict["total corrects"]
                    corrects += subsub_dict["corrects"]
                    naive += subsub_dict["naive"]
                    wrong_naive += len(subsub_dict["wrong naive"])
                    max_wrong_perc = max(subsub_dict["wrong perc."], max_wrong_perc)
                    min_wrong_perc = min(subsub_dict["wrong perc."], min_wrong_perc)

        self.skip_compare_dict_sum["total size"] = int(total_size)
        self.skip_compare_dict_sum["total corrects"] = total_corrects
        self.skip_compare_dict_sum["corrects"] = corrects
        self.skip_compare_dict_sum["naive"] = naive
        self.skip_compare_dict_sum["wrong naive"] = wrong_naive
        self.skip_compare_dict_sum["max. wrong perc."] = max_wrong_perc
        self.skip_compare_dict_sum["min. wrong perc."] = min_wrong_perc
        print(self.skip_compare_dict_sum)

    def eval(self, model, indices):
        if len(indices) == self.test_size:
            test_loader = self.test_loader
        else:
            test_data = torch.utils.data.Subset(self.test_data, indices)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        if len(test_loader) == 0:
            return set()

        model = model.to(self.device)
        model.eval()
        total_size = 0
        correct_ids_ls = []

        for data, target in test_loader:
            data = data.reshape(model.input_shape).float().to(self.device)
            target = target.to(self.device)
            output = model(data)
            _, pred = torch.max(output, 1)
            correct_ids = indices[np.where(pred.eq(target.data.view_as(pred)).cpu())[0] + total_size]
            correct_ids_ls.append(correct_ids)
            total_size += len(target)

        correct_ids = np.concatenate(correct_ids_ls).tolist()

        return set(correct_ids)

    def throw_error(self, e):
        raise e

    def find_skippable(self, model_ids, correct_samples_dict):
        naive_samples = set([])
        all_subsets = make_all_subsets(list(model_ids))
        for subset in all_subsets:
            if len(subset) in range(1, len(model_ids)) and subset in correct_samples_dict.keys():
                complement_set = model_ids - subset
                new_naive_samples = correct_samples_dict[subset].intersection(correct_samples_dict[complement_set])
                naive_samples = naive_samples.union(new_naive_samples)

        return naive_samples

    def find_subsets_for_eval(self, rnd):
        sel_clients = self.clients.selected_clients(rnd)
        all_subsets = make_all_subsets(list(sel_clients.keys()))
        self.local_subsets[rnd] = [subset for subset in all_subsets if len(subset) == 1]
        self.aggr_subsets[rnd] = [subset for subset in all_subsets if len(subset) > 1]

    def eval_local_models(self, rnd):
        correct_samples_dict, acc_dict = {}, {}
        subsets = self.local_subsets[rnd]
        for subset in tqdm(subsets):
            id = list(subset)[0]
            client = self.clients.find_client(id)
            local_model = client.get_model(rnd)
            correct_samples = self.eval(local_model, self.indices)
            acc_dict[subset] = len(correct_samples) / self.test_size
            correct_samples_dict[subset] = correct_samples

        return correct_samples_dict, acc_dict

    def eval_aggr_models(self, correct_samples_dict, rnd, skip=False):
        acc_dict, naive_samples_dict, skipped_samples_dict, skip_compare_dict = {}, {}, {}, {}
        subsets = self.aggr_subsets[rnd]
        subsets.sort(key=lambda x: len(x), reverse=False)
        for subset in tqdm(subsets):
            skipped_naive_samples = set([])
            naive_samples = set([])
            if skip:
                naive_samples = self.find_skippable(subset, correct_samples_dict)
                skipped_naive_samples = set(np.random.choice(list(naive_samples),
                                                             int(len(naive_samples) * self.rate_of_skipping), False))
                naive_samples_dict[subset] = naive_samples
                skipped_samples_dict[subset] = skipped_naive_samples

                indices = np.array(sorted(list(set(self.indices) - naive_samples)))
            else:
                indices = self.indices

            if len(indices) > 0:
                model_ls = self.clients.get_model_list(list(subset), rnd)
                size_ls = self.clients.get_train_size_list(list(subset))
                aggr_model = FL.fedavg(model_ls, size_ls)
                correct_samples = self.eval(aggr_model, indices)
                correct_samples_dict[subset] = correct_samples.union(skipped_naive_samples)
                acc_dict[subset] = len(correct_samples_dict[subset]) / self.test_size
            else:
                correct_samples_dict[subset] = set(self.indices)
                acc_dict[subset] = 1.0

            if skip and self.skip_compare:
                compare_dict = {}
                all_correct_samples = self.eval(aggr_model, self.indices)
                compare_dict["total corrects"] = len(all_correct_samples)
                compare_dict["corrects"] = len(correct_samples)
                compare_dict["naive"] = len(naive_samples)
                compare_dict["wrong naive"] = list(naive_samples.difference(all_correct_samples))
                compare_dict["wrong perc."] = len(compare_dict["wrong naive"]) / self.test_size

                skip_compare_dict[str(list(subset))] = compare_dict

        return correct_samples_dict, acc_dict, naive_samples_dict, skipped_samples_dict, skip_compare_dict

    def sv_eval_one_rnd_aggr_models(self, acc_dict, correct_samples_dict, time_dict, rnd, skip=False):
        start = time.process_time()
        correct_samples_dict, new_acc_dict, naive_samples_dict, skipped_samples_dict, skip_compare_dict = self.eval_aggr_models(
            correct_samples_dict, rnd, skip=skip)

        acc_dict.update(new_acc_dict)
        time_taken = time.process_time() - start + time_dict["communication"] - time_dict["repeated"]
        time_dict["parallel"] += time_taken
        time_dict["sequential"] += time_taken

        if skip:
            self.time_dict_testing_skip[rnd] = time_dict
            self.acc_dict_skip[rnd] = acc_dict
            self.correct_samples_dict_skip[rnd] = correct_samples_dict

            self.naive_dict[rnd] = naive_samples_dict
            self.skipped_dict[rnd] = skipped_samples_dict
            self.skip_compare_dict[rnd] = skip_compare_dict
        else:
            self.time_dict_testing[rnd] = time_dict
            self.acc_dict[rnd] = acc_dict
            self.correct_samples_dict[rnd] = correct_samples_dict

    def sv_eval_one_rnd(self, rnd):
        set_random_seed((os.getpid() * int(time.time())) % 123456789)
        if self.use_cuda: torch.cuda.set_device(rnd % 2)
        time_dict = {}
        self.init_time_dict(time_dict)

        start = time.process_time()
        self.find_subsets_for_eval(rnd)
        correct_samples_dict, acc_dict = self.eval_local_models(rnd)
        time_dict["parallel"] += time.process_time() - start
        time_dict["sequential"] = time_dict["parallel"]

        if self.noskip:
            self.sv_eval_one_rnd_aggr_models(acc_dict.copy(), correct_samples_dict.copy(), time_dict.copy(), rnd, skip=False)
        if self.skip:
            self.sv_eval_one_rnd_aggr_models(acc_dict.copy(), correct_samples_dict.copy(), time_dict.copy(), rnd, skip=True)

    def init_shared_dict(self):
        manager = mp.Manager()

        self.time_dict_data_preparation = manager.dict()
        self.time_dict_testing = manager.dict()
        self.time_dict_testing_skip = manager.dict()

        self.acc_dict = manager.dict()
        self.acc_dict_skip = manager.dict()
        self.naive_dict = manager.dict()
        self.skipped_dict = manager.dict()
        self.skip_compare_dict = manager.dict()
        self.correct_samples_dict = manager.dict()
        self.correct_samples_dict_skip = manager.dict()

        self.local_subsets = manager.dict()
        self.aggr_subsets = manager.dict()

    def clear_shared_dicts(self):
        self.time_dict_data_preparation = self.time_dict_data_preparation.copy()

        self.time_dict_testing = self.time_dict_testing.copy()
        self.time_dict_testing_skip = self.time_dict_testing_skip.copy()

        self.acc_dict = self.acc_dict.copy()
        self.acc_dict_skip = self.acc_dict_skip.copy()
        self.naive_dict = self.naive_dict.copy()
        self.skipped_dict = self.skipped_dict.copy()
        self.skip_compare_dict = self.skip_compare_dict.copy()
        self.correct_samples_dict = self.correct_samples_dict.copy()
        self.correct_samples_dict_skip = self.correct_samples_dict_skip.copy()

        self.local_subsets = self.local_subsets.copy()
        self.aggr_subsets = self.aggr_subsets.copy()

    def eval_init_model(self):
        if self.skip_init_model:
            return

        print("\nEvaluate the initial model")
        if self.debug:
            model = self.clients.get_global_model(self.T - 1)
        else:
            model = self.clients.get_init_model()

        correct_nb = self.eval(model, self.test_data)
        self.init_acc = correct_nb / self.test_size


    def setup(self):
        print("Set up environment")
        start = time.process_time()
        self.init_shared_dict()
        self.submit_data_parallel()
        self.eval_init_model()
        self.time_dict["parallel"] += time.process_time() - start + self.time_dict["communication"] - self.time_dict["repeated"]
        self.time_dict["sequential"] = self.time_dict["parallel"]

    def parallelize(self, func, args_list):
        pool = mp.Pool(self.n_processes)
        workers = []
        for args in args_list:
            worker = pool.apply_async(func, args=args, error_callback=self.throw_error)
            workers.append(worker)

        pool.close()
        pool.join()

        for worker in workers:
            worker.get()

    def testing(self):
        print("\nEvaluate each FL round in parallel")
        self.parallelize(self.sv_eval_one_rnd, [(rnd,) for rnd in range(self.T)])

    def print_time_dicts(self):
        if self.noskip:
            print("\ntime for main process")
            print(self.time_dict)
            print("\ntime for data preparation")
            print(self.time_dict_data_preparation)
            print("\ntime for secure testing")
            print(self.time_dict_testing)

        if self.skip:
            print("\ntime for main process (skip)")
            print(self.time_dict_skip)
            print("\ntime for data preparation (skip)")
            print(self.time_dict_data_preparation)
            print("\ntime for secure testing (skip)")
            print(self.time_dict_testing_skip)

    def sv_eval_mul_rnds_rparallel(self):
        self.setup()
        self.testing()
        self.calc_fsv()
        self.print_time_dicts()
        self.clear_shared_dicts()
        if self.skip:
            self.summarize_skip_compare_dict()

