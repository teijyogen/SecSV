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
from shapley import *
from utils import *
import copy
import random
from collections import Counter
import random128
from train import *
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch.multiprocessing as mp

try:
     mp.set_start_method('spawn')
except RuntimeError:
    pass


class SVEval:
    def __init__(self, clients):
        self.clients = clients
        self.test_data = torch.utils.data.ConcatDataset([clients.data[id].test_data() for id in clients.data.keys()])
        self.idx_set = set([i for i in range(len(self.test_data))])
        self.batch_size = 64
        self.T = len(clients.selection_record)
        self.init_accs = {}
        self.skip = False
        self.skip_compare = False
        self.ssv_dict = {}
        self.msv_dict = {}
        self.dirs = "data/"
        self.test_size = len(self.test_data)
        self.input_shape = (-1, 1, 28, 28)
        self.no_cuda = False
        self.use_cuda = not self.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.skip_dict = {}

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
        data = {"ssv": self.ssv_dict, "msv": self.msv_dict, "skip": self.skip_dict}
        with open(self.dirs+filename, "w") as f:
            json.dump(data, f, indent=4)

        f.close()

    def eval(self, model, test_loader):

        if len(test_loader) == 0:
            return set()

        model = model.to(self.device)
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))

        model.eval()
        total_size = 0
        correct_ids_ls = []
        for data, target in test_loader:
            data = data.reshape(self.input_shape).to(self.device)
            target = target.to(self.device)
            output = model(data)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # compare predictions to true label
            correct_ids = np.where(pred.eq(target.data.view_as(pred)).cpu())[0] + total_size
            correct_ids_ls.append(correct_ids)
            total_size += len(target)

        correct_ids = np.concatenate(correct_ids_ls)

        return set(correct_ids)

    def sv_eval_one_rnd(self, rnd):
        clients = self.clients

        sel_clients = clients.selected_clients(rnd)
        all_subsets = make_all_subsets(list(sel_clients.keys()))
        acc_dict = {frozenset(): self.init_accs[rnd]}

        for client in tqdm(sel_clients.values()):
            subset = frozenset((client.id,))
            local_model = torch.load(client.models[rnd])
            correct_samples = self.eval(local_model, self.test_loader)
            acc_dict[subset] = len(correct_samples) / self.test_size

        for subset in tqdm(all_subsets):
            if len(subset) < 2:
                continue

            model_ls = clients.get_model_list(list(subset), rnd)
            size_ls = clients.get_train_size_list(list(subset), rnd)

            aggr_model = FL.fedavg(model_ls, size_ls)
            correct_samples = self.eval(aggr_model, self.test_loader)
            acc = len(correct_samples) / self.test_size
            acc_dict[subset] = acc
            if len(subset) == len(sel_clients):
                self.init_accs[rnd+1] = acc

        # print(acc_dict)

        SV = ShapleyValue(sel_clients.keys(), acc_dict)

        self.update_ssv(SV.svs, rnd)


    def sv_eval_mul_rnds(self):
        init_model = self.clients.init_model
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
        correct_samples = self.eval(torch.load(init_model), self.test_loader)
        self.init_accs[0] = len(correct_samples) / self.test_size

        for rnd in tqdm(range(self.T)):
            self.sv_eval_one_rnd(rnd)

        self.cal_msv()

    def sv_eval_one_rnd_rparallel(self, rnd, rnds_acc_dict, skip_dict):
        test_loader = self.test_loader
        clients = self.clients

        rnd_skip_dict = {"test size": 0, "naive size": 0, "hard size": 0, "wrong times": 0}
        acc_dict = {}
        sel_clients = clients.selected_clients(rnd)
        all_subsets = make_all_subsets(list(sel_clients.keys()))

        naive_samples = self.idx_set
        hard_samples = self.idx_set

        for client in sel_clients.values():
            subset = frozenset((client.id,))
            local_model = torch.load(client.models[rnd])
            correct_samples = self.eval(local_model, test_loader)
            acc_dict[subset] = len(correct_samples) / self.test_size
            if self.skip:
                naive_samples = naive_samples.intersection(correct_samples)
                hard_samples = hard_samples.intersection(self.idx_set - correct_samples)

        if self.skip:
            test_data = torch.utils.data.Subset(self.test_data, list(self.idx_set-naive_samples-hard_samples))
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
            # print("----------------------")
            # print(len(test_data))
            # print(len(naive_samples))
            # print(len(hard_samples))
            # print("----------------------")

        total_wrong_naive = 0
        total_wrong_hard = 0
        total_skips = 0
        for subset in all_subsets:
            if len(subset) < 2:
                continue



            model_ls = clients.get_model_list(list(subset), rnd)
            size_ls = clients.get_train_size_list(list(subset), rnd)

            aggr_model = FL.fedavg(model_ls, size_ls)

            correct_samples = self.eval(aggr_model, test_loader)

            if self.skip:
                total_skips += 1
                acc = (len(naive_samples) + len(correct_samples)) / self.test_size
                if self.skip_compare:
                    true_correct_samples = self.eval(aggr_model, self.test_loader)
                    if len(true_correct_samples) != (len(naive_samples) + len(correct_samples)):
                        correct_size = len(true_correct_samples)
                        appr_correct_size = len(naive_samples) + len(correct_samples)
                        wrong_naive = len(naive_samples) - len(naive_samples.intersection(true_correct_samples))
                        wrong_hard = len(hard_samples) - len(hard_samples.intersection(self.idx_set-true_correct_samples))
                        # print("Wrong skip")
                        # print("Number of correct samples", correct_size)
                        # print("Number of approximately correct samples", appr_correct_size)
                        # print("Number of wrong naive samples", wrong_naive)
                        # print("Number of wrong hard samples", wrong_hard)
                        rnd_skip_dict["wrong times"] += 1
                        # sub_dict = {"correct size": correct_size, "wrong naive": wrong_naive, "wrong hard": wrong_hard}
                        # rnd_skip_dict["-".join(sorted(list(subset)))] = sub_dict
                        total_wrong_naive += wrong_naive
                        total_wrong_hard += wrong_hard
                    # else:
                        # sub_dict = {"correct size": len(true_correct_samples), "wrong naive": 0, "wrong hard": 0}
                        # rnd_skip_dict["-".join(sorted(list(subset)))] = sub_dict
            else:
                acc = len(correct_samples) / self.test_size

            acc_dict[subset] = acc

        rnd_skip_dict["test size"] += self.test_size * total_skips
        rnd_skip_dict["naive size"] += len(naive_samples) * total_skips
        rnd_skip_dict["hard size"] += len(hard_samples) * total_skips
        rnd_skip_dict["wrong naive"] = total_wrong_naive
        rnd_skip_dict["wrong hard"] = total_wrong_hard
        rnd_skip_dict["skip times"] = total_skips
        rnds_acc_dict[rnd] = acc_dict
        # print(rnd_skip_dict)
        skip_dict[rnd] = rnd_skip_dict

    def sv_eval_mul_rnds_rparallel(self):
        # model = torch.load("/home/zheng/securesv/model/mnist/iid/logi/rnd9/global.pkl")
        # self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
        # correct_samples = self.eval(model, self.test_loader)
        init_model = self.clients.init_model
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
        correct_samples = self.eval(torch.load(init_model), self.test_loader)
        # print(len(correct_samples) / self.test_size)
        self.init_accs[0] = len(correct_samples) / self.test_size

        manager = mp.Manager()
        rnds_acc_dict = manager.dict()
        skip_dict = manager.dict()
        pool = mp.Pool(5)

        workers = []
        for rnd in range(self.T):
            worker = pool.apply_async(self.sv_eval_one_rnd_rparallel, args=(rnd, rnds_acc_dict, skip_dict))
            workers.append(worker)


        pool.close()
        pool.join()

        for worker in workers:
            worker.get()

        self.skip_dict = skip_dict.copy()

        for rnd in rnds_acc_dict:
            acc_dict = rnds_acc_dict[rnd]
            if rnd > 0:
                self.init_accs[rnd] = rnds_acc_dict[rnd-1][frozenset(self.clients.selected_ids(rnd-1))]
            acc_dict[frozenset()] = self.init_accs[rnd]
            SV = ShapleyValue(self.clients.selection_record[rnd], acc_dict)
            self.update_ssv(SV.svs, rnd)

        print(self.init_accs)
        # print(rnds_acc_dict)
        self.cal_msv()
        # print(self.msv_dict)

if __name__ == '__main__':


    for run in tqdm(range(0, 10)):
        clients = Clients()
        clients.dirs = "data/cifar_cnn2/dir2.0/%s/" % (run)
        clients.load("clients.data")

        sveval = SVEval(clients)
        sveval.skip = True
        sveval.skip_compare = True
        # sveval.input_shape = (-1, 784)
        # sveval.input_shape = (-1, 3072)
        sveval.input_shape = (-1, 3, 32, 32)
        sveval.sv_eval_mul_rnds_rparallel()
        sveval.dirs = clients.dirs
        sveval.save_stat("sv_skip.json")








