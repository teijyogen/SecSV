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
        self.global_skip = True

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

    def intersection_except(self, samples_dict, except_key):
        intersection_samples = self.idx_set
        for key, samples in samples_dict.items():
            if key == except_key:
                continue
            intersection_samples = intersection_samples.intersection(samples)
        return intersection_samples

    def test_assumption(self, subset_samples1, subset_samples2, samples_dict, keyword="naive"):
        condition = self.intersection_except(samples_dict, None)
        if len(condition) == 0:
            pr1 = 1.0
            pr2 = 1.0
        else:
            pr1 = len(subset_samples1.intersection(condition)) / len(condition)
            pr2 = len(subset_samples2.intersection(condition)) / len(condition)

        if pr1 < pr2:
            print("--------------------")
            print(keyword)
            print("Probability 1:", pr1)
            print("Probability 2:", pr2)
            print("--------------------")

        if pr1 < 0.95:
            print("--------------------")
            print("High failure rate:", pr1)
            print("--------------------")

        if abs(pr1 - pr2) > 0.05:
            print("--------------------")
            print("Large distance")
            print("Probability 1:", pr1)
            print("Probability 2:", pr2)
            print("--------------------")

    # def test_assumption(self, subset_samples, samples_dict, keyword="naive"):
    #     condition = self.intersection_except(samples_dict, None)
    #     if len(condition) == 0:
    #         pr1 = 1.0
    #     else:
    #         pr1 = len(subset_samples[1].intersection(condition)) / len(condition)
    #
    #     for key in subset_samples[0]:
    #         # if type(key) == frozenset:
    #         #     continue
    #         self.test_assumption_times += 1
    #         condition = self.intersection_except(samples_dict, key)
    #         if len(condition) == 0:
    #             pr2 = 1.0
    #             pr3 = 1.0
    #         else:
    #             pr2 = len(subset_samples[1].intersection(condition)) / len(condition)
    #             pr3 = len(samples_dict[key].intersection(condition)) / len(condition)
    #
    #         if pr1 < pr3:
    #             self.test_assumption_times += 1
    #             print("--------------------")
    #             print(keyword)
    #             print("Probability 1:", pr1)
    #             print("Probability 2:", pr2)
    #             print("Probability 3:", pr3)
    #             print("--------------------")

    def throw_error(self, e):
        raise e

    def sv_eval_one_rnd_rparallel(self, rnd, rnds_acc_dict, skip_dict):
        test_loader = self.test_loader
        clients = self.clients

        rnd_skip_dict = {"test size": 0, "naive size": 0, "wrong times": 0}
        acc_dict = {}
        sel_clients = clients.selected_clients(rnd)
        all_subsets = make_all_subsets(list(sel_clients.keys()))

        correct_samples_dict = {}
        wrong_samples_dict = {}

        for client in sel_clients.values():
            subset = frozenset((client.id,))
            local_model = torch.load(client.models[rnd])
            correct_samples = self.eval(local_model, test_loader)
            acc_dict[subset] = len(correct_samples) / self.test_size
            correct_samples_dict[client.id] = correct_samples
            # wrong_samples_dict[client.id] = self.idx_set - correct_samples

        # if rnd == 0:
        #     base_correct_samples = self.eval(torch.load(self.clients.init_model), test_loader)
        # else:
        #     largest_set = frozenset(clients.selected_clients(rnd - 1).keys())
        #     model_ls = clients.get_model_list(list(largest_set), rnd - 1)
        #     size_ls = clients.get_train_size_list(list(largest_set), rnd - 1)
        #     aggr_model = FL.fedavg(model_ls, size_ls)
        #     base_correct_samples = self.eval(aggr_model, test_loader)

        # largest_set = frozenset(sel_clients.keys())
        # model_ls = clients.get_model_list(list(largest_set), rnd)
        # size_ls = clients.get_train_size_list(list(largest_set), rnd)
        # aggr_model = FL.fedavg(model_ls, size_ls)
        # global_correct_samples = self.eval(aggr_model, test_loader)
        # acc_dict[largest_set] = len(global_correct_samples) / self.test_size

        # correct_samples_dict[largest_set] = correct_samples
        # wrong_samples_dict[largest_set] = self.idx_set - correct_samples


        if self.skip and self.global_skip:
            naive_samples = self.intersection_except(correct_samples_dict, None)
            # hard_samples = self.intersection_except(wrong_samples_dict, None)
            test_data = torch.utils.data.Subset(self.test_data, list(self.idx_set-naive_samples))
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
            # print("----------------------")
            # print(len(test_data))
            # print(len(naive_samples))
            # print(len(hard_samples))
            # print("----------------------")

        total_wrong_naive = 0
        # total_wrong_hard = 0
        total_skips = 0
        for subset in all_subsets:
            # if len(subset) < 2 or len(subset) == len(sel_clients.values()):
            if len(subset) < 2:
                continue

            if self.skip and not self.global_skip:
                sub_correct_samples_dict = {key: value for key, value in correct_samples_dict.items() if key in subset}
                # sub_wrong_samples_dict = {key: value for key, value in wrong_samples_dict.items() if key in subset}
                naive_samples = self.intersection_except(sub_correct_samples_dict, None)
                # hard_samples = self.intersection_except(sub_wrong_samples_dict, None)
                test_data = torch.utils.data.Subset(self.test_data, list(self.idx_set - naive_samples))
                test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

            model_ls = clients.get_model_list(list(subset), rnd)
            size_ls = clients.get_train_size_list(list(subset), rnd)
            aggr_model = FL.fedavg(model_ls, size_ls)
            correct_samples = self.eval(aggr_model, test_loader)


            if self.skip:
                total_skips += 1
                acc = (len(naive_samples) + len(correct_samples)) / self.test_size
                if self.skip_compare:
                    true_correct_samples = self.eval(aggr_model, self.test_loader)
                    # self.test_assumption((largest_set - subset, true_correct_samples), correct_samples_dict, "naive of rnd %s" %(rnd))
                    # self.test_assumption((largest_set - subset, self.idx_set-true_correct_samples), wrong_samples_dict, "hard of rnd %s" %(rnd))
                    # self.test_assumption(true_correct_samples, base_correct_samples, correct_samples_dict,
                                         #
                    # sub_dict = {key: value for key, value in correct_samples_dict.items() if key in subset}
                    # self.test_assumption(true_correct_samples, base_correct_samples, sub_dict, "hard of rnd %s" % (rnd))

                    if len(true_correct_samples) != (len(naive_samples) + len(correct_samples)):
                        correct_size = len(true_correct_samples)
                        appr_correct_size = len(naive_samples) + len(correct_samples)
                        wrong_naive = len(naive_samples) - len(naive_samples.intersection(true_correct_samples))
                        # wrong_hard = len(hard_samples) - len(hard_samples.intersection(self.idx_set-true_correct_samples))
                        # ids_wrong_hard = hard_samples - hard_samples.intersection(self.idx_set-true_correct_samples)
                        print("----------------------")
                        print("Wrong skip of rnd", rnd)
                        print("Clients IDs", subset)
                        # print("Number of correct samples", correct_size)
                        # print("Number of approximately correct samples", appr_correct_size)
                        print("Number of wrong naive samples", wrong_naive)
                        # print("Number of wrong hard samples", wrong_hard)
                        # print("IDs of wrong hard samples", ids_wrong_hard)
                        print("----------------------")
                        rnd_skip_dict["wrong times"] += 1
                        # sub_dict = {"correct size": correct_size, "wrong naive": wrong_naive, "wrong hard": wrong_hard}
                        # rnd_skip_dict["-".join(sorted(list(subset)))] = sub_dict
                        total_wrong_naive += wrong_naive
                        # total_wrong_hard += wrong_hard
                    # else:
                        # sub_dict = {"correct size": len(true_correct_samples), "wrong naive": 0, "wrong hard": 0}
                        # rnd_skip_dict["-".join(sorted(list(subset)))] = sub_dict
            else:
                acc = len(correct_samples) / self.test_size

            acc_dict[subset] = acc

        rnd_skip_dict["test size"] += self.test_size * total_skips
        rnd_skip_dict["naive size"] += len(naive_samples) * total_skips
        # rnd_skip_dict["hard size"] += len(hard_samples) * total_skips
        rnd_skip_dict["wrong naive"] = total_wrong_naive
        # rnd_skip_dict["wrong hard"] = total_wrong_hard
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
        pool = mp.Pool(10)

        workers = []
        for rnd in range(self.T):
            worker = pool.apply_async(self.sv_eval_one_rnd_rparallel, args=(rnd, rnds_acc_dict, skip_dict), error_callback=self.throw_error)
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


class SVPerm(SVEval):
    def __init__(self, clients, epsi, sigma):
        SVEval.__init__(self, clients)
        self.epsi = epsi
        self.sigma = sigma
        self.N = len(clients.selection_record[0])
        self.esti_times = math.ceil(2 * self.N / (epsi ** 2) * math.log(2 * self.N / sigma))

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

    def sv_eval_one_rnd_rparallel(self, rnd, rnds_acc_dict, skip_dict, perms_dict):
        test_loader = self.test_loader
        clients = self.clients

        rnd_skip_dict = {"test size": 0, "naive size": 0, "wrong times": 0}
        acc_dict = {}
        sel_clients = clients.selected_clients(rnd)
        perm_ls, model_subsets = self.permutation_sampling(list(sel_clients.keys()))
        # print(model_subsets)
        print(len(model_subsets))


        correct_samples_dict = {}
        total_wrong_naive = 0
        total_skips = 0

        for subset in model_subsets:
            if self.skip and len(subset) > 1:
                naive_samples = self.find_skippable(subset, correct_samples_dict)
                if len(naive_samples) == 0:
                    print(subset)
                test_data = torch.utils.data.Subset(self.test_data, list(self.idx_set - naive_samples))
                test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
            model_ls = clients.get_model_list(list(subset), rnd)
            size_ls = clients.get_train_size_list(list(subset), rnd)
            aggr_model = FL.fedavg(model_ls, size_ls)
            correct_samples = self.eval(aggr_model, test_loader)

            if self.skip and len(subset) > 1:
                total_skips += 1
                acc = (len(naive_samples) + len(correct_samples)) / self.test_size
                if self.skip_compare:
                    true_correct_samples = self.eval(aggr_model, self.test_loader)

                    if len(true_correct_samples) != (len(naive_samples) + len(correct_samples)):
                        correct_size = len(true_correct_samples)
                        appr_correct_size = len(naive_samples) + len(correct_samples)
                        wrong_naive = len(naive_samples) - len(naive_samples.intersection(true_correct_samples))

                        print("----------------------")
                        print("Wrong skip of rnd", rnd)
                        print("Clients IDs", subset)
                        # print("Number of correct samples", correct_size)
                        # print("Number of approximately correct samples", appr_correct_size)
                        print("Number of wrong naive samples", wrong_naive)
                        # print("Number of wrong hard samples", wrong_hard)
                        # print("IDs of wrong hard samples", ids_wrong_hard)
                        print("----------------------")
                        rnd_skip_dict["wrong times"] += 1
                        # sub_dict = {"correct size": correct_size, "wrong naive": wrong_naive, "wrong hard": wrong_hard}
                        # rnd_skip_dict["-".join(sorted(list(subset)))] = sub_dict
                        total_wrong_naive += wrong_naive
                        # total_wrong_hard += wrong_hard
                    # else:
                        # sub_dict = {"correct size": len(true_correct_samples), "wrong naive": 0, "wrong hard": 0}
                        # rnd_skip_dict["-".join(sorted(list(subset)))] = sub_dict
            else:
                acc = len(correct_samples) / self.test_size

            correct_samples_dict[subset] = correct_samples
            acc_dict[subset] = acc

        rnd_skip_dict["test size"] += self.test_size * total_skips
        rnd_skip_dict["naive size"] += len(naive_samples) * total_skips
        # rnd_skip_dict["hard size"] += len(hard_samples) * total_skips
        rnd_skip_dict["wrong naive"] = total_wrong_naive
        # rnd_skip_dict["wrong hard"] = total_wrong_hard
        rnd_skip_dict["skip times"] = total_skips
        rnds_acc_dict[rnd] = acc_dict
        # print(rnd_skip_dict)
        skip_dict[rnd] = rnd_skip_dict
        perms_dict[rnd] = perm_ls

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
        perms_dict = manager.dict()
        pool = mp.Pool(10)

        workers = []
        for rnd in range(self.T):
            worker = pool.apply_async(self.sv_eval_one_rnd_rparallel, args=(rnd, rnds_acc_dict, skip_dict, perms_dict),
                                      error_callback=self.throw_error)
            workers.append(worker)

        pool.close()
        pool.join()

        for worker in workers:
            worker.get()

        self.skip_dict = skip_dict.copy()

        for rnd in rnds_acc_dict:
            acc_dict = rnds_acc_dict[rnd]
            if rnd > 0:
                self.init_accs[rnd] = rnds_acc_dict[rnd - 1][frozenset(self.clients.selected_ids(rnd - 1))]
            acc_dict[frozenset()] = self.init_accs[rnd]
            SV = ShapleyValue(self.clients.selection_record[rnd], acc_dict)
            SV.calculate_svs_perm(perms_dict[rnd])
            self.update_ssv(SV.svs, rnd)

        print(self.init_accs)
        # print(rnds_acc_dict)
        self.cal_msv()
        # print(self.msv_dict)

class SVGroupTesting(SVEval):
    def __init__(self, clients, epsi, sigma):
        SVEval.__init__(self, clients)
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
        qtot = (N - 2) / N * q[0] + np.array([q[k-1] * (1 + 2 * k * (k - N) / N / (N -1)) for k in range(2, N)]).sum().item()
        def h(u):
            return (1 + u) * math.log(1 + u) - u
        self.esti_times = math.ceil(8 * math.log(N * (N -1) / (2 * sigma)) / ((1 - qtot ** 2) * h(epsi / (Z * N ** 0.5 * (1 - qtot ** 2)))))

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

    def sv_eval_one_rnd_rparallel(self, rnd, rnds_acc_dict, skip_dict, beta_mat_dict, model_subsets_ls_dict):
        test_loader = self.test_loader
        clients = self.clients

        rnd_skip_dict = {"test size": 0, "naive size": 0, "wrong times": 0}
        acc_dict = {}
        sel_clients = clients.selected_clients(rnd)
        beta_mat, model_subsets_ls = self.group_testing_sampling(list(sel_clients.keys()))
        # print(model_subsets)

        model_subsets = list(set(model_subsets_ls + [frozenset(sel_clients)]))
        model_subsets.sort(key=lambda x:len(x))
        print(len(model_subsets))

        correct_samples_dict = {}
        total_wrong_naive = 0
        total_skips = 0

        for subset in model_subsets:
            if self.skip and len(subset) > 1:
                naive_samples = self.find_skippable(subset, correct_samples_dict)
                if len(naive_samples) == 0:
                    print(subset)
                test_data = torch.utils.data.Subset(self.test_data, list(self.idx_set - naive_samples))
                test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
            model_ls = clients.get_model_list(list(subset), rnd)
            size_ls = clients.get_train_size_list(list(subset), rnd)
            aggr_model = FL.fedavg(model_ls, size_ls)
            correct_samples = self.eval(aggr_model, test_loader)

            if self.skip and len(subset) > 1:
                total_skips += 1
                acc = (len(naive_samples) + len(correct_samples)) / self.test_size
                if self.skip_compare:
                    true_correct_samples = self.eval(aggr_model, self.test_loader)

                    if len(true_correct_samples) != (len(naive_samples) + len(correct_samples)):
                        correct_size = len(true_correct_samples)
                        appr_correct_size = len(naive_samples) + len(correct_samples)
                        wrong_naive = len(naive_samples) - len(naive_samples.intersection(true_correct_samples))

                        print("----------------------")
                        print("Wrong skip of rnd", rnd)
                        print("Clients IDs", subset)
                        # print("Number of correct samples", correct_size)
                        # print("Number of approximately correct samples", appr_correct_size)
                        print("Number of wrong naive samples", wrong_naive)
                        # print("Number of wrong hard samples", wrong_hard)
                        # print("IDs of wrong hard samples", ids_wrong_hard)
                        print("----------------------")
                        rnd_skip_dict["wrong times"] += 1
                        # sub_dict = {"correct size": correct_size, "wrong naive": wrong_naive, "wrong hard": wrong_hard}
                        # rnd_skip_dict["-".join(sorted(list(subset)))] = sub_dict
                        total_wrong_naive += wrong_naive
                        # total_wrong_hard += wrong_hard
                    # else:
                        # sub_dict = {"correct size": len(true_correct_samples), "wrong naive": 0, "wrong hard": 0}
                        # rnd_skip_dict["-".join(sorted(list(subset)))] = sub_dict
            else:
                acc = len(correct_samples) / self.test_size

            correct_samples_dict[subset] = correct_samples
            acc_dict[subset] = acc

        rnd_skip_dict["test size"] += self.test_size * total_skips
        rnd_skip_dict["naive size"] += len(naive_samples) * total_skips
        # rnd_skip_dict["hard size"] += len(hard_samples) * total_skips
        rnd_skip_dict["wrong naive"] = total_wrong_naive
        # rnd_skip_dict["wrong hard"] = total_wrong_hard
        rnd_skip_dict["skip times"] = total_skips
        rnds_acc_dict[rnd] = acc_dict
        # print(rnd_skip_dict)
        skip_dict[rnd] = rnd_skip_dict
        beta_mat_dict[rnd] = beta_mat
        model_subsets_ls_dict[rnd] = model_subsets_ls

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
        beta_mat_dict = manager.dict()
        model_subsets_ls_dict = manager.dict()
        pool = mp.Pool(10)

        workers = []
        for rnd in range(self.T):
            worker = pool.apply_async(self.sv_eval_one_rnd_rparallel,
                                      args=(rnd, rnds_acc_dict, skip_dict, beta_mat_dict, model_subsets_ls_dict),
                                      error_callback=self.throw_error)
            workers.append(worker)

        pool.close()
        pool.join()

        for worker in workers:
            worker.get()

        self.skip_dict = skip_dict.copy()

        for rnd in rnds_acc_dict:
            acc_dict = rnds_acc_dict[rnd]
            if rnd > 0:
                self.init_accs[rnd] = rnds_acc_dict[rnd - 1][frozenset(self.clients.selected_ids(rnd - 1))]
            acc_dict[frozenset()] = self.init_accs[rnd]
            SV = ShapleyValue(self.clients.selection_record[rnd], acc_dict)
            SV.calculate_svs_group_testing(beta_mat_dict[rnd], model_subsets_ls_dict[rnd],
                                           {"Z": self.Z,
                                            "Utot": rnds_acc_dict[rnd][frozenset(self.clients.selected_ids(rnd))] - self.init_accs[rnd],
                                            "epsi": self.epsi})
            self.update_ssv(SV.svs, rnd)

        print(self.init_accs)
        # print(rnds_acc_dict)
        self.cal_msv()
        # print(self.msv_dict)


if __name__ == '__main__':

    clients = Clients()
    clients.dirs = "data/mnist_cnn1/iid/0/"
    clients.load("clients.data")
    print("Directory:", clients.dirs)

    sveval = SVGroupTesting(clients, 0.25, 0.1)
    # sveval = SVPerm(clients, 0.25, 0.1)
    sveval.skip = True
    sveval.skip_compare = True
    sveval.global_skip = False
    # sveval.input_shape = (-1, 784)
    # sveval.input_shape = (-1, 3072)
    # sveval.input_shape = (-1, 3, 32, 32)
    sveval.sv_eval_mul_rnds_rparallel()
    sveval.dirs = clients.dirs
    sveval.save_stat("sv_grouptesting_skip.json")



    # for distr in ["dir0.1", "dir0.5", "dir1.0", "dir2.0", "dir5.0", "iid"]:
    #     print("Model: cifar_cnn2")
    #     for run in tqdm(range(0, 10)):
    #         clients = Clients()
    #         clients.dirs = "data/cifar_cnn2/%s/%s/" % (distr, run)
    #         clients.load("clients.data")
    #         print("Directory:", clients.dirs)
    #
    #         sveval = SVEval(clients)
    #         sveval.skip = True
    #         sveval.skip_compare = True
    #         sveval.global_skip = False
    #         # sveval.input_shape = (-1, 784)
    #         # sveval.input_shape = (-1, 3072)
    #         sveval.input_shape = (-1, 3, 32, 32)
    #         sveval.sv_eval_mul_rnds_rparallel()
    #         sveval.dirs = clients.dirs
    #         sveval.save_stat("sv_skip+.json")
    #
    # for distr in ["dir0.1", "dir0.5", "dir1.0", "dir2.0", "dir5.0", "iid"]:
    #     print("Model: cifar_logi")
    #     for run in tqdm(range(0, 10)):
    #         clients = Clients()
    #         clients.dirs = "data/cifar_logi/%s/%s/" % (distr, run)
    #         clients.load("clients.data")
    #         print("Directory:", clients.dirs)
    #
    #         sveval = SVEval(clients)
    #         sveval.skip = True
    #         sveval.skip_compare = True
    #         sveval.global_skip = False
    #         # sveval.input_shape = (-1, 784)
    #         sveval.input_shape = (-1, 3072)
    #         # sveval.input_shape = (-1, 3, 32, 32)
    #         sveval.sv_eval_mul_rnds_rparallel()
    #         sveval.dirs = clients.dirs
    #         sveval.save_stat("sv_skip+.json")
    #
    # for distr in ["dir0.1", "dir0.5", "dir1.0", "dir2.0", "dir5.0", "iid"]:
    #     print("Model: mnist_cnn1")
    #     for run in tqdm(range(0, 10)):
    #         clients = Clients()
    #         clients.dirs = "data/mnist_cnn1/%s/%s/" % (distr, run)
    #         clients.load("clients.data")
    #         print("Directory:", clients.dirs)
    #
    #         sveval = SVEval(clients)
    #         sveval.skip = True
    #         sveval.skip_compare = True
    #         sveval.global_skip = False
    #         # sveval.input_shape = (-1, 784)
    #         # sveval.input_shape = (-1, 3072)
    #         # sveval.input_shape = (-1, 3, 32, 32)
    #         sveval.sv_eval_mul_rnds_rparallel()
    #         sveval.dirs = clients.dirs
    #         sveval.save_stat("sv_skip+.json")
    #
    # for distr in ["dir0.1", "dir0.5", "dir1.0", "dir2.0", "dir5.0", "iid"]:
    #     print("Model: mnist_logi")
    #     for run in tqdm(range(0, 10)):
    #         clients = Clients()
    #         clients.dirs = "data/mnist_logi/%s/%s/" % (distr, run)
    #         clients.load("clients.data")
    #         print("Directory:", clients.dirs)
    #
    #         sveval = SVEval(clients)
    #         sveval.skip = True
    #         sveval.skip_compare = True
    #         sveval.global_skip = False
    #         sveval.input_shape = (-1, 784)
    #         # sveval.input_shape = (-1, 3072)
    #         # sveval.input_shape = (-1, 3, 32, 32)
    #         sveval.sv_eval_mul_rnds_rparallel()
    #         sveval.dirs = clients.dirs
    #         sveval.save_stat("sv_skip+.json")









