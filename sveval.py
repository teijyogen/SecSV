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
        self.indices = np.arange(len(self.test_data))
        self.batch_size = 1024
        self.T = len(clients.selection_record)
        self.init_accs = {}
        self.skip = False
        self.skip_compare = False
        self.ssv_dict = {}
        self.msv_dict = {}
        self.dirs = clients.data_dirs
        self.test_size = len(self.test_data)
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
        if self.skip:
            data = {"ssv": self.ssv_dict, "msv": self.msv_dict, "skip": self.skip_dict}
        else:
            data = {"ssv": self.ssv_dict, "msv": self.msv_dict}
        with open(self.dirs+filename, "w") as f:
            json.dump(data, f, indent=4)

        f.close()

    def eval(self, model, test_data):

        test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        if len(test_loader) == 0:
            return set()
        indices = test_data.indices

        model = model.to(self.device)
        model.eval()
        total_size = 0
        correct_ids_ls = []

        for data, target in test_loader:
            data = data.reshape(model.input_shape).float().to(self.device)
            target = target.to(self.device)
            output = model(data)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # compare predictions to true label
            correct_ids = indices[np.where(pred.eq(target.data.view_as(pred)).cpu())[0] + total_size]
            correct_ids_ls.append(correct_ids)
            total_size += len(target)

        correct_ids = np.concatenate(correct_ids_ls).tolist()

        return set(correct_ids)

    def throw_error(self, e):
        raise e

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

    def sv_eval_one_rnd_rparallel(self, rnd, rnds_acc_dict, rnds_skip_dict):
        test_data = torch.utils.data.Subset(self.test_data, self.indices)
        clients = self.clients
        acc_dict = {}
        skip_dict = {}
        correct_samples_dict = {}
        sel_clients = clients.selected_clients(rnd)
        all_subsets = make_all_subsets(list(sel_clients.keys()))
        local_subsets = [subset for subset in all_subsets if len(subset) == 1]
        aggr_subsets = [subset for subset in all_subsets if len(subset) > 1]

        for subset in local_subsets:
            id = list(subset)[0]
            client = clients.find_client(id)
            local_model = client.get_model(rnd)
            correct_samples = self.eval(local_model, test_data)
            acc_dict[subset] = len(correct_samples) / self.test_size
            correct_samples_dict[subset] = correct_samples

        for subset in aggr_subsets:
            naive_samples = set([])
            if self.skip:
                naive_samples = self.find_skippable(subset, correct_samples_dict)
                indices = np.array(sorted(list(set(self.indices) - naive_samples)))
                test_data = torch.utils.data.Subset(self.test_data, indices)

            model_ls = clients.get_model_list(list(subset), rnd)
            size_ls = clients.get_train_size_list(list(subset), rnd)
            aggr_model = FL.fedavg(model_ls, size_ls)
            correct_samples = self.eval(aggr_model, test_data)
            correct_samples_dict[subset] = correct_samples.union(naive_samples)
            acc = (len(naive_samples) + len(correct_samples)) / self.test_size
            acc_dict[subset] = acc

            if self.skip:
                subset_skip_dict = {}
                test_data = torch.utils.data.Subset(self.test_data, self.indices)
                all_correct_samples = self.eval(aggr_model, test_data)
                subset_skip_dict["total corrects"] = len(all_correct_samples)
                subset_skip_dict["corrects"] = len(correct_samples)
                subset_skip_dict["naive"] = len(naive_samples)
                subset_skip_dict["wrong naive"] = list(naive_samples.difference(all_correct_samples))

                skip_dict[str(list(subset))] = subset_skip_dict

        rnds_acc_dict[rnd] = acc_dict
        rnds_skip_dict[rnd] = skip_dict

    def sv_eval_mul_rnds_rparallel(self):
        init_model = self.clients.get_init_model()
        test_data = torch.utils.data.Subset(self.test_data, self.indices)
        correct_samples = self.eval(init_model, test_data)
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
            SV.calculate_svs()
            self.update_ssv(SV.svs, rnd)

        self.cal_msv()


class SVPerm(SVEval):
    def __init__(self, clients, epsi, sigma):
        SVEval.__init__(self, clients)
        self.epsi = epsi
        self.sigma = sigma
        self.N = len(clients.selection_record[0])
        self.esti_times = math.ceil(2 * self.N / (epsi ** 2) * math.log(2 * self.N / sigma, 2))

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
            return (1 + u) * math.log(1 + u, 2) - u
        self.esti_times = math.ceil(8 * math.log(N * (N -1) / (2 * sigma), 2) / ((1 - qtot ** 2) * h(epsi / (Z * N ** 0.5 * (1 - qtot ** 2)))))

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
        test_data = torch.utils.data.Subset(self.test_data, list(self.idx_set))
        correct_samples = self.eval(torch.load(init_model), test_data)
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

        for rnd in range(self.T):
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

    dir = "agnews_logi/dirt0.5sr0.2/0/"
    clients = Clients(dir)
    clients.load("clients.data")
    sveval = SVEval(clients)
    sveval.skip = True
    # sveval.skip_compare = True
    sveval.sv_eval_mul_rnds_rparallel()
    sveval.save_stat("sv_skip.json")


    # sveval = SVGroupTesting(clients, 0.25, 0.1)
    # print(sveval.esti_times)
    # sveval = SVPerm(clients, 0.25, 0.1)
    # sveval.skip = True
    # sveval.skip_compare = True
    # sveval.global_skip = False
    # sveval.input_shape = (-1, 784)
    # sveval.input_shape = (-1, 3072)
    # sveval.input_shape = (-1, 3, 32, 32)
    # sveval.sv_eval_mul_rnds_rparallel()
    # sveval.dirs = clients.dirs
    # sveval.save_stat("sv_grouptesting_skip.json")



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









