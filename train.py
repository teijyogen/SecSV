import torch
import copy
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
from model import *
from client import Clients
import torch.multiprocessing as mp
from utils import set_random_seed
import time


try:
     mp.set_start_method('spawn')
except RuntimeError:
    pass

class Args:
    def __init__(self):
        self.seed = 42
        # self.seed = (os.getpid() * int(time.time())) % 123456789
        self.alpha = 0.5
        self.n_clients = 5
        self.test_data_sample_rate = 1.0
        self.rnds = 10
        self.sel_rate = 1.0

class FL:
    def __init__(self, init_model, clients, rnds=10, sel_rate=1.0):
        self.init_model = init_model
        self.global_models = {}
        self.rnds = rnds
        self.sel_rate = sel_rate
        self.clients = clients
        self.n_clients = clients.size
        self.store_model = True
        self.need_eval = True

    @staticmethod
    def fedavg(model_ls, size_ls):
        glo_model = copy.deepcopy(model_ls[0])
        model_nb = len(model_ls)
        keys = list(glo_model.state_dict().keys())
        total = sum(size_ls)
        avg = copy.deepcopy(glo_model.state_dict())

        for key in keys:
            avg[key] = torch.mul(avg[key], size_ls[0])
            for i in range(1, model_nb):
                avg[key] += torch.mul(model_ls[i].state_dict()[key], size_ls[i])
            avg[key] = torch.div(avg[key], total)

        glo_model.load_state_dict(avg)
        return glo_model

    def error_handler(self, e):
        print('error')
        print("-->{}<--".format(e.__cause__))

    def one_rnd_fl(self, rnd):
        clients = self.clients
        sel_clients = clients.select_clients(self.sel_rate, rnd)
        if rnd == 0:
            previous_model = self.init_model
        else:
            previous_model = self.global_models[rnd-1]

        for client in sel_clients.values():
            model = copy.deepcopy(previous_model)
            train_data = client.train_data()
            model.model_train(train_data)

            if self.store_model:
                client.save_model(model, rnd)

        model_ls = clients.get_model_list(sel_clients.keys(), rnd)
        size_ls = clients.get_train_size_list(sel_clients.keys(), rnd)

        global_model = self.fedavg(model_ls, size_ls)
        if self.store_model:
            clients.save_global_model(global_model, rnd)

        if self.need_eval:
            global_model.model_eval(self.test_data)
        self.global_models[rnd] = global_model

    def mul_rnds_fl(self, clients):
        clients.clear_fl_info()
        clients.save_init_model(self.init_model)

        for rnd in tqdm(range(self.rnds)):
            self.one_rnd_fl(rnd)

        clients.save()


def run_fl(model_func, run=0, args=Args()):
    set_random_seed(args.seed)

    train_data, test_data, indices_train_ls, indices_test_ls = model_func.data_func(args.n_clients, alpha=args.alpha,
                                                                         sample_rate_test=args.test_data_sample_rate)

    clients = Clients("%s/dirt%.1fsr%.1f/%s/" % (model_func.model_name, args.alpha, args.test_data_sample_rate, run))
    clients.filename = "clients.data"
    clients.generate_clients(model_func.data_name, indices_train_ls, indices_test_ls)

    fl = FL(model_func(), clients, rnds=args.rnds, sel_rate=args.sel_rate)
    fl.test_data = test_data
    fl.mul_rnds_fl(clients)


def parallel_train(train_func, run_nb, alpha=0.5):

    pool = mp.Pool(10)
    workers = []
    for run in range(0, run_nb):
        workers.append(pool.apply_async(train_func, args=(alpha, run)))

    pool.close()
    pool.join()

    for worker in workers:
        worker.get()


if __name__ == '__main__':

    # run_fl(BANK_Logi)
    # run_fl(AGNEWS_Logi)
    args = Args()
    args.test_data_sample_rate = 0.1
    run_fl(AGNEWS_Logi, args=args)
    # run_fl(MNIST_CNN)








