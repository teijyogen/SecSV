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
        # self.seed = 42
        self.seed = (os.getpid() * int(time.time())) % 123456789
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


def run_fl(model_func, run=0, sample_rate=1.0, device_id=0):
    args = Args()
    set_random_seed(args.seed)

    train_data, test_data, indices_train_ls, indices_test_ls = model_func.data_func(args.n_clients, alpha=args.alpha,
                                                                         sample_rate_test=sample_rate)

    clients = Clients("%s/dirt%.1fsr%.1f/%s/" % (model_func.model_name, args.alpha, sample_rate, run))
    clients.filename = "clients.data"
    clients.generate_clients(model_func.data_name, indices_train_ls, indices_test_ls)

    model = model_func()
    model.device = torch.device("cuda", device_id)
    fl = FL(model, clients, rnds=args.rnds, sel_rate=args.sel_rate)
    fl.test_data = test_data
    fl.mul_rnds_fl(clients)


def parallel_train(model_func, runs, sample_rates):
    pool = mp.Pool(10)
    workers = []
    count = 0
    for run in runs:
        for sample_rate in sample_rates:
            device_id = count % torch.cuda.device_count()
            workers.append(pool.apply_async(run_fl, args=(model_func, run, sample_rate, device_id)))
            count += 1

    pool.close()
    pool.join()

    for worker in workers:
        worker.get()


if __name__ == '__main__':

    runs = range(0, 5)
    sample_rates = [0.2, 0.4, 0.6, 0.8, 1.0]
    print("Start to train BANK_Logi")
    parallel_train(BANK_Logi, runs, sample_rates)
    print("Start to train mRNA_RNN")
    parallel_train(mRNA_RNN, runs, sample_rates)
    print("Start to train AGNEWS_Logi")
    parallel_train(AGNEWS_Logi, runs, sample_rates)
    print("Start to train MNIST_CNN")
    parallel_train(MNIST_CNN, runs, sample_rates)









