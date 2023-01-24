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
        self.seed = 0
        self.alpha = 0.5
        self.n_clients = 5
        self.test_data_sample_rate = 1.0
        self.rnds = 10
        self.sel_rate = 1.0
        self.dnn_layer_nbs = (1, 3, 5, 7)
        # self.dnn_layer_nbs = range(1, 2)
        self.run = 0
        self.device_id = 0
        self.runs = range(0, 10)
        self.track_name = "main"
        self.load_clients = False
        self.test_data_usages = [0.1, 0.25, 0.5, 0.75, 1.0]
        self.client_nbs = [8]
        self.overlap = True

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
        self.test_data = self.clients.return_test_data(1.0)

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
        size_ls = clients.get_train_size_list(sel_clients.keys())

        global_model = self.fedavg(model_ls, size_ls)
        if self.store_model:
            clients.save_global_model(global_model, rnd)

        if self.need_eval:
            global_model.model_eval(self.test_data)
        self.global_models[rnd] = global_model

    def mul_rnds_fl(self):
        self.clients.clear_fl_info()
        self.clients.save_init_model(self.init_model)

        for rnd in tqdm(range(self.rnds)):
            self.one_rnd_fl(rnd)

        self.clients.save()


def run_fl(args):
    model_func = args.model_func
    args.seed = (os.getpid() * int(time.time())) % 123456789
    set_random_seed(args.seed)

    train_data, test_data, indices_train_ls, indices_test_ls = model_func.data_func(args.n_clients, alpha=args.alpha)

    if args.load_clients:
        clients = Clients("%s/%s/" % (args.load_clients_dir, args.run))
        clients.load("clients.data")
        new_dir = "%s/%s/" % (args.save_clients_dir, args.run)
        clients.set_dir(new_dir)
        clients.save()
    else:
        clients = Clients("%s/%s/" % (args.save_clients_dir, args.run))
        clients.filename = "clients.data"
        clients.generate_clients(model_func.data_name, indices_train_ls, indices_test_ls, overlap=args.overlap)

    if issubclass(model_func, DNN):
        model = model_func(hidden_layer_nb=args.hidden_layer_nb)
    else:
        model = model_func()

    model.device = torch.device("cuda", args.device_id)
    fl = FL(model, clients, rnds=args.rnds, sel_rate=args.sel_rate)
    fl.mul_rnds_fl()


def parallel_train(args):
    pool = mp.Pool(10)
    workers = []
    count = 0
    for run in args.runs:
        local_args = copy.deepcopy(args)
        local_args.device_id = count % torch.cuda.device_count()
        local_args.run = run
        workers.append(pool.apply_async(run_fl, args=(local_args,)))
        count += 1

    pool.close()
    pool.join()

    for worker in workers:
        worker.get()

def train_main_track():
    args = Args()
    # args.runs = [0, 1]

    print("Start to train BANK_Logi")
    args.model_func = BANK_Logi
    args.save_clients_dir = "main/bank_logi"
    parallel_train(args)

    print("Start to train AGNEWS_Logi")
    args.model_func = AGNEWS_Logi
    args.save_clients_dir = "main/agnews_logi"
    parallel_train(args)
    #
    print("Start to train MNIST_CNN")
    args.model_func = MNIST_CNN
    args.save_clients_dir = "main/mnist_cnn"
    parallel_train(args)

    print("Start to train mRNA_RNN")
    args.model_func = mRNA_RNN
    args.save_clients_dir = "main/mrna_rnn"
    parallel_train(args)

def train_dnn_track():
    args = Args()
    # args.runs = [0, 1]
    args.load_clients = True

    print("Start to train BANK_DNN")
    args.model_func = BANK_DNN
    for layer_nb in args.dnn_layer_nbs:
        args.hidden_layer_nb = layer_nb
        args.load_clients_dir = "main/bank_logi"
        args.save_clients_dir = "nlayer/bank_dnn/nlayer%s" % (layer_nb,)
        print(args.save_clients_dir)
        parallel_train(args)

    print("Start to train AGNEWS_DNN")
    args.model_func = AGNEWS_DNN
    for layer_nb in args.dnn_layer_nbs:
        args.hidden_layer_nb = layer_nb
        args.load_clients_dir = "main/agnews_logi"
        args.save_clients_dir = "nlayer/agnews_dnn/nlayer%s" % (layer_nb,)
        print(args.save_clients_dir)
        parallel_train(args)

    print("Start to train MNIST_DNN")
    args.model_func = MNIST_DNN
    for layer_nb in args.dnn_layer_nbs:
        args.hidden_layer_nb = layer_nb
        args.load_clients_dir = "main/mnist_cnn"
        args.save_clients_dir = "nlayer/mnist_dnn/nlayer%s" % (layer_nb,)
        print(args.save_clients_dir)
        parallel_train(args)

    print("Start to train mRNA_DNN")
    args.model_func = mRNA_DNN
    for layer_nb in args.dnn_layer_nbs:
        args.hidden_layer_nb = layer_nb
        args.load_clients_dir = "main/mrna_rnn"
        args.save_clients_dir = "nlayer/mrna_dnn/nlayer%s" % (layer_nb,)
        print(args.save_clients_dir)
        parallel_train(args)

def train_ncl_track():
    args = Args()
    # args.runs = [0, 1]

    for ncl in args.client_nbs:
        args.n_clients = ncl
        args.model_func = BANK_Logi
        args.save_clients_dir = "ncl/bank_logi/ncl%s" % (ncl,)
        print(args.save_clients_dir)
        parallel_train(args)

    for ncl in args.client_nbs:
        args.n_clients = ncl
        args.model_func = AGNEWS_Logi
        args.save_clients_dir = "ncl/agnews_logi/ncl%s" % (ncl,)
        print(args.save_clients_dir)
        parallel_train(args)

    for ncl in args.client_nbs:
        args.n_clients = ncl
        args.model_func = MNIST_CNN
        args.save_clients_dir = "ncl/mnist_cnn/ncl%s" % (ncl,)
        print(args.save_clients_dir)
        parallel_train(args)

    for ncl in args.client_nbs:
        args.n_clients = ncl
        args.model_func = mRNA_RNN
        args.save_clients_dir = "ncl/mrna_rnn/ncl%s" % (ncl,)
        print(args.save_clients_dir)
        parallel_train(args)

def construct_usage_track():
    args = Args()
    # args.runs = [0, 1]

    for usage in args.test_data_usages:
        for run in args.runs:
            clients = Clients("main/bank_logi/%s/" % (run,))
            clients.load("clients.data")
            new_dir = "usage/bank_logi/usage%s/%s/" % (usage, run)
            clients.set_data_dir(new_dir)
            clients.save()

    for usage in args.test_data_usages:
        for run in args.runs:
            clients = Clients("main/agnews_logi/%s/" % (run,))
            clients.load("clients.data")
            new_dir = "usage/agnews_logi/usage%s/%s/" % (usage, run)
            clients.set_data_dir(new_dir)
            clients.save()

    for usage in args.test_data_usages:
        for run in args.runs:
            clients = Clients("main/mnist_cnn/%s/" % (run,))
            clients.load("clients.data")
            new_dir = "usage/mnist_cnn/usage%s/%s/" % (usage, run)
            clients.set_data_dir(new_dir)
            clients.save()

    args.model_func = mRNA_DNN
    for usage in args.test_data_usages:
        for run in args.runs:
            clients = Clients("main/mrna_rnn/%s/" % (run,))
            clients.load("clients.data")
            new_dir = "usage/mrna_rnn/usage%s/%s/" % (usage, run)
            clients.set_data_dir(new_dir)
            clients.save()

def train_skip_track():
    args = Args()
    args.runs = range(100)

    print("Start to train BANK_Logi")
    args.model_func = BANK_Logi
    args.save_clients_dir = "skip/bank_logi"
    parallel_train(args)

    print("Start to train AGNEWS_Logi")
    args.model_func = AGNEWS_Logi
    args.save_clients_dir = "skip/agnews_logi"
    parallel_train(args)



if __name__ == '__main__':
    train_main_track()
    train_dnn_track()
    train_ncl_track()
    construct_usage_track()
    train_skip_track()









