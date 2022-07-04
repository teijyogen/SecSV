import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import copy
import random
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import pickle
from model import *
from client import Clients
from shapley import make_all_subsets
from dataset import get_mnist_iid, get_mnist_dirt, get_cifar_iid, get_cifar_dirt
import torch.multiprocessing as mp
from utils import make_dir
import time

try:
     mp.set_start_method('spawn')
except RuntimeError:
    pass

torch.cuda.manual_seed(0)
class FL:
    def __init__(self, init_model, dirs):
        self.init_model = init_model
        self.global_models = {}
        self.batch_size = 64
        self.rnds = 10
        self.lr = 0.001
        self.n_epochs = 5
        self.sel_rate = 1.0
        self.n_clients = 5
        self.no_cuda = False
        self.store_model = True
        self.seed = 0
        self.input_shape = (-1, 1, 28, 28)
        self.use_cuda = not self.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model_dirs = dirs
        self.need_eval = True
        make_dir(self.model_dirs)

    def train(self, model, train_loader, criterion, optimizer):
    # model in training mode
        model.train().to(self.device)
        for epoch in range(self.n_epochs):

            train_loss = 0.0
            for data, target in train_loader:
                data = data.to(self.device).reshape(self.input_shape)
                target = target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # calculate average losses
            train_loss = train_loss / len(train_loader)

            # print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))

        # model in evaluation mode
        model.eval()
        return model.cpu()

    def eval(self, model):
        test_loader = self.test_loader
        model = model.to(self.device)
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))

        model.eval().to(self.device)
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
        model.cpu()
        print("Accuracy:", len(correct_ids) / total_size)
        return set(correct_ids)

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

    def one_rnd_fl(self, clients, rnd):
        sel_clients = clients.select_clients(self.sel_rate, rnd)
        if rnd == 0:
            previous_model = self.init_model
        else:
            previous_model = self.global_models[rnd-1]

        for client in sel_clients.values():
            train_loader = torch.utils.data.DataLoader(client.train_data(), batch_size=self.batch_size, shuffle=True)
            model = copy.deepcopy(previous_model)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)

            model = self.train(model, train_loader, criterion, optimizer)
            # if self.need_eval:


            if self.store_model:
                dirs = self.model_dirs + "rnd%s/" % rnd
                make_dir(dirs)
                filename = "%s.pkl" % client.id
                torch.save(model, dirs+filename)
                client.models[rnd] = dirs+filename

        model_ls = clients.get_model_list(sel_clients.keys(), rnd)
        size_ls = clients.get_train_size_list(sel_clients.keys(), rnd)

        global_model = self.fedavg(model_ls, size_ls)
        if self.store_model:
            torch.save(global_model, dirs+"global.pkl")

        self.eval(global_model)
        self.global_models[rnd] = global_model

    def mul_rnds_fl(self, clients):
        clients.clear_fl_info()
        torch.save(self.init_model, self.model_dirs + "init.pkl")
        clients.init_model = self.model_dirs + "init.pkl"

        self.init_model.to(self.device)

        for rnd in tqdm(range(self.rnds)):
            self.one_rnd_fl(clients, rnd)

        clients.save()

def cnn1_iid_mnist(run=0):

    fl = FL(CNN1_MNIST(), dirs="model/mnist_cnn1/iid/%s/" %(run))

    train_data, test_data, indices_train_ls, indices_test_ls = get_mnist_iid(fl.n_clients)

    clients = Clients()
    clients.dirs = "data/mnist_cnn1/iid/%s/" %(run)
    clients.filename = "clients.data"
    clients.generate_clients("MNIST", indices_train_ls, indices_test_ls)

    fl.mul_rnds_fl(clients)

def cnn1_niid_mnist(alpha=0.5, run=0):
    torch.manual_seed((os.getpid() * int(time.time())) % 123456789)
    torch.cuda.manual_seed_all((os.getpid() * int(time.time())) % 123456789)
    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    fl = FL(CNN1_MNIST(), dirs="model/mnist_cnn1/dir%.1f/%s/" % (alpha, run))

    train_data, test_data, indices_train_ls, indices_test_ls = get_mnist_dirt(fl.n_clients, alpha=alpha)

    fl.test_loader = torch.utils.data.DataLoader(test_data, batch_size=1024, shuffle=False)
    clients = Clients()
    clients.dirs = "data/mnist_cnn1/dir%.1f/%s/" % (alpha, run)
    clients.filename = "clients.data"
    clients.generate_clients("MNIST", indices_train_ls, indices_test_ls)

    fl.mul_rnds_fl(clients)

def cnn2_iid_cifar(run=0):

    fl = FL(CNN2_CIFAR(), dirs="model/cifar_cnn2/iid/%s/" %(run))
    fl.input_shape = (-1, 3, 32, 32)

    train_data, test_data, indices_train_ls, indices_test_ls = get_cifar_iid(fl.n_clients)

    clients = Clients()
    clients.dirs = "data/cifar_cnn2/iid/%s/" %(run)
    clients.filename = "clients.data"
    clients.generate_clients("CIFAR", indices_train_ls, indices_test_ls)

    fl.mul_rnds_fl(clients)

def cnn2_niid_cifar(alpha=0.5, run=0):
    torch.manual_seed((os.getpid() * int(time.time())) % 123456789)
    torch.cuda.manual_seed_all((os.getpid() * int(time.time())) % 123456789)
    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    fl = FL(CNN2_CIFAR(), dirs="model/cifar_cnn2/dir%.1f/%s/" % (alpha, run))
    fl.input_shape = (-1, 3, 32, 32)
    fl.batch_size = 4

    train_data, test_data, indices_train_ls, indices_test_ls = get_cifar_dirt(fl.n_clients, alpha=alpha)

    fl.test_loader = torch.utils.data.DataLoader(test_data, batch_size=1024, shuffle=False)
    clients = Clients()
    clients.dirs = "data/cifar_cnn2/dir%.1f/%s/" % (alpha, run)
    clients.filename = "clients.data"
    clients.generate_clients("CIFAR", indices_train_ls, indices_test_ls)

    fl.mul_rnds_fl(clients)

if __name__ == '__main__':
    # linear_niid_mnist()
    # pool = mp.Pool(5)
    #
    # workers = []
    # for run in range(10):
    #     workers.append(pool.apply_async(cifar_alpha, args=(run,)))
    #
    # for worker in workers:
    #     worker.get()
    #
    # pool.close()
    # pool.join()

    # pool = mp.Pool(5)
    # workers = []
    # for run in range(0, 5):
    #     workers.append(pool.apply_async(cnn1_niid_mnist, args=(0.5, run)))
    #
    # for run in range(0, 5):
    #     workers.append(pool.apply_async(cnn2_niid_cifar, args=(0.5, run)))
    #
    # pool.close()
    # pool.join()
    #
    # for worker in workers:
    #     worker.get()

    cnn1_niid_mnist(run=5)




