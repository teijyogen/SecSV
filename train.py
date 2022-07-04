import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import copy
import random
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pickle
from model import *
from client import Clients
from shapley import make_all_subsets
from dataset import get_mnist_iid, get_mnist_dirt, get_cifar_iid, get_cifar_dirt
import torch.multiprocessing as mp

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
        self.sel_rate = 0.8
        self.n_clients = 5
        self.no_cuda = False
        self.store_model = True
        self.seed = 0
        self.input_shape = (-1, 1, 28, 28)
        self.use_cuda = not self.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model_dirs = dirs
        self.need_eval = True

        if not os.path.exists(self.model_dirs):
            os.makedirs(self.model_dirs)

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
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)

            model = self.train(model, train_loader, criterion, optimizer)
            # if self.need_eval:


            if self.store_model:
                dirs = self.model_dirs + "rnd%s/" % rnd
                if not os.path.exists(dirs):
                    os.makedirs(dirs)
                filename = "%s.pkl" % client.id
                torch.save(model, dirs+filename)
                client.models[rnd] = dirs+filename

        model_ls = clients.get_model_list(sel_clients.keys(), rnd)
        size_ls = clients.get_train_size_list(sel_clients.keys(), rnd)

        global_model = self.fedavg(model_ls, size_ls)
        if self.store_model:
            torch.save(global_model, dirs+"global.pkl")

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

    fl = FL(CNN1_MNIST(), dirs="model/mnist_cnn1/dir%.1f/%s/" % (alpha, run))

    train_data, test_data, indices_train_ls, indices_test_ls = get_mnist_dirt(fl.n_clients, alpha=alpha)

    clients = Clients()
    clients.dirs = "data/mnist_cnn1/dir%.1f/%s/" % (alpha, run)
    clients.filename = "clients.data"
    clients.generate_clients("MNIST", indices_train_ls, indices_test_ls)

    fl.mul_rnds_fl(clients)

def logi_iid_mnist(run=0):

    fl = FL(Logistic(784, 10), dirs="model/mnist_logi/iid/%s/" %(run))
    fl.input_shape = (-1, 784)

    clients = Clients()
    clients.dirs = "data/mnist_cnn1/iid/%s/" %(run)
    clients.load("clients.data")
    clients.dirs = "data/mnist_logi/iid/%s/" %(run)

    fl.mul_rnds_fl(clients)


def logi_niid_mnist(alpha=0.5, run=0):

    fl = FL(Logistic(784, 10), dirs="model/mnist_logi/dir%.1f/%s/" % (alpha, run))
    fl.input_shape = (-1, 784)

    clients = Clients()
    clients.dirs = "data/mnist_cnn1/dir%.1f/%s/" % (alpha, run)
    clients.load("clients.data")
    clients.dirs = "data/mnist_logi/dir%.1f/%s/" % (alpha, run)

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

    fl = FL(CNN2_CIFAR(), dirs="model/cifar_cnn2/dir%.1f/%s/" % (alpha, run))
    fl.input_shape = (-1, 3, 32, 32)

    train_data, test_data, indices_train_ls, indices_test_ls = get_cifar_dirt(fl.n_clients, alpha=alpha)

    clients = Clients()
    clients.dirs = "data/cifar_cnn2/dir%.1f/%s/" % (alpha, run)
    clients.filename = "clients.data"
    clients.generate_clients("CIFAR", indices_train_ls, indices_test_ls)

    fl.mul_rnds_fl(clients)

def logi_iid_cifar(run=0):

    fl = FL(Logistic(3072, 10), dirs="model/cifar_logi/iid/%s/" %(run))
    fl.input_shape = (-1, 3072)

    clients = Clients()
    clients.dirs = "data/cifar_cnn2/iid/%s/" %(run)
    clients.load("clients.data")
    clients.dirs = "data/cifar_logi/iid/%s/" %(run)

    fl.mul_rnds_fl(clients)


def logi_niid_cifar(alpha=0.5, run=0):

    fl = FL(Logistic(3072, 10), dirs="model/cifar_logi/dir%.1f/%s/" % (alpha, run))
    fl.input_shape = (-1, 3072)

    clients = Clients()
    clients.dirs = "data/cifar_cnn2/dir%.1f/%s/" % (alpha, run)
    clients.load("clients.data")
    clients.dirs = "data/cifar_logi/dir%.1f/%s/" % (alpha, run)

    fl.mul_rnds_fl(clients)



def cnn1_iid_mnist_lr(lr=0.001, run=0):

    fl = FL(CNN1_MNIST(), dirs="model/mnist_cnn1/iid_lr%.4f/%s/" % (lr, run))

    train_data, test_data, indices_train_ls, indices_test_ls = get_mnist_iid(fl.n_clients)

    clients = Clients()
    clients.dirs = "data/mnist_cnn1/iid_lr%.4f/%s/" % (lr, run)
    clients.filename = "clients.data"
    clients.generate_clients("MNIST", indices_train_ls, indices_test_ls)

    fl.mul_rnds_fl(clients)

def logi_iid_mnist_lr(lr=0.001, run=0):

    fl = FL(Logistic(784, 10), dirs="model/mnist_logi/iid_lr%.4f/%s/" % (lr, run))
    fl.input_shape = (-1, 784)

    clients = Clients()
    clients.dirs = "data/mnist_cnn1/iid_lr%.4f/%s/" % (lr, run)
    clients.load("clients.data")
    clients.dirs = "data/mnist_logi/iid_lr%.4f/%s/" % (lr, run)

    fl.mul_rnds_fl(clients)


def cnn2_iid_cifar_lr(lr=0.001, run=0):

    fl = FL(CNN2_CIFAR(), dirs="model/cifar_cnn2/iid_lr%.4f/%s/" % (lr, run))
    fl.input_shape = (-1, 3, 32, 32)

    train_data, test_data, indices_train_ls, indices_test_ls = get_cifar_iid(fl.n_clients)

    clients = Clients()
    clients.dirs = "data/cifar_cnn2/iid_lr%.4f/%s/" % (lr, run)
    clients.filename = "clients.data"
    clients.generate_clients("CIFAR", indices_train_ls, indices_test_ls)

    fl.mul_rnds_fl(clients)

def logi_iid_cifar_lr(lr=0.001, run=0):

    fl = FL(Logistic(3072, 10), dirs="model/cifar_logi/iid_lr%.4f/%s/" % (lr, run))
    fl.input_shape = (-1, 3072)

    clients = Clients()
    clients.dirs = "data/cifar_cnn2/iid_lr%.4f/%s/" % (lr, run)
    clients.load("clients.data")
    clients.dirs = "data/cifar_logi/iid_lr%.4f/%s/" % (lr, run)

    fl.mul_rnds_fl(clients)

def mnist_alpha(run):
    cnn1_iid_mnist(run)
    logi_iid_mnist(run)
    for alpha in tqdm([0.1, 0.5, 1.0, 2.0, 5.0]):
        cnn1_niid_mnist(alpha=alpha, run=run)
        logi_niid_mnist(alpha=alpha, run=run)

def cifar_alpha(run):
    cnn2_iid_cifar(run)
    logi_iid_cifar(run)
    for alpha in tqdm([0.1, 0.5, 1.0, 2.0, 5.0]):
        cnn2_niid_cifar(alpha=alpha, run=run)
        logi_niid_cifar(alpha=alpha, run=run)

def mnist_lr(run):
    for lr in tqdm([0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]):
        cnn1_iid_mnist_lr(lr=lr, run=run)
        logi_iid_mnist_lr(lr=lr, run=run)

def cifar_lr(run):
    for lr in tqdm([0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]):
        cnn2_iid_cifar_lr(lr=lr, run=run)
        logi_iid_cifar_lr(lr=lr, run=run)

if __name__ == '__main__':

    pool = mp.Pool(5)

    workers = []
    for run in range(10):
        workers.append(pool.apply_async(cifar_alpha, args=(run,)))

    for worker in workers:
        worker.get()

    pool.close()
    pool.join()


