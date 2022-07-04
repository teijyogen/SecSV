import torch
import random
import os
import pickle
from operator import itemgetter
from torchvision import datasets, transforms

class Client:
    def __init__(self, id, dataset_name="", train_indices=None, test_indices=None):
        self.id = id
        self.dataset_name = dataset_name
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.train_size = len(train_indices)
        self.test_size = len(test_indices)
        self.models = {}

    def train_data(self):
        if self.dataset_name == "MNIST":
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
            dataset = datasets.MNIST('data', train=True, download=True, transform=apply_transform)
        elif self.dataset_name == "CIFAR":
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset = datasets.CIFAR10('data', train=True, download=True, transform=apply_transform)

        return torch.utils.data.Subset(dataset, self.train_indices)

    def test_data(self):
        if self.dataset_name == "MNIST":
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
            dataset = datasets.MNIST('data', train=False, download=True, transform=apply_transform)
        elif self.dataset_name == "CIFAR":
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset = datasets.CIFAR10('data', train=False, download=True, transform=apply_transform)

        return torch.utils.data.Subset(dataset, self.test_indices)

class Clients:
    def __init__(self):
        self.data = None
        self.size = 0
        self.selection_record = {}
        self.dirs = "data/"
        self.filename = ""
        self.init_model = None

    def clear_fl_info(self):
        print("Clear Information Regarding FL")
        self.selection_record = {}
        for client in self.data.values():
            client.models = {}
        self.init_model = None

    def add_client(self, client):
        self.data[client.id] = client
        self.size += 1

    def find_client(self, id):
        return self.data[id]

    def find_clients(self, ids):
        return dict(zip(ids, itemgetter(*ids)(self.data)))

    def select_client_id(self):
        id = random.choice(list(self.data.keys()))
        return id

    def select_client(self):
        id = self.select_client_id()
        return self.find_client(id)

    def select_clients_ids(self, rate, rnd):
        ids = random.sample(list(self.data.keys()), k=int(self.size*rate))
        self.selection_record[rnd] = ids
        return ids

    def select_clients(self, rate, rnd):
        sel_ids = self.select_clients_ids(rate, rnd)
        return self.find_clients(sel_ids)

    def selected_ids(self, rnd):
        return self.selection_record[rnd]

    def selected_clients(self, rnd):
        ids = self.selected_ids(rnd)
        return self.find_clients(ids)

    def get_model_list(self, ids, rnd):
        model_ls = []
        for id in ids:
            model = torch.load(self.data[id].models[rnd])
            model_ls.append(model)

        return model_ls

    def get_train_size_list(self, ids, rnd):
        size_list = []
        for id in ids:
            size_list.append(self.data[id].train_size)

        return size_list

    def load(self, filename):
        f = open(self.dirs + filename, "rb")
        print("Load clients data")
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)
        self.filename = filename

    def save(self):
        if not os.path.exists(self.dirs):
            os.makedirs(self.dirs)
        f = open(self.dirs+self.filename, "wb")
        pickle.dump(self.__dict__, f)
        f.close()

    def generate_clients(self, dataset_name, indices_train_ls, indices_test_ls, overlap=False):
        dirs = self.dirs
        filename = self.filename

        if not os.path.exists(dirs):
            os.makedirs(dirs)

        if os.path.exists(dirs+filename):
            print("Clients data exists")
            if not overlap:
                self.load(filename)
                return

        n_clients = len(indices_train_ls)

        clients = {}
        for i in range(n_clients):
            id = str(i+1)
            indices_train = indices_train_ls[i].astype(int)
            indices_test = indices_test_ls[i].astype(int)

            client = Client(id, dataset_name, indices_train, indices_test)
            clients[id] = client

        self.data = clients
        self.size = n_clients
        self.save()