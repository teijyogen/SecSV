import torch
import random
import os
import pickle
from operator import itemgetter
from torchvision import datasets, transforms
from torchtext import datasets as txt_datasets
from torchtext.data.functional import to_map_style_dataset
from dataset import Dataset, load_bank
from utils import make_dirs

class Client:
    def __init__(self, id, dataset_name="", train_indices=None, test_indices=None, model_dirs="model/"):
        self.id = id
        self.dataset_name = dataset_name
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.train_size = len(train_indices)
        self.test_size = len(test_indices)
        self.models = {}
        self.model_dirs = model_dirs

    def save_model(self, model, rnd):
        dirs = self.model_dirs + f"rnd{rnd}/"
        make_dirs(dirs)
        filename = f"{self.id}.pkl"
        torch.save(model.cpu(), dirs + filename)
        self.models[rnd] = dirs + filename

    def get_model(self, rnd):
        if os.path.exists(self.models[rnd]):
            return torch.load(self.models[rnd])
        else:
            new_filename = self.model_dirs + f"rnd{rnd}/{self.id}.pkl"
            if os.path.exists(new_filename):
                self.models[rnd] = new_filename
                return torch.load(new_filename)
            else:
                raise ValueError(f"Failed to get client {self.id}'s model for round {rnd}.")

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
        elif self.dataset_name == "AGNEWS":
            data, labels = torch.load('data/AGNEWS/train_data.pt')
            dataset = Dataset(data, labels)
        elif self.dataset_name == "mRNA":
            data, labels = torch.load('data/mRNA/train_data.pt')
            dataset = Dataset(data, labels)
        elif self.dataset_name == "BANK":
            dataset, _ = load_bank()
        else:
            raise ValueError(f"{self.dataset_name} dataset is not defined")

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
        elif self.dataset_name == "AGNEWS":
            data, labels = torch.load('data/AGNEWS/test_data.pt')
            dataset = Dataset(data, labels)
        elif self.dataset_name == "mRNA":
            data, labels = torch.load('data/mRNA/test_data.pt')
            dataset = Dataset(data, labels)
        elif self.dataset_name == "BANK":
            _, dataset = load_bank()
        else:
            raise ValueError(f"{self.dataset_name} dataset is not defined")

        return torch.utils.data.Subset(dataset, self.test_indices)

class Clients:
    def __init__(self, dirs):
        self.data = None
        self.size = 0
        self.selection_record = {}
        self.data_dirs = "data/" + dirs
        self.model_dirs = "model/" + dirs
        self.filename = ""
        self.init_model = None
        self.global_models = {}

    def clear_fl_info(self):
        print("Clear Information Regarding FL")
        self.selection_record = {}
        for client in self.data.values():
            client.models = {}
        self.init_model = None

    def set_model_dirs(self, dirs):
        make_dirs(dirs)
        self.model_dirs = dirs
        for client in self.data:
            client.model_dirs = dirs

    def save_init_model(self, model):
        dirs = self.model_dirs
        make_dirs(dirs)
        filename = "init.pkl"
        torch.save(model.cpu(), dirs + filename)
        self.init_model = dirs + filename

    def get_init_model(self):
        if os.path.exists(self.init_model):
            return torch.load(self.init_model)
        else:
            new_filename = self.model_dirs + "init.pkl"
            if os.path.exists(new_filename):
                self.models[rnd] = new_filename
                return torch.load(new_filename)
            else:
                raise ValueError("Failed to get the init model.")

    def save_global_model(self, model, rnd):
        dirs = self.model_dirs + f"rnd{rnd}/"
        make_dirs(dirs)
        filename = "global.pkl"
        torch.save(model.cpu(), dirs + filename)
        self.global_models[rnd] = dirs + filename

    def get_global_model(self, rnd):
        if os.path.exists(self.global_models[rnd]):
            return torch.load(self.global_models[rnd])
        else:
            new_filename = self.model_dirs + f"rnd{rnd}/global.pkl"
            if os.path.exists(new_filename):
                self.models[rnd] = new_filename
                return torch.load(new_filename)
            else:
                raise ValueError(f"Failed to get the global model for round {rnd}.")

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
            model = self.data[id].get_model(rnd)
            model_ls.append(model)
        return model_ls

    def get_train_size_list(self, ids, rnd):
        size_list = []
        for id in ids:
            size_list.append(self.data[id].train_size)

        return size_list

    def load(self, filename):
        f = open(self.data_dirs + filename, "rb")
        print("Load clients data")
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)
        self.filename = filename

    def save(self):
        if not os.path.exists(self.data_dirs):
            os.makedirs(self.data_dirs)
        f = open(self.data_dirs+self.filename, "wb")
        pickle.dump(self.__dict__, f)
        f.close()

    def generate_clients(self, dataset_name, indices_train_ls, indices_test_ls, overlap=False):
        dirs = self.data_dirs
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

            client = Client(id, dataset_name, indices_train, indices_test, self.model_dirs)
            clients[id] = client

        self.data = clients
        self.size = n_clients
        self.save()