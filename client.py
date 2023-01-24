import torch
import random
import os
import pickle
from operator import itemgetter
from torchvision import datasets, transforms
from torchtext import datasets as txt_datasets
from torchtext.data.functional import to_map_style_dataset
from dataset import Dataset, load_bank
from utils import make_dir


class Client:
    def __init__(self, id, dataset_name="", train_indices=None, test_indices=None, model_dir="model/"):
        self.id = id
        self.dataset_name = dataset_name
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.train_size = len(train_indices)
        self.test_size = len(test_indices)
        self.models = {}
        self.model_dir = model_dir

    def save_model(self, model, rnd):
        dir = self.model_dir + f"rnd{rnd}/"
        make_dir(dir)
        filename = f"{self.id}.pkl"
        torch.save(model.cpu(), dir + filename)
        self.models[rnd] = dir + filename

    def get_model(self, rnd):
        if os.path.exists(self.models[rnd]):
            return torch.load(self.models[rnd])
        else:
            new_filename = self.model_dir + f"rnd{rnd}/{self.id}.pkl"
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

    def test_data(self, usage=1.0):
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

        test_indices = self.test_indices[:int(self.test_size * usage)]

        return torch.utils.data.Subset(dataset, test_indices)


class Clients:
    def __init__(self, file_dir):
        self.data = None
        self.size = 0
        self.selection_record = {}
        self.dir = file_dir
        self.data_dir = "data/" + file_dir
        self.model_dir = "model/" + file_dir
        self.stat_dir = "stat/" + file_dir
        self.filename = ""
        self.init_model = None
        self.global_models = {}

    def return_test_data(self, test_data_usage):
        test_data = torch.utils.data.ConcatDataset(
            [self.data[id].test_data(test_data_usage) for id in self.data.keys()])
        return test_data

    def clear_fl_info(self):
        print("Clear Information Regarding FL")
        self.selection_record = {}
        for client in self.data.values():
            client.models = {}
        self.init_model = None

    def set_dir(self, file_dir):
        self.dir = file_dir
        self.data_dir = "data/" + file_dir
        self.model_dir = "model/" + file_dir
        self.stat_dir = "stat/" + file_dir
        make_dir(self.data_dir)
        make_dir(self.model_dir)
        make_dir(self.stat_dir)

        for client in self.data.values():
            client.model_dir = self.model_dir

    def set_data_dir(self, data_dir):
        self.data_dir = "data/" + data_dir
        make_dir(self.data_dir)

    def save_init_model(self, model):
        make_dir(self.model_dir)
        filename = "init.pkl"
        torch.save(model.cpu(), self.model_dir + filename)
        self.init_model = self.model_dir + filename

    def get_init_model(self):
        if os.path.exists(self.init_model):
            return torch.load(self.init_model)
        else:
            new_filename = self.model_dir + "init.pkl"
            if os.path.exists(new_filename):
                self.models[rnd] = new_filename
                return torch.load(new_filename)
            else:
                raise ValueError("Failed to get the init model.")

    def save_global_model(self, model, rnd):
        rnd_dir = self.model_dir + f"rnd{rnd}/"
        make_dir(rnd_dir)
        filename = "global.pkl"
        torch.save(model.cpu(), rnd_dir + filename)
        self.global_models[rnd] = rnd_dir + filename

    def get_global_model(self, rnd):
        if os.path.exists(self.global_models[rnd]):
            return torch.load(self.global_models[rnd])
        else:
            new_filename = self.model_dir + f"rnd{rnd}/global.pkl"
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
        ids = random.sample(list(self.data.keys()), k=int(self.size * rate))
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

    def get_model(self, cid, rnd):
        model = self.data[cid].get_model(rnd)
        return model

    def get_model_list(self, ids, rnd):
        model_ls = []
        for cid in ids:
            model = self.get_model(cid, rnd)
            model_ls.append(model)
        return model_ls

    def get_train_size_list(self, ids):
        size_list = []
        for cid in ids:
            size_list.append(self.data[cid].train_size)

        return size_list

    def load(self, filename):
        f = open(self.data_dir + filename, "rb")
        print("Load clients data")
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)
        self.filename = filename

    def save(self):
        make_dir(self.data_dir)
        f = open(self.data_dir + self.filename, "wb")
        pickle.dump(self.__dict__, f)
        f.close()

    def generate_clients(self, dataset_name, indices_train_ls, indices_test_ls, overlap=False):
        data_dir = self.data_dir
        filename = self.filename
        make_dir(data_dir)

        if os.path.exists(data_dir + filename):
            print("Clients data exists")
            if not overlap:
                self.load(filename)
                return

        n_clients = len(indices_train_ls)

        clients = {}
        for i in range(n_clients):
            id = str(i + 1)
            indices_train = indices_train_ls[i].astype(int)
            indices_test = indices_test_ls[i].astype(int)

            client = Client(id, dataset_name, indices_train, indices_test, self.model_dir)
            clients[id] = client

        self.data = clients
        self.size = n_clients
        self.save()
