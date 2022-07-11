import pandas as pd
import torch
import numpy as np
from torchvision import datasets as vis_datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.targets.shape[0]

def load_csv_data(filename):
    f = open(filename, "rb")
    data = np.loadtxt(f, delimiter=",", skiprows=1)
    x = data[:, :-1]
    x = torch.tensor(x)
    y = data[:, -1]
    y = torch.tensor(y)
    data_set = Dataset(x, y)
    f.close()
    return data_set

def load_bank():
    train_set = load_csv_data("data/BANK/train_data.csv")
    test_set = load_csv_data("data/BANK/test_data.csv")
    return train_set, test_set


def extr_noniid_dirt(train_dataset, test_dataset, num_users, num_classes, alpha=0.5, sample_rate_test=1.0):

    num_imgs_train_total, num_imgs_test_total = len(train_dataset), len(test_dataset)
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idxs, idxs_test = np.arange(num_imgs_train_total), np.arange(num_imgs_test_total)
    labels, labels_test = np.array(train_dataset.targets), np.array(test_dataset.targets)

    labels_df, labels_test_df = pd.DataFrame(labels), pd.DataFrame(labels_test)
    num_imgs_perc_train = labels_df[0].value_counts().sort_index().array
    num_imgs_perc_test = labels_test_df[0].value_counts().sort_index().array

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]
    #print(idxs_labels_test[1, :])

    distribution = np.random.dirichlet(np.repeat(alpha, num_users), size=num_classes)


    # divide and assign
    idxs_classes = []
    for j in range(num_classes):
        idxs_classj = list(idxs[num_imgs_perc_train[:j].sum():num_imgs_perc_train[:j+1].sum()])
        idxs_classes.append(idxs_classj)

    idxs_classes_test = []
    for j in range(num_classes):
        idxs_classj_test = list(idxs_test[num_imgs_perc_test[:j].sum():num_imgs_perc_test[:j+1].sum()])
        idxs_classes_test.append(idxs_classj_test)

    for j in range(num_classes):
        remained_test_size = int(np.round(num_imgs_perc_test[j] * sample_rate_test))
        for i in range(num_users):
            if i == num_users - 1:
                rand_set = idxs_classes[j]
                rand_set_test = np.random.choice(idxs_classes_test[j], remained_test_size, replace=False)
            else:
                rand_set = np.random.choice(idxs_classes[j], int(distribution[j][i]*num_imgs_perc_train[j]), replace=False)
                rand_set_test = np.random.choice(idxs_classes_test[j],
                                                 int(distribution[j][i] * num_imgs_perc_test[j] * sample_rate_test),
                                                 replace=False)

            remained_test_size -= len(rand_set_test)
            idxs_classes[j] = list(set(idxs_classes[j]) - set(rand_set))
            dict_users_train[i] = np.concatenate((dict_users_train[i], rand_set), axis=0)

            idxs_classes_test[j] = list(set(idxs_classes_test[j]) - set(rand_set_test))
            dict_users_test[i] = np.concatenate((dict_users_test[i], rand_set_test), axis=0)

    test_sizes = np.array([len(dict_users_test[i]) for i in range(num_users)])
    print(test_sizes)
    print(test_sizes.sum())

    return dict_users_train, dict_users_test

# def get_mnist_iid(num_users):
#     apply_transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))])
#     train_data = vis_datasets.MNIST('data', train=True, download=True, transform=apply_transform)
#     test_data = vis_datasets.MNIST('data', train=False, download=True, transform=apply_transform)
#
#     indices_train = np.array([i for i in range(len(train_data))])
#     indices_test = np.array([i for i in range(len(test_data))])
#     np.random.shuffle(indices_train)
#     np.random.shuffle(indices_test)
#
#     indices_train_ls = np.array_split(indices_train, num_users)
#     indices_test_ls = np.array_split(indices_test, num_users)
#
#     return train_data, test_data, indices_train_ls, indices_test_ls
#
# def get_cifar_iid(num_users):
#     apply_transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     train_data = vis_datasets.CIFAR10('data', train=True, download=True, transform=apply_transform)
#     test_data = vis_datasets.CIFAR10('data', train=False, download=True, transform=apply_transform)
#
#     indices_train = np.array([i for i in range(len(train_data))])
#     indices_test = np.array([i for i in range(len(test_data))])
#     np.random.shuffle(indices_train)
#     np.random.shuffle(indices_test)
#
#     indices_train_ls = np.array_split(indices_train, num_users)
#     indices_test_ls = np.array_split(indices_test, num_users)
#
#     return train_data, test_data, indices_train_ls, indices_test_ls

def get_mnist_dirt(num_users, alpha=0.5, sample_rate_test=1.0):
    apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
    train_data = vis_datasets.MNIST('data', train=True, download=True, transform=apply_transform)
    test_data = vis_datasets.MNIST('data', train=False, download=True, transform=apply_transform)

    indices_train_ls, indices_test_ls = extr_noniid_dirt(train_data, test_data, num_users, 10, alpha, sample_rate_test=sample_rate_test)
    return train_data, test_data, indices_train_ls, indices_test_ls

# def get_cifar_dirt(num_users, alpha=0.5, sample_rate_test=1.0):
#     apply_transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     train_data = vis_datasets.CIFAR10('data', train=True, download=True, transform=apply_transform)
#
#     test_data = vis_datasets.CIFAR10('data', train=False, download=True, transform=apply_transform)
#
#     indices_train_ls, indices_test_ls = extr_noniid_dirt(train_data, test_data, num_users, 10, alpha, sample_rate_test=sample_rate_test)
#     return train_data, test_data, indices_train_ls, indices_test_ls

def get_agnews_dirt(num_users, alpha=0.5, sample_rate_test=1.0):
    train_data, test_data = torch.load('data/AGNEWS/train_data.pt'), torch.load('data/AGNEWS/test_data.pt')
    train_data, test_data = Dataset(train_data[0], train_data[1]), Dataset(test_data[0], test_data[1])

    indices_train_ls, indices_test_ls = extr_noniid_dirt(train_data, test_data, num_users, 4, alpha,
                                                         sample_rate_test=sample_rate_test)
    return train_data, test_data, indices_train_ls, indices_test_ls

def get_mrna_dirt(num_users, alpha=0.5, sample_rate_test=1.0):
    train_data, test_data = torch.load('data/mRNA/train_data.pt'), torch.load('data/mRNA/test_data.pt')
    train_data, test_data = Dataset(train_data[0], train_data[1]), Dataset(test_data[0], test_data[1])
    
    indices_train_ls, indices_test_ls = extr_noniid_dirt(train_data, test_data, num_users, 2, alpha,
                                                         sample_rate_test=sample_rate_test)
    return train_data, test_data, indices_train_ls, indices_test_ls


def get_bank_dirt(num_users, alpha=0.5, sample_rate_test=1.0):
    train_data, test_data = load_bank()
    indices_train_ls, indices_test_ls = extr_noniid_dirt(train_data, test_data, num_users, 2, alpha,
                                                         sample_rate_test=sample_rate_test)
    return train_data, test_data, indices_train_ls, indices_test_ls











