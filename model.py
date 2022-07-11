import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import *


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.optimizer = torch.optim.SGD
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.CrossEntropyLoss()
        self.test_batch_size = 1000
        self.log_interval = 10
        self.lr = 1e-3

    def model_train(self, train_data):
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        # train_loader = torch.batch
        self.to(self.device).train()

        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        for epoch in range(self.n_epochs):
            total_acc, total_count = 0, 0
            # print(scheduler.get_last_lr())
            for idx, (data, label) in enumerate(train_loader):
                data, label = data.view(self.input_shape).float().to(self.device), label.to(self.device).to(torch.long)
                # print(data)
                optimizer.zero_grad()
                output = self(data)

                loss = self.criterion(output, label)
                loss.backward()
                optimizer.step()
                total_acc += (output.argmax(1) == label).sum().item()
                total_count += label.size(0)

                # if idx % self.log_interval == 0 and idx > 0:
                #     print('| epoch {:3d} | {:5d}/{:5d} batches '
                #           '| accuracy {:8.3f}'.format(epoch, idx, len(train_loader),
                #                                       total_acc / total_count))

        self.cpu()

    def model_eval(self, test_data):
        self.to(self.device).eval()
        test_loader = DataLoader(test_data, batch_size=self.test_batch_size, shuffle=False)
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.view(self.input_shape).float().to(self.device), label.to(self.device)
                output = self(data)
                total_acc += (output.argmax(1) == label).sum().item()
                total_count += label.size(0)

            print('| accuracy {:8.3f}'.format(total_acc / total_count))

        self.cpu()


class MNIST_CNN(Model):
    model_name = "mnist_cnn"
    data_name = "MNIST"
    data_func = get_mnist_dirt

    def __init__(self, hidden=64, output=10):
        super(MNIST_CNN, self).__init__()
        self.input_shape = (-1, 1, 28, 28)
        self.conv1 = nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
        self.fc1 = nn.Linear(256, hidden)
        self.fc2 = nn.Linear(hidden, output)
        self.n_epochs = 20
        self.batch_size = 64
        self.lr = 1e-3
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv1(x)
        x = x * x
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)
        return x


# class CNN2_CIFAR(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         # print(x.shape)
#         x = F.relu(x)
#         x = self.pool(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = self.pool(x)
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class BANK_Logi(Model):
    model_name = "bank_logi"
    data_name = "BANK"
    data_func = get_bank_dirt

    def __init__(self):
        super(BANK_Logi, self).__init__()
        self.input_shape = (-1, 48)
        self.fc = nn.Linear(48, 2)
        self.batch_size = 64
        self.n_epochs = 20
        self.lr = 1e-2
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.fc(x)
        return x


class AGNEWS_Logi(Model):
    model_name = "agnews_logi"
    data_name = "AGNEWS"
    data_func = get_agnews_dirt

    def __init__(self):
        super(AGNEWS_Logi, self).__init__()
        self.optimizer = torch.optim.SGD
        self.embed_size = 300
        self.input_shape = (-1, self.embed_size)
        self.n_classes = 4
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc = nn.Linear(self.embed_size, self.n_classes)
        # self.init_weights()
        self.n_epochs = 20
        self.batch_size = 64
        self.lr = 1e-3
        self.criterion = torch.nn.CrossEntropyLoss()

    def init_weights(self):
        initrange = 0.5
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, input):
        output = self.fc(input)
        return output


class mRNA_RNN(Model):
    model_name = "mrna_rnn"
    data_name = "mRNA"
    data_func = get_mrna_dirt

    def __init__(self, seq_len=10, input_size=64, hidden=32):
        super(mRNA_RNN, self).__init__()
        self.optimizer = torch.optim.Adam
        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size = hidden
        self.input_shape = (-1, self.seq_len, self.input_size)
        self.n_classes = 2
        self.rnn = torch.nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(self.hidden_size, self.n_classes)
        # self.i2h = torch.nn.Linear(self.input_size+self.hidden_size, self.hidden_size)0.
        # self.i2o = torch.nn.Linear(self.input_size+self.hidden_size, self.n_classes)
        # self.o2o = torch.nn.Linear(self.hidden_size+self.n_classes, self.n_classes)
        self.n_epochs = 20
        self.batch_size = 64
        self.lr = 1e-3
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, input):
        output, state = self.rnn(input)
        output = self.fc(output[:, -1, :])
        # hidden = torch.zeros(input.shape[0], self.hidden_size).to(self.device)
        # for i in range(self.seq_len):
        #     input_combined = torch.cat((input[:, i, :], hidden), dim=1)
        #     hidden = self.i2h(input_combined)
        #     output = self.i2o(input_combined)
        #     output_combined = torch.cat((hidden, output), dim=1)
        #     output = self.o2o(output_combined)

        return output


