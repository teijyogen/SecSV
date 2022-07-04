import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms

# class CNN1_MNIST(nn.Module):
#     def __init__(self):
#         super(CNN1_MNIST, self).__init__()
#         self.conv1 = nn.Conv2d(1, 5, kernel_size=5, stride=2)
#         self.fc1 = nn.Linear(845, 100)
#         self.fc2 = nn.Linear(100, 10)
#
#     def forward(self, x):
#         x = nn.functional.pad(x, pad=(1, 0, 1, 0))
#         x = self.conv1(x)
#         x = x.view(-1, 845)
#         x = x * x
#         x = self.fc1(x)
#         x = x * x
#         x = self.fc2(x)
#         return x


class CNN1_MNIST(nn.Module):
    def __init__(self, hidden=64, output=10):
        super(CNN1_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
        self.fc1 = nn.Linear(256, hidden)
        self.fc2 = nn.Linear(hidden, output)

    def forward(self, x):
        x = self.conv1(x)
        x = x * x
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)
        return x


# class CNN2_MNIST(nn.Module):
#     def __init__(self):
#         super(CNN2_MNIST, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0, stride=1)
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool2d((2, 2))
#         self.conv2 = nn.Conv2d(16, 16, kernel_size=5, padding=0, stride=1)
#         self.fc1 = nn.Linear(256, 100)
#         self.fc2 = nn.Linear(100, 10)
#
#     def forward(self, x):
#         x = self.conv1(x).view(-1, 16, 24, 24)
#         x = self.relu(x)
#         x = self.maxpool(x).view(-1, 16, 12, 12)
#         x = self.conv2(x).view(-1, 16, 64)
#         x = self.relu(x)
#         x = self.maxpool(x).view(-1, 256)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#
#         return x

# class CNN2_CIFAR(nn.Module):
#     def __init__(self):
#         super(CNN2_CIFAR, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
#         self.pool = nn.AvgPool2d((2, 2))
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
#         self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
#         self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
#         self.conv6 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
#         self.conv7 = nn.Conv2d(64, 16, kernel_size=1, stride=1)
#         self.fc = nn.Linear(1024, 10)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.pool(x)
#         x = self.conv3(x)
#         x = self.relu(x)
#         x = self.conv4(x)
#         x = self.relu(x)
#         x = self.pool(x)
#         x = self.conv5(x)
#         x = self.relu(x)
#         x = self.conv6(x)
#         x = self.relu(x)
#         x = self.conv7(x)
#         x = self.relu(x).view(-1, 1024)
#         x = self.fc(x)
#
#         return x

class CNN2_CIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class Logistic(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Logistic, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        # self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        # out = self.sig(out)
        return out

if __name__ == '__main__':
    input = torch.rand((3, 32, 32)).view(-1, 3, 32, 32)
    model = CNN2_CIFAR()
    params = model.state_dict()
    weight_ls = params["conv1.weight"]
    bias_ls = params["conv1.bias"]
    count = 0
    for weight, bias in zip(weight_ls, bias_ls):
        count += 1

    print(count)
    # output = model(input)
