import torch
import torch.nn as nn
from torch.autograd.variable import Variable



class FCNet(nn.Module):

    def __init__(self, in_features, out_features=16):
        super(FCNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=256, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=256, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU()
        )
        self.policy = nn.Linear(in_features=256, out_features=out_features)
        self.value = nn.Linear(in_features=256, out_features=1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.net(x)
        policy_result = self.policy(x)
        value_result = self.value(x)
        return self.softmax(policy_result), value_result


class ConvNet(nn.Module):
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.affine(x)
        policy_result = self.policy(self.activation(x))
        value_result = self.value(self.activation(x))
        return self.softmax(policy_result), value_result

    def conv_layers(self, x):
        x = self.maxpool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.maxpool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.maxpool3(self.relu3(self.bn2(self.conv3(x))))
        return x

    def __init__(self, out_features=16):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=36, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=36)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=36, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=36)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.affine = nn.Linear(in_features=2304, out_features=128)
        self.policy = nn.Linear(in_features=128, out_features=out_features)
        self.value = nn.Linear(in_features=128, out_features=1)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.saved_actions = []
        self.rewards = []
