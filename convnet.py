import torch
from torch import nn, optim
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, num_classes: int):
        super(ConvNet, self).__init__()
        self._num_inp_channels = 3
        self._num_conv_channels = [64, 128, 256, 512]
        self._kernel_sizes = [3, 3, 3, 3]
        self._pool_padding = 2
        self._pool_stride = 2
        self.num_linear_neurons = [512, 256]

        self.pool = nn.MaxPool2d(self._pool_padding, self._pool_stride)

        self.conv1 = nn.Conv2d(self._num_inp_channels,
                               self._num_conv_channels[0],
                               self._kernel_sizes[0])
        self.bn1 = nn.BatchNorm2d(self._num_conv_channels[0])

        self.conv2 = nn.Conv2d(self._num_conv_channels[0],
                               self._num_conv_channels[1],
                               self._kernel_sizes[1])
        self.bn2 = nn.BatchNorm2d(self._num_conv_channels[1])

        self.conv3 = nn.Conv2d(self._num_conv_channels[1],
                               self._num_conv_channels[2],
                               self._kernel_sizes[2])
        self.bn3 = nn.BatchNorm2d(self._num_conv_channels[2])

        self.conv4 = nn.Conv2d(self._num_conv_channels[2],
                               self._num_conv_channels[3],
                               self._kernel_sizes[3])
        self.bn4 = nn.BatchNorm2d(self._num_conv_channels[3])

        self._resulting_size = self._num_conv_channels[-1] * 4 * 4

        self.fc1 = nn.Linear(self._resulting_size,
                             self.num_linear_neurons[0])
        self.fc2 = nn.Linear(self.num_linear_neurons[0],
                             self.num_linear_neurons[1])
        self.fc3 = nn.Linear(self.num_linear_neurons[1],
                             num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.avg_pool2d(x, 5)
        # print(x.shape)
        x = x.view(-1, self._resulting_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

