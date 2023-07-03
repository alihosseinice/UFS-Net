from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.leaky_relu(x, inplace=True)


class MultiConv(nn.Module):
    __constants__ = ['branch2', 'branch3']

    def __init__(self, in_channels, ch3x3red, ch3x3, ch5x5red, ch5x5, conv_block=None):
        super(MultiConv, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

    def _forward(self, x):
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        outputs = [branch2, branch3]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class UFSNet(nn.Module):
    __constants__ = ['transform_input']

    def __init__(self, num_classes=3, transform_input=False, init_weights=True, blocks=None):
        super(UFSNet, self).__init__()
        if blocks is None:
            blocks = [BasicConv2d, MultiConv]
        assert len(blocks) == 2
        conv_block = blocks[0]
        multi_conv_block = blocks[1]

        self.transform_input = transform_input

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = multi_conv_block(192, 64, 128, 16, 32)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = multi_conv_block(160, 96, 192, 24, 48)
        self.inception4b = multi_conv_block(240, 128, 256, 32, 64)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = multi_conv_block(320, 160, 256, 40, 96)
        self.inception5b = multi_conv_block(352, 192, 288, 48, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(416, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _transform_input(self, x):
        # type: (Tensor) -> Tensor
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x):
        # type: (Tensor) -> Tensor
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 160 x 28 x 28
        x = self.maxpool3(x)
        # N x 160 x 14 x 14
        x = self.inception4a(x)
        # N x 288 x 14 x 14
        x = self.inception4b(x)
        # N x 272 x 14 x 14
        x = self.maxpool4(x)
        # N x 272 x 7 x 7

        x = self.inception5a(x)
        # N x 416 x 7 x 7
        x = self.inception5b(x)
        # N x 512 x 7 x 7

        x = self.avgpool(x)
        # N x 512 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 512
        x = self.dropout(x)
        x = self.fc(x)
        # N x 8 (num_classes)
        return x

    def forward(self, x):
        # type: (Tensor) -> UFS-NetOutputs
        x = self._transform_input(x)
        x = self._forward(x)
        return x
