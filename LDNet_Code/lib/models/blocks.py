from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from lib.models.channel_shuffle import channel_shuffle


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class conv_bn_relu(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding,
                 has_bn=True, has_relu=True, efficient=False, groups=1):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups)
        self.has_bn = has_bn
        self.has_relu = has_relu
        self.efficient = efficient
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        def _func_factory(conv, bn, relu, has_bn, has_relu):
            def func(x):
                x = conv(x)
                if has_bn:
                    x = bn(x)
                if has_relu:
                    x = relu(x)
                return x

            return func

        func = _func_factory(
            self.conv, self.bn, self.relu, self.has_bn, self.has_relu)

        x = func(x)

        return x


class PRM(nn.Module):

    def __init__(self, output_chl_num, efficient=False):
        super(PRM, self).__init__()
        self.output_chl_num = output_chl_num
        self.conv_bn_relu_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=3,
                                           stride=1, padding=1, has_bn=True, has_relu=True,
                                           efficient=efficient)
        self.conv_bn_relu_2_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
                                             stride=1, padding=0, has_bn=True, has_relu=True,
                                             efficient=efficient)
        self.conv_bn_relu_2_2 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
                                             stride=1, padding=0, has_bn=True, has_relu=True,
                                             efficient=efficient)
        self.sigmoid2 = nn.Sigmoid()
        self.conv_bn_relu_3_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
                                             stride=1, padding=0, has_bn=True, has_relu=True,
                                             efficient=efficient)
        self.conv_bn_relu_3_2 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=9,
                                             stride=1, padding=4, has_bn=True, has_relu=True,
                                             efficient=efficient, groups=self.output_chl_num)
        self.sigmoid3 = nn.Sigmoid()

    def forward(self, x):
        out = self.conv_bn_relu_1(x)
        out_1 = out
        out_2 = torch.nn.functional.adaptive_avg_pool2d(out_1, (1, 1))
        out_2 = self.conv_bn_relu_2_1(out_2)
        out_2 = self.conv_bn_relu_2_2(out_2)
        out_2 = self.sigmoid2(out_2)
        out_3 = self.conv_bn_relu_3_1(out_1)
        out_3 = self.conv_bn_relu_3_2(out_3)
        out_3 = self.sigmoid3(out_3)
        out = out_1.mul(1 + out_2.mul(out_3))
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class DWBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, out_planes, stride=1, downsample=None):
        super(DWBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1, groups=inplanes, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes, momentum=BN_MOMENTUM)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Stem(nn.Module):

    def __init__(self,
                 in_channels,
                 stem_channels,
                 out_channels,
                 expand_ratio):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=3,
            stride=2,
            padding=1)    #relu
        self.bn1 = nn.BatchNorm2d(stem_channels, momentum=BN_MOMENTUM)

        mid_channels = int(round(stem_channels * expand_ratio))
        branch_channels = stem_channels // 2
        if stem_channels == self.out_channels:
            inc_channels = self.out_channels - branch_channels
        else:
            inc_channels = self.out_channels - stem_channels

        self.branch1 = nn.Sequential(
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=branch_channels),
            nn.BatchNorm2d(branch_channels),
            nn.Conv2d(
                branch_channels,
                inc_channels,
                kernel_size=1,
                stride=1,
                padding=0),          #relu
            nn.BatchNorm2d(inc_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        self.expand_conv = nn.Conv2d(
            branch_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0)       #relu
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.depthwise_conv = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=mid_channels)
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.linear_conv = nn.Conv2d(
            mid_channels,
            branch_channels
            if stem_channels == self.out_channels else stem_channels,
            kernel_size=1,
            stride=1,
            padding=0)            #relu
        self.bn4 = nn.BatchNorm2d(branch_channels)

    def forward(self, x):

        def _inner_forward(x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x1, x2 = x.chunk(2, dim=1)         #在指定维度上对张量分块，返回一个张量列表

            x2 = self.expand_conv(x2)
            x2 = self.bn2(x2)
            x2 = self.relu(x2)
            x2 = self.depthwise_conv(x2)
            x2 = self.bn3(x2)
            x2 = self.linear_conv(x2)
            x2 = self.bn4(x2)
            x2 = self.relu(x2)

            out = torch.cat((self.branch1(x1), x2), dim=1)

            out = channel_shuffle(out, 2)

            return out

        out = _inner_forward(x)

        return out




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out