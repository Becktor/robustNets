import numpy as np
from collections import defaultdict
from functools import reduce
import torch
import torch.nn as nn
import sys
import math


from network.activations import *


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, act=GroupSort(2, axis=1), conv=nn.Conv2d):
        super(BasicBlock, self).__init__()
        self.conv1 = conv(inplanes, planes, kernel_size=3, stride=1,
                          padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.act = act
        self.conv2 = conv(planes, planes, kernel_size=3, stride=1,
                          padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.maxpool(out)
        #out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        #out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out


class GroupBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, act=GroupSort(2, axis=1), conv=nn.Conv2d):
        super(BasicBlock, self).__init__()
        self.conv1 = conv(inplanes, planes, kernel_size=3, stride=1,
                          padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        self.act = act
        self.conv2 = conv(planes, planes, kernel_size=3, stride=1,
                          padding=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.act(out)
        out = self.maxpool(out)

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 act=GroupSort(2, axis=1), conv=nn.Conv2d, spectral_norm=False):
        super(Bottleneck, self).__init__()
        self.spectral_norm = spectral_norm

        self.conv1 = conv(inplanes, planes, kernel_size=1, bias=False)

        self.conv2 = conv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)

        self.act = act
        self.downsample = downsample
        self.stride = stride
        if spectral_norm:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
            self.conv2 = nn.utils.spectral_norm(self.conv2)
            self.conv3 = nn.utils.spectral_norm(self.conv3)
        #else:
            #self.bn1 = nn.BatchNorm2d(planes)
            #self.bn2 = nn.BatchNorm2d(planes)
            #self.bn3 = nn.BatchNorm2d(planes * 4)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        #if not self.spectral_norm:
        #    out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        #if not self.spectral_norm:
        #    out = self.bn2(out)
        out = self.act(out)
        out = self.maxpool(out)

        out = self.conv3(out)
        #if not self.spectral_norm:
        #   out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out


class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
        else:
            self.std = std

    def forward(self, boxes, deltas):

        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes




def default_collate_op(x, y):
    if x is None:
        return [y]
    if y is None:  # avoid appending None and nan
        return x
    if type(y) == list:
        x.extend(y)
    else:
        x.append(y)
    return x


def default_summarize_op(x, dtype):
    if dtype == "scalar":
        if len(x) == 0:
            return 0
        return sum(x) / len(x)
    if dtype == "histogram":
        return torch.tensor(x)
    return x


def default_display_op(x, dtype):
    if dtype == "scalar":
        return "{:.4f}".format(x)
    if dtype == "histogram":
        return "histogram[n={}]".format(len(x))
    return x


def prod(x):
    return reduce(lambda a, b: a * b, x)


class StreamlinedModule(nn.Module):
    def __init__(self):
        self.streamline = False
        super(StreamlinedModule, self).__init__()

    def set_streamline(self, streamline=False):
        self.streamline = streamline
        return streamline


def streamline_model(model, streamline=False):
    for m in model.modules():
        if isinstance(m, StreamlinedModule):
            m.set_streamline(streamline)


# Context manager that streamlines the module of interest in the context only
class Streamline:
    def __init__(self, module, new_flag=True, old_flag=False):
        self.module = module
        self.new_flag = new_flag
        self.old_flag = old_flag

    def __enter__(self):
        streamline_model(self.module, self.new_flag)

    def __exit__(self, *args, **kwargs):
        streamline_model(self.module, self.old_flag)


# A helper object for logging all the data
class Accumulator:
    def __init__(self):
        self.data = defaultdict(list)
        self.data_dtype = defaultdict(None)

    def __call__(
        self,
        name,
        value=None,
        dtype=None,
        collate_op=default_collate_op,
        summarize_op=None,
    ):
        if value is None:
            if summarize_op is not None:
                return summarize_op(self.data[name])
            return self.data[name]
        self.data[name] = default_collate_op(self.data[name], value)
        if dtype is not None:
            self.data_dtype[name] = dtype
        assert dtype == self.data_dtype[name]

    def summarize(self, summarize_op=default_summarize_op):
        for key in self.data:
            self.data[key] = summarize_op(self.data[key], self.data_dtype[key])

    def collect(self):
        return {key: self.__call__(key) for key in self.data}

    def filter(self, dtype=None, level=None, op=None):
        if op is None:
            op = lambda x: x
        if dtype is None:
            return self.collect()
        return {
            key: op(self.__call__(key))
            for key in filter(
                lambda x: self.data_dtype[x] == dtype
                and (x.count("/") <= level if (level is not None) else True),
                self.data,
            )
        }

    def latest_str(self):
        return ", ".join(
            "{}={:.4f}".format(key, value[-1] if len(value) > 0 else math.nan)
            for key, value in self.collect().items()
        )

    def summary_str(self, dtype=None, level=None):
        return ", ".join(
            "{}={}".format(
                key, default_display_op(self.__call__(key), self.data_dtype[key])
            )
            for key in self.filter(dtype=dtype, level=level)
        )

    def __str__(self):
        return self.summary_str()


# A logger that sync terminal output to a logger file
class Logger(object):
    def __init__(self, logdir):
        self.terminal = sys.stdout
        self.log = open(logdir, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s