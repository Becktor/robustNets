import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch import Tensor
from torch.nn import Conv2d
from torchvision.ops import nms

from network import losses
from network.activations import *
from network.anchors import Anchors
from network.layers.bjork_conv2d import BjorckConv2d
from network.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256, act=nn.ReLU(), conv=nn.Conv2d):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = conv(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_1 = nn.utils.spectral_norm(self.P5_1)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = conv(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P5_2 = nn.utils.spectral_norm(self.P5_2)

        # add P5 elementwise to C4
        self.P4_1 = conv(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_1 = nn.utils.spectral_norm(self.P4_1)

        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = conv(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P4_2 = nn.utils.spectral_norm(self.P4_2)

        # add P4 elementwise to C3
        self.P3_1 = conv(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_1 = nn.utils.spectral_norm(self.P3_1)
        self.P3_2 = conv(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P3_2 = nn.utils.spectral_norm(self.P3_2)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = conv(C5_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.P6 = nn.utils.spectral_norm(self.P6)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = act  # GroupSort(2,axis=1)
        self.P7_2 = conv(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.P7_2 = nn.utils.spectral_norm(self.P7_2)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256, act=nn.ReLU(), conv=nn.Conv2d):
        super(RegressionModel, self).__init__()

        self.conv1 = conv(num_features_in, feature_size, kernel_size=3, padding=1)
        self.conv1 = nn.utils.spectral_norm(self.conv1)
        self.act1 = act

        self.conv2 = conv(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv2 = nn.utils.spectral_norm(self.conv2)
        self.act2 = act

        self.conv3 = conv(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv3 = nn.utils.spectral_norm(self.conv3)
        self.act3 = act

        self.conv4 = conv(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv4 = nn.utils.spectral_norm(self.conv4)
        self.act4 = act

        self.output = conv(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):

    def __init__(self, num_features_in, num_anchors=9, num_classes=80, feature_size=256, act=nn.ReLU(), conv=nn.Conv2d,
                 sigmoid=True):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = conv(num_features_in, feature_size, kernel_size=3, padding=1)
        self.conv1 = nn.utils.spectral_norm(self.conv1)
        self.act1 = act
        self.conv2 = conv(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv2 = nn.utils.spectral_norm(self.conv2)
        self.act2 = act

        self.conv3 = conv(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv3 = nn.utils.spectral_norm(self.conv3)
        self.act3 = act

        self.conv4 = conv(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv4 = nn.utils.spectral_norm(self.conv4)
        self.act4 = act

        self.output = conv(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = act
        if sigmoid:
            self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers, act=nn.ReLU(), conv=nn.Conv2d, spectral_norm=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.spectral_norm = spectral_norm
        self.conv1 = conv(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.act = act
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if self.spectral_norm:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
        else:
            self.bn1 = nn.BatchNorm2d(64)
        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], act=act, conv=conv)

        self.regressionModel = RegressionModel(256, num_anchors=15, act=act, conv=conv)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes, num_anchors=15, act=act, conv=conv)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, conv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1, act=nn.ReLU(), conv=nn.Conv2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.spectral_norm:
                downsample = nn.utils.spectral_norm(conv(self.inplanes, planes * block.expansion,
                                                         kernel_size=1, stride=stride, bias=False))
            else:
                downsample = nn.Sequential(
                    conv(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = [block(self.inplanes, planes, stride, downsample, act=act,
                        conv=conv, spectral_norm=self.spectral_norm)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, act=act, conv=conv))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        """Freeze BatchNorm layers."""
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):

        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        x = self.conv1(img_batch)
        if not self.spectral_norm:
            x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch)

        if self.training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            scores = torch.max(classification, dim=2, keepdim=True)[0]

            scores_over_thresh = (scores > 0.05)[0, :, 0]  # certainty threshold

            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            anchors_nms_idx = nms(transformed_anchors[0, :, :], scores[0, :, 0], 0.5)  # iou?

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]


def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model
