import argparse
import os
import time

import cv2
import numpy as np
import torch

if os.name == 'nt':
    import ctypes

    ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')
from network import retinanet, csv_eval
from torch.utils.data import DataLoader
from torchvision import transforms

from network.dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, UnNormalizer, Normalizer

assert torch.__version__.split('.')[0] == '1'


# print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None, model=None):
    parser = argparse.ArgumentParser(description='Validation script for RetinaNet network.')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--model', help='Path to model (.pt) file.')

    parser = parser.parse_args(args)

    dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                             transform=transforms.Compose([Normalizer(), Resizer()]))

    model = torch.load(parser.model)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            model = model.cuda()

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    model.eval()

    print('Evaluating dataset')
    mAP, rl = csv_eval.evaluate(dataset_val, model, 0.3, 0.3)
    mAP, rl = csv_eval.evaluate(dataset_val, model, 0.3, 0.5)
    mAP, rl = csv_eval.evaluate(dataset_val, model, 0.3, 0.7)

if __name__ == '__main__':
    main()
