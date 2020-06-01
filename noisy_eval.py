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

def main(args=None, retinanet=None):
    parser = argparse.ArgumentParser(description='Validation script for RetinaNet network.')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--model', help='Path to model (.pt) file.')

    parser = parser.parse_args(args)

    dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                             transform=transforms.Compose([Normalizer(), Resizer()]))

    retinanet = retinanet.resnet50(dataset_val.num_classes())

    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet = torch.nn.DataParallel(retinanet).cuda()
    checkpoint = torch.load(parser.model)
    retinanet.load_state_dict(checkpoint['state_dict'])

    retinanet.eval()
    unnormalize = UnNormalizer()

    print('Evaluating dataset')
    mAP, rl = csv_eval.evaluate(dataset_val, retinanet, 0.3, 0.7)

if __name__ == '__main__':
    main()
