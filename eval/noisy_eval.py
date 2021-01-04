import argparse
import os

import torch
from torchvision import transforms
from network.dataloader import CSVDataset, Resizer, Crop
from network import csv_eval, retinanet
from utils import *
import network.layers.meta_layers


def main(args=None, model=None):
    parser = argparse.ArgumentParser(description='Validation script for RetinaNet network.')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--model', help='Path to model (.pt) file.')

    parser = parser.parse_args(args)

    dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                             transform=transforms.Compose([Crop(val=True), Resizer()]))
    model = retinanet.resnet18(num_classes=dataset_val.num_classes())
    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            model = model.cuda()

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    model, _, _ = load_ckp(parser.model, model)
    #model = model2.load(parser.model)
    model.eval()

    print('Evaluating dataset')
    mAP, rl = csv_eval.evaluate(dataset_val, model, 0.3, 0.3)
    mAP, rl = csv_eval.evaluate(dataset_val, model, 0.3, 0.5)
    mAP, rl = csv_eval.evaluate(dataset_val, model, 0.3, 0.7)

if __name__ == '__main__':
    main()
