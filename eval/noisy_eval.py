import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from network.dataloader import CSVDataset, Resizer, Crop, AspectRatioBasedSampler, crop_collater
from network import csv_eval, retinanet
from utils import *
import network.layers.meta_layers


def main(args=None, model=None):
    parser = argparse.ArgumentParser(description='Validation script for RetinaNet network.')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--model', help='Path to model (.pt) file.')

    parser = parser.parse_args(args)

    dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes)
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=True)
    dataloader_val = DataLoader(dataset_val, num_workers=4, collate_fn=crop_collater, batch_sampler=sampler_val)
    model = retinanet.resnet50(num_classes=dataset_val.num_classes())
    use_gpu = True
    path = "trained_models/final_models/"
    for file in os.listdir(path)[6:]:
        if file.endswith(".pt"):
            model_path = os.path.join(path, file)
            print(model_path)
            if use_gpu:
                if torch.cuda.is_available():
                    model = model.cuda()

            model = load_base_model(model_path, model)
            model.eval()

            print('Evaluating dataset on {}'.format(file[:-3]))
            mAP, rl = csv_eval.evaluate(dataloader_val, model, wdb=False, name=file[:-3])
            print(mAP, rl)

if __name__ == '__main__':
    main()
