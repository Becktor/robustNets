import argparse
import collections
import datetime as dt
import os

import numpy as np

import torch

from network.activations import GroupSort
from network.layers.bjork_conv2d import BjorckConv2d

if os.name == 'nt':
    import ctypes

    ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from network import retinanet
from network.dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader
from network import csv_eval
from utils import *

assert torch.__version__.split('.')[0] == '1'


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--depth', help='ResNet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=2)
    parser.add_argument('--noise', help='Batch size', type=bool, default=False)
    parser.add_argument('--continue_training', help='Path to previous ckp', type=str, default=None)
    parser.add_argument('--pre_trained', help='ResNet base pre-trained or not', type=bool, default=True)

    parser = parser.parse_args(args)

    dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                               transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

    if parser.csv_val is None:
        dataset_val = None
        print('No validation annotations provided.')
    else:
        dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                 transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    pre_trained = False
    if parser.pre_trained:
        pre_trained = True
    # Create the model
    if parser.depth == 18:
        model = retinanet.resnet18(num_classes=dataset_train.num_classes(), pretrained=pre_trained)
    elif parser.depth == 34:
        model = retinanet.resnet34(num_classes=dataset_train.num_classes(), pretrained=pre_trained)
    elif parser.depth == 50:
        model = retinanet.resnet50(num_classes=dataset_train.num_classes(), pretrained=pre_trained)
    elif parser.depth == 51:
        model = retinanet.resnet50(num_classes=dataset_train.num_classes(), pretrained=pre_trained,
                                   act=GroupSort(2, axis=1), conv=BjorckConv2d)
    elif parser.depth == 101:
        model = retinanet.resnet101(num_classes=dataset_train.num_classes(), pretrained=pre_trained)
    elif parser.depth == 152:
        model = retinanet.resnet152(num_classes=dataset_train.num_classes(), pretrained=pre_trained)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True
    if use_gpu:
        model = model.cuda()
    prev_epoch = 0
    boat_mAP = 0
    buoy_mAP = 0
    model = torch.nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    checkpoint_dir = os.path.join('trained_models', 'model') + dt.datetime.now().strftime("%j_%H%M")

    writer = SummaryWriter(checkpoint_dir + "/tb_event")
    if parser.continue_training is None:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        writer.add_graph(retinanet)
    else:
        model, optimizer, checkpoint_dict = load_ckp(parser.continue_training, model, optimizer)
        checkpoint_dir = parser.continue_training
        prev_epoch = checkpoint_dict['epoch']
        boat_mAP = checkpoint_dict['boat_mAP']
        buoy_mAP = checkpoint_dict['buoy_mAP']

    model.training = True
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=500)
    model.train()
    model.module.freeze_bn()
    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(parser.epochs):
        curr_epoch = prev_epoch + epoch_num
        model.train()
        model.module.freeze_bn()
        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()
                classification_loss, regression_loss = model([data['img'].cuda().float(), data['annot']])
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss
                if bool(loss == 0):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()

                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))
                if iter_num % 1 == 0:
                    print(
                        'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | '
                        'Running loss: {:1.5f}'.format(
                            curr_epoch, iter_num, float(classification_loss), float(regression_loss),
                            np.mean(loss_hist)), end='\r')

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        if parser.csv_val is not None:
            print('Evaluating dataset')
            mAP, rl = csv_eval.evaluate(dataset_val, model, 0.3, 0.3)

        scheduler.step(np.mean(epoch_loss))

        # Write to Tensorboard
        writer.add_scalar("train/running_loss", np.mean(loss_hist), curr_epoch)
        writer.add_scalar("val/mAP_Buoy", rl[2][1], curr_epoch)
        writer.add_scalar("val/mAP_Boat", rl[3][1], curr_epoch)

        writer.add_scalar("val/Buoy_Precision", rl[0][2], curr_epoch)
        writer.add_scalar("val/Buoy_Recall", rl[0][1], curr_epoch)

        writer.add_scalar("val/Boat_Precision", rl[1][2], curr_epoch)
        writer.add_scalar("val/Boat_Recall", rl[1][1], curr_epoch)

        checkpoint = {
            'epoch': curr_epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'buoy_mAP': rl[2][1],
            'boat_mAP': rl[3][1]
        }

        if rl[3][1] > boat_mAP and rl[2][1] > buoy_mAP:
            boat_mAP = rl[3][1]
            buoy_mAP = rl[2][1]
            save_ckp(checkpoint, model, True, checkpoint_dir, curr_epoch)
        else:
            save_ckp(checkpoint, model, False, checkpoint_dir, curr_epoch)

        loss_file = open(os.path.join(checkpoint_dir, "loss.csv"), "a+")
        loss_file.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(curr_epoch, np.mean(loss_hist),
                                                                  rl[0], rl[1], rl[2], rl[3], buoy_mAP, boat_mAP))
        loss_file.close()

    model.eval()
    torch.save(model, 'model_final.pt')


if __name__ == '__main__':
    main()
