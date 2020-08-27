import argparse
import collections
import datetime as dt
import os

import numpy as np

import torch

from network.activations import GroupSort, MaxMin
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
    parser.add_argument('--depth', help='ResNet depth, must be one of 18, 34, 50, 101, 152', type=int, default=18)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=1)
    parser.add_argument('--noise', help='Batch size', type=bool, default=False)
    parser.add_argument('--continue_training', help='Path to previous ckp', type=str, default=None)
    parser.add_argument('--pre_trained', help='ResNet base pre-trained or not', type=bool, default=True)

    parser = parser.parse_args(args)

    dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                               transform=transforms.Compose([Augmenter(), Resizer()]))

    if parser.csv_val is None:
        dataset_val = None
        print('No validation annotations provided.')
    else:
        dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                 transform=transforms.Compose([Resizer()]))

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
        model = retinanet.resnet50(num_classes=dataset_train.num_classes(), pretrained=pre_trained,
                                   act=MaxMin(axis=1))
    elif parser.depth == 101:
        model = retinanet.resnet101(num_classes=dataset_train.num_classes(), pretrained=pre_trained)
    elif parser.depth == 152:
        model = retinanet.resnet152(num_classes=dataset_train.num_classes(), pretrained=pre_trained)
    else:
        raise ValueError('Unsupported model d/home/jobe/git/robustNetsepth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True
    if use_gpu:
        model = model.cuda()
    prev_epoch = 0
    mAP = 0
    model = model.cuda() #torch.nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_dir = os.path.join('trained_models', 'model') + dt.datetime.now().strftime("%j_%H%M")

    if parser.continue_training is None:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        writer = SummaryWriter(checkpoint_dir + "/tb_event")
        dataiter = iter(dataloader_train)
        data = dataiter.next()
        writer.add_graph(model, data['img'].cuda().float())

    else:
        model, optimizer, checkpoint_dict = load_ckp(parser.continue_training, model, optimizer)
        checkpoint_dir = parser.continue_training
        writer = SummaryWriter(checkpoint_dir + "/tb_event")
        prev_epoch = checkpoint_dict['epoch']
        mAP = checkpoint_dict['mAP']

    model.training = True
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
    loss_hist = collections.deque(maxlen=500)
    model.train()
    model.freeze_bn()
    print('Num training images: {}'.format(len(dataset_train)))
    meta_losses_clean = []
    net_losses = []
    for epoch_num in range(parser.epochs):
        curr_epoch = prev_epoch + epoch_num
        model.train()
        model.freeze_bn()
        epoch_loss = []
        lr = get_lr(optimizer)
        print(lr)
        print('============= Starting Epoch {}============\n'.format(curr_epoch))
        for iter_num, data in enumerate(dataloader_train):
            #try:
                # Line 2 get batch of data
                # since validation data is small I just fixed them instead of building an iterator
                # initialize a dummy network for the meta learning of the weights
                meta_model = retinanet.resnet18(num_classes=dataset_train.num_classes(), pretrained=pre_trained)
                #meta_model = torch.nn.DataParallel(meta_model).cuda()
                meta_model.load_state_dict(model.state_dict())

                if torch.cuda.is_available():
                    meta_model.cuda()

                image = to_var(data['img'], requires_grad=False)
                labels = to_var(data['annot'], requires_grad=False)

                # Lines 4 - 5 initial forward pass to compute the initial weighted loss
                meta_classification_loss, meta_regression_loss = meta_model([image, labels])
                meta_classification_loss = meta_classification_loss.mean()
                meta_regression_loss = meta_regression_loss.mean()

                eps = to_var(torch.zeros(meta_classification_loss.size()))
                l_c_meta = torch.sum(meta_classification_loss * eps)
                l_r_meta = torch.sum(meta_regression_loss * eps)
                l_f_meta = l_c_meta + l_r_meta
                meta_model.zero_grad()

                # Line 6 perform a parameter update
                grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)

                lr = 1e-3
                meta_model.update_params(lr, source_params=grads)

                # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
                v_image = to_var(data['img'], requires_grad=False)
                v_labels = to_var(data['annot'], requires_grad=False)

                y_meta_classification_loss, y_meta_regression_loss = meta_model([v_image, v_labels])
                y_c_meta = torch.sum(y_meta_classification_loss)
                y_r_meta = torch.sum(y_meta_regression_loss)
                l_g_meta = y_c_meta + y_r_meta

                grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)
                # Line 11 computing and normalizing the weights
                w_tilde = torch.clamp(-grad_eps[0], min=0)
                norm_c = torch.sum(w_tilde)

                if norm_c != 0:
                    w = w_tilde / norm_c
                else:
                    w = w_tilde

                # Lines 12 - 14 computing for the loss with the computed weights
                # and then perform a parameter update
                classification_loss, regression_loss = meta_model([v_image, v_labels])

                cost = classification_loss + regression_loss

                loss = torch.sum(cost * w)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

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
            #except Exception as e:
            #    print(e)
            #    continue

        if parser.csv_val is not None:
            print('Evaluating dataset')

            _ap, rl = csv_eval.evaluate(dataset_val, model, 0.3, 0.7)

        scheduler.step(np.mean(epoch_loss))

        # Write to Tensorboard
        writer.add_scalar("train/running_loss", np.mean(loss_hist), curr_epoch)

        writer.add_scalar("val/Buoy_Recall", rl[0][1], curr_epoch)
        writer.add_scalar("val/Buoy_Precision", rl[0][2], curr_epoch)

        writer.add_scalar("val/Boat_Recall", rl[1][1], curr_epoch)
        writer.add_scalar("val/Boat_Precision", rl[1][2], curr_epoch)

        writer.add_scalar("mAP/AP_Buoy", rl[2][1], curr_epoch)
        writer.add_scalar("mAP/AP_Boat", rl[3][1], curr_epoch)
        writer.add_scalar("mAP/mAP", rl[4], curr_epoch)

        writer.add_scalar("lr/Learning Rate", lr, curr_epoch)

        checkpoint = {
            'epoch': curr_epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'buoy_AP': rl[2][1],
            'boat_AP': rl[3][1],
            'mAP': rl[4]
        }

        if rl[4] > mAP:
            save_ckp(checkpoint, model.module, True, checkpoint_dir, curr_epoch)
        else:
            save_ckp(checkpoint, model.module, False, checkpoint_dir, curr_epoch)

        loss_file = open(os.path.join(checkpoint_dir, "loss.csv"), "a+")
        loss_file.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(curr_epoch, np.mean(loss_hist),
                                                                      rl[0], rl[1], rl[2], rl[3],
                                                                      rl[2][1], rl[3][1], rl[4]))
        loss_file.close()

    model.eval()
    torch.save(model.module, 'model_final.pt')


if __name__ == '__main__':
    main()
