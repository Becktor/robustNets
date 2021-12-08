import os
import re
import torch
from torch.autograd import Variable
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import cv2
import numpy as np


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def save_ckp(state, model, is_best, checkpoint_dir, epoch):
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    f_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
    torch.save(state, f_path)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best_modelpt')
        torch.save(state, best_filepath)


def load_ckp(checkpoint_filepath, model, optimizer=None, scheduler=None):
    cwd = os.path.join(os.getcwd(), checkpoint_filepath)
    path = os.path.join(cwd, 'checkpoint.pt')
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        return model, optimizer, scheduler, checkpoint
    return model


def load_base_model(path, model):
    checkpoint = torch.load(path)
    print("loading epoch: {}".format(checkpoint['epoch']))
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model


def load_best_ckp(checkpoint_filepath, model, optimizer):
    cwd = os.path.join(os.getcwd(), checkpoint_filepath)
    path = os.path.join(cwd, 'best.pt')
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def plot_input(v_image, v_labels, w_names, title="input"):
    images = v_image.cpu().detach().numpy()
    labels = v_labels.cpu().detach().numpy()
    names = w_names
    for i in range(len(images)):
        img = images[i].transpose([1, 2, 0])
        val = labels[i]
        img2 = img.copy()
        for v in val:
            color = (0, 1, 0)
            if v[4] == 0:
                color = (0, 0, 1)

            img2 = cv2.rectangle(img2, (int(v[0]), int(v[1])),
                                 (int(v[2]), int(v[3])), color=color, thickness=2)
        plt.imshow(img2)
        plt.title(title + "-> Bouy is blue, boat is green indx: " + str(i))
        plt.show()


def plot_sample(img, lbl, title="input"):
    img = img.cpu().detach().numpy().transpose([1, 2, 0])
    val = lbl.cpu().detach().numpy()
    img2 = img.copy()
    for x, v in enumerate(val):
        color = (0, 1, 0)
        if v[4] == 0:
            color = (0, 0, 1)

        img2 = cv2.rectangle(img2, (int(v[0]), int(v[1])),
                             (int(v[2]), int(v[3])), color=color, thickness=2)
    plt.imshow(img2)
    plt.title(title + "-> Bouy is blue, boat is green")
    plt.show()
