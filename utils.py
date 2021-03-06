import os
import re
import shutil

import torch


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def save_ckp(state, model, is_best, checkpoint_dir, epoch):
    f_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
    torch.save(state, f_path)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best_model_{}.pt'.format(epoch))
        torch.save(model, best_filepath)


def load_ckp(checkpoint_filepath, model, optimizer):
    cwd = os.path.join(os.getcwd(), checkpoint_filepath)
    path = os.path.join(cwd, 'checkpoint.pt')
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint


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
