import collections
import random

import torch.optim as optim
import torchvision
from torchvision import transforms
from network import retinanet, csv_eval, retinanet_normal
from network.dataloader import (
    CSVDataset,
    collater,
    Resizer,
    AspectRatioBasedSampler,
    Augmenter,
    Crop,
    crop_collater,
    LabelFlip,
)

from network.mnist_data_loader import *
import torch.nn.functional as F

from torch.utils.data import DataLoader
from utils import *
import wandb
import time
import csv
import higher
from tqdm import tqdm
import plotly

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

hyperparameters = {
    'lr': 1e-3,
    'momentum': 0.9,
    'batch_size': 100,
    'num_iterations': 8000,
}

data_loader = get_mnist_loader(hyperparameters['batch_size'], classes=[0, 9], proportion=0.1,
                               mode="train")
test_loader = get_mnist_loader(hyperparameters['batch_size'], classes=[0, 9], proportion=0.0,
                               mode="test")


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


val_data = to_var(data_loader.dataset.data_val, requires_grad=False)
val_labels = to_var(data_loader.dataset.labels_val, requires_grad=False)


for i, (img, label) in enumerate(data_loader):
    print(img.size(), label)
    break

for i, (img, label) in enumerate(test_loader):
    print(img.size(), label)
    break


def build_model():
    net = retinanet.LeNet(n_out=10)

    if torch.cuda.is_available():
        net.cuda()
        torch.backends.cudnn.benchmark = True

    opt = torch.optim.SGD(net.params(), lr=hyperparameters["lr"])

    return net, opt


val_data = to_var(data_loader.dataset.data_val, requires_grad=False)
val_labels = to_var(data_loader.dataset.labels_val, requires_grad=False)

loss = torch.nn.CrossEntropyLoss()

def donorm(prob):
    net, opt = build_model()

    net_losses = []
    plot_step = 8000
    net_l = 0

    smoothing_alpha = 0.9
    accuracy_log = []
    for i in tqdm(range(hyperparameters['num_iterations'])):
        net.train()
        image, labels = next(iter(data_loader))

        image = to_var(image, requires_grad=False)
        labels = to_var(labels, requires_grad=False)

        y = net(image)

        cost = loss(y, labels)

        opt.zero_grad()
        cost.backward()
        opt.step()

        net_l = smoothing_alpha * net_l + (1 - smoothing_alpha) * cost.item()
        net_losses.append(net_l / (1 - smoothing_alpha ** (i + 1)))

        if (1+i) % 100 == 0:
            net.eval()

            acc = []
            for itr, (test_img, test_label) in enumerate(test_loader):
                test_img = to_var(test_img, requires_grad=False)
                test_label = to_var(test_label, requires_grad=False)

                output = net(test_img)
                predicted = torch.argmax(torch.softmax(output, dim=1), dim=1)#(torch.sigmoid(output) > 0.5).int()

                acc.append((predicted.int() == test_label.int()).float())

            accuracy = torch.cat(acc, dim=0).mean().cpu()
            accuracy_log.append(np.array([i, accuracy])[None])
            if (1+i) % plot_step == 0:
                fig, axes = plt.subplots(1, 2, figsize=(13, 5))
                ax1, ax2 = axes.ravel()
                fig.suptitle(f"norm w. Prob: {prob}")
                ax1.plot(net_losses, label='net_losses')
                ax1.set_ylabel("Losses")
                ax1.set_xlabel("Iteration")
                ax1.legend()

                acc_log = np.concatenate(accuracy_log, axis=0)
                ax2.plot(acc_log[:, 0], acc_log[:, 1])
                ax2.set_ylabel('Accuracy')
                ax2.set_xlabel('Iteration')
                plt.show()

lossr = torch.nn.CrossEntropyLoss(reduction='none')


def train_lre(prob):
    net, opt = build_model()

    meta_losses_clean = []
    net_losses = []
    plot_step = 8000

    smoothing_alpha = 0.9

    meta_l = 0
    net_l = 0
    accuracy_log = []
    for i in tqdm(range(hyperparameters['num_iterations'])):
        net.train()
        # Line 2 get batch of data
        image, labels = next(iter(data_loader))
        # since validation data is small I just fixed them instead of building an iterator
        # initialize a dummy network for the meta learning of the weights
        meta_net = retinanet.LeNet(n_out=10)
        meta_net.load_state_dict(net.state_dict())

        if torch.cuda.is_available():
            meta_net.cuda()

        image = to_var(image, requires_grad=False)
        labels = to_var(labels, requires_grad=False)

        # Lines 4 - 5 initial forward pass to compute the initial weighted loss
        y_f_hat = meta_net(image)
        cost = lossr(y_f_hat, labels)
        eps = to_var(torch.zeros(cost.size()))
        l_f_meta = torch.sum(cost * eps)

        meta_net.zero_grad()

        # Line 6 perform a parameter update
        grads = torch.autograd.grad(l_f_meta, (meta_net.params()), create_graph=True)
        meta_net.update_params(hyperparameters['lr'], source_params=grads)

        # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
        y_g_hat = meta_net(val_data)

        l_g_meta = loss(y_g_hat, val_labels)

        grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]

        # Line 11 computing and normalizing the weights
        w_tilde = torch.clamp(-grad_eps, min=0)
        norm_c = torch.sum(w_tilde)

        if norm_c != 0:
            w = w_tilde / norm_c
        else:
            w = w_tilde

        # Lines 12 - 14 computing for the loss with the computed weights
        # and then perform a parameter update
        y_f_hat = net(image)
        cost = lossr(y_f_hat, labels)
        l_f = torch.sum(cost * w)

        opt.zero_grad()
        l_f.backward()
        opt.step()

        meta_l = smoothing_alpha * meta_l + (1 - smoothing_alpha) * l_g_meta.item()
        meta_losses_clean.append(meta_l / (1 - smoothing_alpha ** (i + 1)))

        net_l = smoothing_alpha * net_l + (1 - smoothing_alpha) * l_f.item()
        net_losses.append(net_l / (1 - smoothing_alpha ** (i + 1)))

        if (1+i) % 100 == 0:
            net.eval()

            acc = []
            for itr, (test_img, test_label) in enumerate(test_loader):
                test_img = to_var(test_img, requires_grad=False)
                test_label = to_var(test_label, requires_grad=False)

                output = net(test_img)
                predicted = torch.argmax(torch.softmax(output, dim=1), dim=1) #(torch.sigmoid(output) > 0.5).int()

                acc.append((predicted.int() == test_label.int()).float())

            accuracy = torch.cat(acc, dim=0).mean().cpu()
            accuracy_log.append(np.array([i, accuracy])[None])
            if (1+i) % plot_step == 0:
                fig, axes = plt.subplots(1, 2, figsize=(13, 5))
                ax1, ax2 = axes.ravel()
                fig.suptitle(f"Rw with Prob{prob}")
                ax1.plot(meta_losses_clean, label='meta_losses_clean')
                ax1.plot(net_losses, label='net_losses')
                ax1.set_ylabel("Losses")
                ax1.set_xlabel("Iteration")
                ax1.legend()

                acc_log = np.concatenate(accuracy_log, axis=0)
                ax2.plot(acc_log[:, 0], acc_log[:, 1])
                ax2.set_ylabel('Accuracy')
                ax2.set_xlabel('Iteration')
                plt.show()

        # return accuracy
    return np.mean(acc_log[-6:-1, 1])


num_repeats = 5
proportions = [0, 0.5, 0.75, 0.9]
accuracy_log = {}

for prop in proportions:
    data_loader = get_mnist_loader(hyperparameters['batch_size'], classes=[9, 4], proportion=prop, mode="train")
    val_data = to_var(data_loader.dataset.data_val, requires_grad=False)
    val_labels = to_var(data_loader.dataset.labels_val, requires_grad=False)

    for k in range(num_repeats):
        donorm(prop)
        accuracy = train_lre(prop)

        if prop in accuracy_log:
            accuracy_log[prop].append(accuracy)
        else:
            accuracy_log[prop] = [accuracy]

plt.figure(figsize=(10, 8))
for prop in proportions:
    accuracies = accuracy_log[prop]
    plt.scatter([prop] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k, v in sorted(accuracy_log.items())])
accuracies_std = np.array([np.std(v) for k, v in sorted(accuracy_log.items())])
plt.errorbar(proportions, accuracies_mean, yerr=accuracies_std)
plt.title('Performance on varying class proportions')
plt.xlabel('proportions')
plt.ylabel('Accuracy')
plt.show()





def replace_with_pseudo(idxs, crop_ids, pseudo_labels, labels, parser, epoch):
    for i, x in enumerate(idxs):
        pseudo_label_id = "{}_{}".format(str(x), crop_ids[i])
        if pseudo_label_id in pseudo_labels.keys():
            label_made_at_epoch = pseudo_labels[pseudo_label_id][1]
            if label_made_at_epoch < epoch - 10:
                return
            pseudo = pseudo_labels[pseudo_label_id][0].shape
            original = labels.data.shape
            if pseudo[0] > original[1]:
                temp = torch.ones([parser.batch_size, pseudo[0], pseudo[1]]) * -1
                temp[:, 0 : original[1], :] = labels.data
                labels.data = temp.cuda()
            elif pseudo[0] < original[1]:
                labels.data[i][0 : pseudo[0], :] = pseudo_labels[pseudo_label_id][
                    0
                ].cuda()
            else:
                labels.data[i] = pseudo_labels[pseudo_label_id][0].cuda()

