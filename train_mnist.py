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


def build_model(hyperparameters):
    net = retinanet.LeNet(n_out=10)

    if torch.cuda.is_available():
        net.cuda()
        torch.backends.cudnn.benchmark = True

    opt = torch.optim.SGD(net.params(), lr=hyperparameters["lr"])

    return net, opt


def vanilla_mnist_conv_net(hyperparameters, data_loader, test_loader, prop, k, loss_funcs):
    net, opt = build_model(hyperparameters)

    net_losses = []
    plot_step = hyperparameters['num_iterations']
    net_l = 0

    smoothing_alpha = 0.9
    accuracy_log = []
    for i in tqdm(range(hyperparameters['num_iterations'])):
        net.train()
        image, labels, _, _ = next(iter(data_loader))

        image = to_var(image, requires_grad=False)
        labels = to_var(labels, requires_grad=False)

        y = net(image)

        cost = loss_funcs["red"](y, labels)

        opt.zero_grad()
        cost.backward()
        opt.step()

        net_l = smoothing_alpha * net_l + (1 - smoothing_alpha) * cost.item()
        net_losses.append(net_l / (1 - smoothing_alpha ** (i + 1)))

        if (1 + i) % 100 == 0:
            net.eval()

            acc = []
            for itr, (test_img, test_label, _, _) in enumerate(test_loader):
                test_img = to_var(test_img, requires_grad=False)
                test_label = to_var(test_label, requires_grad=False)

                output = net(test_img)
                predicted = torch.argmax(torch.softmax(output, dim=1), dim=1)  # (torch.sigmoid(output) > 0.5).int()

                acc.append((predicted.int() == test_label.int()).float())

            accuracy = torch.cat(acc, dim=0).mean().cpu()
            accuracy_log.append(np.array([i, accuracy])[None])
            if (1 + i) % plot_step == 0:
                fig, axes = plt.subplots(1, 2, figsize=(13, 5))
                ax1, ax2 = axes.ravel()
                fig.suptitle(f"norm w. Prob: {prop}")
                ax1.plot(net_losses, label='net_losses')
                ax1.set_ylabel("Losses")
                ax1.set_xlabel("Iteration")
                ax1.legend()

                acc_log = np.concatenate(accuracy_log, axis=0)
                ax2.plot(acc_log[:, 0], acc_log[:, 1])
                ax2.set_ylabel('Accuracy')
                ax2.set_xlabel('Iteration')
                plt.savefig(f'mnist/{k}_BS_{prop}.png')
                plt.show()
            # return accuracy
    return np.mean(acc_log[-6:-1, 1])


def train_lre(hyperparameters, data_loader, test_loader, prop, k, loss_funcs, val_data, val_labels):
    net, opt = build_model(hyperparameters)

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
        image, labels, _ = next(iter(data_loader))
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
        cost = loss_funcs["no_red"](y_f_hat, labels)
        eps = to_var(torch.zeros(cost.size()))
        l_f_meta = torch.sum(cost * eps)

        meta_net.zero_grad()

        # Line 6 perform a parameter update
        grads = torch.autograd.grad(l_f_meta, (meta_net.params()), create_graph=True)
        meta_net.update_params(hyperparameters['lr'], source_params=grads)

        # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
        y_g_hat = meta_net(val_data)

        l_g_meta = loss_funcs["red"](y_g_hat, val_labels)

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
        cost = loss_funcs["no_red"](y_f_hat, labels)
        l_f = torch.sum(cost * w)

        opt.zero_grad()
        l_f.backward()
        opt.step()

        meta_l = smoothing_alpha * meta_l + (1 - smoothing_alpha) * l_g_meta.item()
        meta_losses_clean.append(meta_l / (1 - smoothing_alpha ** (i + 1)))

        net_l = smoothing_alpha * net_l + (1 - smoothing_alpha) * l_f.item()
        net_losses.append(net_l / (1 - smoothing_alpha ** (i + 1)))

        if (1 + i) % 100 == 0:
            net.eval()

            acc = []
            for itr, (test_img, test_label, _) in enumerate(test_loader):
                test_img = to_var(test_img, requires_grad=False)
                test_label = to_var(test_label, requires_grad=False)

                output = net(test_img)
                predicted = torch.argmax(torch.softmax(output, dim=1), dim=1)  # (torch.sigmoid(output) > 0.5).int()

                acc.append((predicted.int() == test_label.int()).float())

            accuracy = torch.cat(acc, dim=0).mean().cpu()
            accuracy_log.append(np.array([i, accuracy])[None])
            if (1 + i) % plot_step == 0:
                fig, axes = plt.subplots(1, 2, figsize=(13, 5))
                ax1, ax2 = axes.ravel()
                fig.suptitle(f"Rw w. Prob: {prop}")
                ax1.plot(meta_losses_clean, label='meta loss')
                ax1.plot(net_losses, label='net loss')
                ax1.set_ylabel("Losses")
                ax1.set_xlabel("Iteration")
                ax1.legend()

                acc_log = np.concatenate(accuracy_log, axis=0)
                ax2.plot(acc_log[:, 0], acc_log[:, 1])
                ax2.set_ylabel('Accuracy')
                ax2.set_xlabel('Iteration')
                plt.savefig(f'mnist/RW_{prop}.png')
                plt.show()

        # return accuracy
    return np.mean(acc_log[-6:-1, 1])


def train_lra(hyperparameters, data_loader, test_loader, prop, k, loss_funcs, val_data, val_labels):
    net, opt = build_model(hyperparameters)
    reweight_cases = {}
    pseudo_labels = {}
    meta_losses_clean = []
    net_losses = []
    plot_step = hyperparameters['num_iterations']
    pls = []
    smoothing_alpha = 0.9

    proportions = {0: "No noise",
                   0.5: "50%",
                   0.75: "75%",
                   0.9: "90%"}

    meta_l = 0
    net_l = 0
    accuracy_log = []
    for i in tqdm(range(hyperparameters['num_iterations'])):
        net.train()
        # Line 2 get batch of data
        image, labels, index, real_label = next(iter(data_loader))

        check_and_replace_with_pseudo(index, pseudo_labels, labels, i)
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
        cost = loss_funcs["no_red"](y_f_hat, labels)
        eps = to_var(torch.zeros(cost.size()))
        l_f_meta = torch.sum(cost * eps)

        meta_net.zero_grad()

        # Line 6 perform a parameter update
        grads = torch.autograd.grad(l_f_meta, (meta_net.params()), create_graph=True)
        meta_net.update_params(hyperparameters['lr'], source_params=grads)

        # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
        y_g_hat = meta_net(val_data)

        l_g_meta = loss_funcs["red"](y_g_hat, val_labels)

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
        cost = loss_funcs["no_red"](y_f_hat, labels)
        l_f = torch.sum(cost * w)

        ## Select reannotation
        wl = torch.le(w, 0.25 / eps.shape[0])

        add_reweight_cases_to_update_anno_dict(w, wl, reweight_cases, index)
        softmax = torch.softmax(y_f_hat, dim=1)
        scores, predictions = torch.max(softmax, axis=1)
        # pseudo_labels = torch.argmax(softmax, dim=1)

        score_thresh = max(0.80, min(0.95, 1 - (i / (hyperparameters['num_iterations'] * 2))))

        score_thresh = 1-((1/4) * np.sin((i/8000)*hyperparameters['num_iterations']))

        update_annotation(wl.cpu(), index, scores, predictions, pseudo_labels, i, score_thresh, real_label)

        opt.zero_grad()
        l_f.backward()
        opt.step()

        meta_l = smoothing_alpha * meta_l + (1 - smoothing_alpha) * l_g_meta.item()
        meta_losses_clean.append(meta_l / (1 - smoothing_alpha ** (i + 1)))

        net_l = smoothing_alpha * net_l + (1 - smoothing_alpha) * l_f.item()
        net_losses.append(net_l / (1 - smoothing_alpha ** (i + 1)))

        # pseudo label accuracy

        if (1 + i) % 100 == 0:
            net.eval()

            acc = []
            for itr, (test_img, test_label, _, _) in enumerate(test_loader):
                test_img = to_var(test_img, requires_grad=False)
                test_label = to_var(test_label, requires_grad=False)

                output = net(test_img)
                predicted = torch.argmax(torch.softmax(output, dim=1), dim=1)  # (torch.sigmoid(output) > 0.5).int()

                acc.append((predicted.int() == test_label.int()).float())

            accuracy = torch.cat(acc, dim=0).mean().cpu()
            accuracy_log.append(np.array([i, accuracy])[None])

            if len(pseudo_labels) > 0:
                pr, idx, cl = zip(*pseudo_labels.values())
                vals = np.where(np.array(idx) >= i - 500)
                pr = torch.stack(pr).cpu()[vals]
                cl = torch.stack(cl)[vals]
                pl_acc = torch.mean((pr.int() == cl.int()).float())
                pls.append((pl_acc, len(idx)))
            else:
                pls.append((0, 0))

            if (1 + i) % plot_step == 0:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                ax1, ax2 = axes.ravel()
                if prop == 0:
                    fig.suptitle(f"RA with {proportions[prop]}%", size=16, y=1.05)
                else:
                    fig.suptitle(f"RA with {proportions[prop]}% noise", size=16, y=1.05)
                ax1.plot(meta_losses_clean, label='meta network loss')
                ax1.plot(net_losses, label='network loss')
                ax1.set_ylabel("Losses")
                ax1.set_xlabel("Iteration")
                ax1.legend(loc="upper right")

                accs, pls_c = zip(*pls)
                acc_log = np.concatenate(accuracy_log, axis=0)
                ax2.plot(acc_log[:, 0], acc_log[:, 1], label='validation accuracy')
                ax2.plot(acc_log[:, 0], accs, label='pseudo label accuracy')
                ax2.set_ylabel('Accuracy')
                ax2.set_xlabel('Iteration')
                ax2.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(f'mnist/{k}_RA_{prop}.png')
                plt.show()

                fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
                ax1.plot(acc_log[:, 0], pls_c, label='pseudo labels')
                ax1.set_ylabel("# Pseudo Labels")
                ax1.set_xlabel("Iteration")
                plt.tight_layout()
                plt.savefig(f'mnist/{k}_{prop}_pseudo.png')
                plt.show()

        # return accuracy

    print(f"number of reannotated: {pls[-1]}")
    return np.mean(acc_log[-6:-1, 1]), pls


### Reannotation
def add_reweight_cases_to_update_anno_dict(w, wl, reweight_cases, ds_index):
    for index, weight in enumerate(w):
        if wl[index]:
            if ds_index[index] in reweight_cases:
                tmp_cnt, tmp_loss = reweight_cases[ds_index[index]]
                tmp_loss.append(float(weight))
                sample = (tmp_cnt + 1, tmp_loss)
                reweight_cases[ds_index[index]] = sample
            else:
                reweight_cases[ds_index[index]] = (1, [float(weight)])


def update_annotation(
        update_anno,
        idxs,
        scores,
        predictions,
        pseudo_labels,
        epoch,
        score_thresh,
        real_label
):
    update_names = np.array(idxs)[update_anno]
    scores = scores[update_anno]
    for ii in range(len(scores)):
        if scores[ii] > score_thresh:
            key = f"{update_names[ii]}"
            pseudo_labels[key] = (predictions[ii].detach(), epoch, real_label[ii])


def check_and_replace_with_pseudo(index, pseudo_labels, labels, epoch):
    for ij, x in enumerate(index):
        pseudo_label_id = f"{x}"
        if pseudo_label_id in pseudo_labels.keys():
            label_made_at_epoch = pseudo_labels[pseudo_label_id][1]
            if label_made_at_epoch < epoch - 500:
                return
            labels.data[ij] = pseudo_labels[pseudo_label_id][0]


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def main():
    hyperparameters = {
        'lr': 1e-3,
        'momentum': 0.9,
        'batch_size': 100,
        'num_iterations': 20000,
    }

    data_loader = get_mnist_loader(hyperparameters['batch_size'], classes=[0, 9], proportion=0.1,
                                   mode="train")
    test_loader = get_mnist_loader(hyperparameters['batch_size'], classes=[0, 9], proportion=0.0,
                                   mode="test")

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    for i, (img, label, _, _) in enumerate(data_loader):
        print(img.size(), label)
        break

    for i, (img, label, _, _) in enumerate(test_loader):
        print(img.size(), label)
        break

    loss = torch.nn.CrossEntropyLoss()
    loss_no_red = torch.nn.CrossEntropyLoss(reduction='none')
    loss_funcs = {"red": loss, "no_red": loss_no_red}
    num_repeats = 5
    proportions = [0.9, 0.75, 0.5, 0]
    accuracy_log_v = {}
    accuracy_log_rw = {}
    accuracy_log_ra = {}
    pseudo_label_list = []
    for prop in proportions:
        data_loader = get_mnist_loader(hyperparameters['batch_size'], classes=[9, 4], proportion=prop, mode="train")
        val_data = to_var(data_loader.dataset.data_val, requires_grad=False)
        val_labels = to_var(data_loader.dataset.labels_val, requires_grad=False)

        for k in range(num_repeats):
            #accuracy_v = vanilla_mnist_conv_net(hyperparameters, data_loader, test_loader,
            #                            prop, k, loss_funcs)
            # accuracy_rw = train_lre(prop)
            accuracy_ra, pl = train_lra(hyperparameters, data_loader, test_loader,
                                        prop, k, loss_funcs, val_data, val_labels)
            pseudo_label_list.append(pl)
            #if prop in accuracy_log_v:
            #    accuracy_log_v[prop].append(accuracy_v)
            #else:
            #    accuracy_log_v[prop] = [accuracy_v]
            #
            # if prop in accuracy_log_rw:
            #     accuracy_log_rw[prop].append(accuracy_rw)
            # else:
            #     accuracy_log_rw[prop] = [accuracy_rw]

            if prop in accuracy_log_ra:

                accuracy_log_ra[prop].append(accuracy_ra)
            else:

                accuracy_log_ra[prop] = [accuracy_ra]
    #
    plt.figure(figsize=(10, 8))
    for prop in proportions:
        accuracies = accuracy_log_v[prop]
        plt.scatter([prop] * len(accuracies), accuracies)
        print(f"{prop}_accuracies: {accuracies}")
    # plot the trend line with error bars that correspond to standard deviation
    accuracies_mean = np.array([np.mean(v) for k, v in sorted(accuracy_log_v.items())])
    accuracies_std = np.array([np.std(v) for k, v in sorted(accuracy_log_v.items())])
    plt.errorbar(proportions, accuracies_mean, yerr=accuracies_std)
    plt.title('Performance on varying class proportions for baseline')
    plt.xlabel('proportions')
    plt.ylabel('Accuracy')
    plt.savefig('mnist/BS_avg.png')
    plt.show()
    #
    # plt.figure(figsize=(10, 8))
    # for prop in proportions:
    #     accuracies = accuracy_log_rw[prop]
    #     plt.scatter([prop] * len(accuracies), accuracies)
    #     print(f"{prop}_accuracies: {accuracies}")
    #
    # # plot the trend line with error bars that correspond to standard deviation
    # accuracies_mean = np.array([np.mean(v) for k, v in sorted(accuracy_log_rw.items())])
    # accuracies_std = np.array([np.std(v) for k, v in sorted(accuracy_log_rw.items())])
    # plt.errorbar(proportions, accuracies_mean, yerr=accuracies_std)
    # plt.title('Performance on varying class proportions RW')
    # plt.xlabel('proportions')
    # plt.ylabel('Accuracy')
    # plt.savefig('mnist/RW_avg.png')
    # plt.show()

    plt.figure(figsize=(10, 8))
    for prop in proportions:
        accuracies = accuracy_log_ra[prop]
        plt.scatter([prop] * len(accuracies), accuracies)
        print(f"{prop}_accuracies: {accuracies}")
    # plot the trend line with error bars that correspond to standard deviation
    accuracies_mean = np.array([np.mean(v) for k, v in sorted(accuracy_log_ra.items(), reverse=True)])
    accuracies_std = np.array([np.std(v) for k, v in sorted(accuracy_log_ra.items(), reverse=True)])

    plt.errorbar(proportions, accuracies_mean, yerr=accuracies_std)
    plt.title('Performance on varying class proportions RA')
    plt.xlabel('proportions')
    plt.ylabel('Accuracy')
    plt.savefig('mnist/RA_average.png')
    plt.show()

    #print(pseudo_label_list)


if __name__ == "__main__":
    main()
