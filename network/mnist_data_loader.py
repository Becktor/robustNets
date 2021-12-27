import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import plotly


class MNISTImbalanced:
    def __init__(self, n_items=5000, proportion=0.9, n_val=5, random_seed=1, mode="train"):
        if mode == "train":
            self.mnist = datasets.MNIST('data', train=True, download=True)
        else:
            self.mnist = datasets.MNIST('data', train=False, download=True)
            proportion = 0.0
            n_val = 0
        self.transform = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.data = []
        self.data_val = []
        self.labels = []
        self.labels_val = []

        if mode == "train":
            data_source = self.mnist.data
            label_source = self.mnist.targets
        else:
            data_source = self.mnist.data
            label_source = self.mnist.targets
        classes = list(range(10))

        for i in classes:
            tmp_idx = np.where(label_source == i)[0]
            np.random.shuffle(tmp_idx)
            tmp_idx = torch.from_numpy(tmp_idx)
            n_class = int(np.floor(len(tmp_idx) * proportion))
            img = data_source[tmp_idx[:n_class]]
            self.data.append(img)
            cl = torch.from_numpy(np.random.randint(9, size=len(tmp_idx[:n_class]))).type(
                torch.LongTensor)
            self.labels.append(cl)

            if mode == "train":
                img_val = data_source[tmp_idx[n_class:n_class + n_val]]
                for idx in range(img_val.size(0)):
                    img_tmp = Image.fromarray(img_val[idx].numpy(), mode='L')
                    img_tmp = self.transform(img_tmp)

                    self.data_val.append(img_tmp.unsqueeze(0))
                cl_val = label_source[tmp_idx[n_class:n_class + n_val]].type(torch.LongTensor)
                self.labels_val.append(cl_val)

            img_val = data_source[tmp_idx[n_class + n_val:]]
            self.data.append(img_val)
            cl_val = label_source[tmp_idx[n_class + n_val:]].type(torch.LongTensor)
            self.labels.append(cl_val)

        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

        if mode == "train":
            self.data_val = torch.cat(self.data_val, dim=0)
            self.labels_val = torch.cat(self.labels_val, dim=0)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target



def get_mnist_loader(batch_size, classes=[9, 4], n_items=5000, proportion=0.9, n_val=5, mode='train'):
    """Build and return data loader."""

    dataset = MNISTImbalanced(n_items=n_items, proportion=proportion, n_val=n_val, mode=mode)

    shuffle = False
    if mode == 'train':
        shuffle = True
    shuffle = True
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader
