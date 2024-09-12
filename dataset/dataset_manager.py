import torch
from torch.utils.data import random_split, DataLoader, TensorDataset, Subset
import torchvision
from torchvision import transforms
import numpy as np
import pickle as pkl

from data_utils import load_amat_file, RotMNIST, split_train_val, get_bg_rot_data

class DatasetManager:
    def __init__(self, cfg, overfit=False):
        self.cfg = cfg

        self.overfit = overfit
        #print(self.cfg.exp.data)
        self.data = cfg.exp.data
        
        if self.overfit:
            self.bs = cfg.exp.overfit.bs
            self.num_workers = 0
        else:
            self.bs = self.data.batch_size
            self.num_workers = self.data.num_workers

    def get_dataloader(self):
        if self.cfg.exp.model.dataset == 'cifar10':
            return self._load_cifar10()
        elif self.cfg.exp.model.dataset == 'mnist-noise':
            return self._load_mnist_noise()
        elif self.cfg.exp.model.dataset == 'mnist-bg-rot':
            return self._load_mnist_bg_rot()
        elif self.cfg.exp.model.dataset == 'norb':
            return self._load_norb()
        elif self.cfg.exp.model.dataset == 'smallnorb':
            return self._load_smallnorb()
        elif self.cfg.exp.model.dataset == 'rot-mnist':
            return self._load_rot_mnist()
        elif self.cfg.exp.model.dataset == 'rectangles':
            pass

    def _load_cifar10(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=self.data.path, train=True, transform=transform, download=False)
        test_dataset = torchvision.datasets.CIFAR10(root=self.data.path, train=False, transform=transform, download=False)
        train_dataset, valid_dataset = self._split_dataset(train_dataset)
        return self._create_dataloaders(train_dataset, valid_dataset, test_dataset)

    def _load_mnist_noise(self):
        # Assumes `load_amat_file` is already defined somewhere
        data, labels = load_amat_file(self.cfg.exp.data.path)
        train_set, train_labels = data[:-2000], labels[:-2000]
        test_set, test_labels = data[-2000:], labels[-2000:]
        train_dataset = RotMNIST(train_set, train_labels)
        test_dataset = RotMNIST(test_set, test_labels)
        train_dataset, valid_dataset = self._split_dataset(train_dataset)
        return self._create_dataloaders(train_dataset, valid_dataset, test_dataset)

    def _load_mnist_bg_rot(self):
        train_X, train_Y, test_X, test_Y = get_bg_rot_data()
        train_X, train_Y, val_X, val_Y = split_train_val(train_X, train_Y, val_fraction=0.2, train_fraction=None)
        train_dataset = TensorDataset(train_X, train_Y)
        valid_dataset = TensorDataset(val_X, val_Y)
        test_dataset = TensorDataset(test_X, test_Y)
        return self._create_dataloaders(train_dataset, valid_dataset, test_dataset)

    def _load_norb(self):
        train_X = np.load(self.cfg.exp.data.path + '/train_X.npy', allow_pickle=True)
        train_Y = np.load(self.cfg.exp.data.path + '/train_Y.npy', allow_pickle=True)
        test_X = np.load(self.cfg.exp.data.path + '/test_X.npy', allow_pickle=True)
        test_Y = np.load(self.cfg.exp.data.path + '/test_Y.npy', allow_pickle=True)
        train_X, train_Y, val_X, val_Y = split_train_val(torch.FloatTensor(train_X), torch.FloatTensor(train_Y), val_fraction=0.2)
        train_dataset = TensorDataset(train_X, train_Y)
        valid_dataset = TensorDataset(val_X, val_Y)
        test_dataset = TensorDataset(test_X, test_Y)
        return self._create_dataloaders(train_dataset, valid_dataset, test_dataset)

    def _load_smallnorb(self):
        with open(self.cfg.exp.data.path, 'rb') as f:
            data = pkl.load(f)
        train_X, train_Y = torch.FloatTensor(data['train_X']), torch.FloatTensor(data['train_Y'])
        test_X, test_Y = torch.FloatTensor(data['test_X']), torch.FloatTensor(data['test_Y'])
        train_X, train_Y, val_X, val_Y = split_train_val(train_X, train_Y, val_fraction=0.2)
        train_dataset = TensorDataset(train_X, train_Y)
        valid_dataset = TensorDataset(val_X, val_Y)
        test_dataset = TensorDataset(test_X, test_Y)
        return self._create_dataloaders(train_dataset, valid_dataset, test_dataset)

    def _load_rot_mnist(self):
        train_set, train_labels = load_amat_file(self.cfg.exp.data.path + '/mnist_all_rotation_normalized_float_train_valid.amat')
        test_set, test_labels = load_amat_file(self.cfg.exp.data.path + '/mnist_all_rotation_normalized_float_test.amat')
        train_dataset = RotMNIST(train_set, train_labels)
        test_dataset = RotMNIST(test_set, test_labels)
        train_dataset, valid_dataset = self._split_dataset(train_dataset)
        return self._create_dataloaders(train_dataset, valid_dataset, test_dataset)

    def _split_dataset(self, dataset):
        train_set_size = int(len(dataset) * 0.8)
        valid_set_size = len(dataset) - train_set_size
        seed = torch.Generator().manual_seed(42)
        return random_split(dataset, [train_set_size, valid_set_size], generator=seed)

    def _create_dataloaders(self, train_dataset, valid_dataset, test_dataset):
        if self.overfit:
            indices = list(range(self.bs))
            train_dataset = Subset(train_dataset, indices)
            valid_dataset = train_dataset
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.bs, shuffle=True, pin_memory=True, num_workers=self.num_workers)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=self.bs, shuffle=False, pin_memory=True, num_workers=self.num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.data.batch_size, shuffle=False, pin_memory=True, num_workers=self.data.num_workers)
        return train_loader, valid_loader, test_loader

