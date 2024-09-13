import torch
from torch.utils.data import random_split, DataLoader, TensorDataset, Subset
import torchvision
from torchvision import transforms
import numpy as np
import pickle as pkl

from data_utils import load_amat_file, RotMNIST, split_train_val, get_bg_rot_data, EqDataset


class DatasetManager:
    """
    Manages loading and preprocessing of datasets based on configuration settings.

    Attributes:
        cfg (Config): Configuration object containing experiment settings.
        overfit (bool): Flag to indicate overfitting mode, which uses smaller datasets.
        bs (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
    """
    def __init__(self, cfg, overfit=False):
        self.cfg = cfg
        self.overfit = overfit
        self.data = cfg.exp.data

        # Set batch size and number of workers based on overfit mode
        if self.overfit:
            self.bs = cfg.exp.overfit.bs
            self.num_workers = 0
        else:
            self.bs = self.data.batch_size
            self.num_workers = self.data.num_workers
    
    def get_dataloader(self):
        """
        Returns appropriate data loaders based on the specified dataset in the configuration.

        Returns:
            tuple: Data loaders for training, validation, and testing.
        """
        dataset = self.cfg.exp.model.dataset
        if dataset == 'cifar10':
            return self._load_cifar10()
        elif dataset == 'mnist-noise':
            return self._load_mnist_noise()
        elif dataset == 'mnist-bg-rot':
            return self._load_mnist_bg_rot()
        elif dataset == 'norb':
            return self._load_norb()
        elif dataset == 'smallnorb':
            return self._load_smallnorb()
        elif dataset == 'rot-mnist':
            return self._load_rot_mnist()
        elif dataset == 'phiflow':
            return self._load_phiflow()
        elif dataset == 'rectangles':
            pass
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
    def _load_cifar10(self):
        """
        Loads the CIFAR-10 dataset with appropriate transformations.

        Returns:
            tuple: Data loaders for CIFAR-10 training, validation, and test sets.
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=self.data.path, train=True, transform=transform, download=False)
        test_dataset = torchvision.datasets.CIFAR10(root=self.data.path, train=False, transform=transform, download=False)
        train_dataset, valid_dataset = self._split_dataset(train_dataset)
        return self._create_dataloaders(train_dataset, valid_dataset, test_dataset)

    def _load_mnist_noise(self):
        """
        Loads a noisy MNIST dataset from .amat files.

        Returns:
            tuple: Data loaders for MNIST noise training, validation, and test sets.
        """
        data, labels = load_amat_file(self.cfg.exp.data.path)
        train_set, train_labels = data[:-2000], labels[:-2000]
        test_set, test_labels = data[-2000:], labels[-2000:]
        train_dataset = RotMNIST(train_set, train_labels)
        test_dataset = RotMNIST(test_set, test_labels)
        train_dataset, valid_dataset = self._split_dataset(train_dataset)
        return self._create_dataloaders(train_dataset, valid_dataset, test_dataset)

    def _load_mnist_bg_rot(self):
        """
        Loads a background-rotated MNIST dataset.

        Returns:
            tuple: Data loaders for MNIST BG-rot training, validation, and test sets.
        """
        train_X, train_Y, test_X, test_Y = get_bg_rot_data()
        train_X, train_Y, val_X, val_Y = split_train_val(train_X, train_Y, val_fraction=0.2)
        train_dataset = TensorDataset(train_X, train_Y)
        valid_dataset = TensorDataset(val_X, val_Y)
        test_dataset = TensorDataset(test_X, test_Y)
        return self._create_dataloaders(train_dataset, valid_dataset, test_dataset)

    def _load_norb(self):
        """
        Loads the NORB dataset.

        Returns:
            tuple: Data loaders for NORB training, validation, and test sets.
        """
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
        """
        Loads the SmallNORB dataset.

        Returns:
            tuple: Data loaders for SmallNORB training, validation, and test sets.
        """
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
        """
        Loads the rotated MNIST dataset from .amat files.

        Returns:
            tuple: Data loaders for rotated MNIST training, validation, and test sets.
        """
        train_set, train_labels = load_amat_file(self.cfg.exp.data.path + '/mnist_all_rotation_normalized_float_train_valid.amat')
        test_set, test_labels = load_amat_file(self.cfg.exp.data.path + '/mnist_all_rotation_normalized_float_test.amat')
        train_dataset = RotMNIST(train_set, train_labels)
        test_dataset = RotMNIST(test_set, test_labels)
        train_dataset, valid_dataset = self._split_dataset(train_dataset)
        return self._create_dataloaders(train_dataset, valid_dataset, test_dataset)

    def _load_phiflow(self):
        """
        Loads the PhiFlow dataset using custom EqDataset class.

        Returns:
            tuple: Data loaders for PhiFlow training, validation, future test, and domain test sets.
        """
        def create_eq_dataset(task_list, sample_list):
            return EqDataset(
                input_length=self.cfg.data.input_length, 
                mid=self.cfg.data.mid, 
                output_length=self.cfg.data.output_length,
                direc=self.cfg.data_direc, 
                task_list=task_list, 
                sample_list=sample_list, 
                stack=self.cfg.data.stack
            )

        train_set = create_eq_dataset(self.cfg.data.train_task, self.cfg.data.train_time)
        valid_set = create_eq_dataset(self.cfg.data.valid_task, self.cfg.data.valid_time)
        test_set_future = create_eq_dataset(self.cfg.data.test_future_task, self.cfg.data.test_future_time)
        test_set_domain = create_eq_dataset(self.cfg.data.test_domain_task, self.cfg.data.test_domain_time)

        return self._create_dataloaders(train_set, valid_set, test_set_future, test_set_domain, task='regression')

    def _create_dataloaders(self, train_dataset, valid_dataset, test_dataset=None, test_set_future=None, test_set_domain=None, task='classification'):
        """
        Create data loaders for training, validation, and testing.

        Parameters:
        train_dataset (Dataset): The training dataset.
        valid_dataset (Dataset): The validation dataset.
        test_dataset (Dataset, optional): The test dataset for classification.
        test_set_future (Dataset, optional): The future test dataset for regression.
        test_set_domain (Dataset, optional): The domain test dataset for regression.
        task (str): The task type ('classification' or 'regression').

        Returns:
        tuple: Data loaders for training, validation, and testing.
        """
        if task == 'classification':
            if self.overfit:
                indices = list(range(self.bs))
                train_dataset = Subset(train_dataset, indices)
                valid_dataset = train_dataset
            
            train_loader = DataLoader(dataset=train_dataset, batch_size=self.bs, shuffle=True, pin_memory=True, num_workers=self.num_workers)
            valid_loader = DataLoader(dataset=valid_dataset, batch_size=self.bs, shuffle=False, pin_memory=True, num_workers=self.num_workers)
            test_loader = DataLoader(dataset=test_dataset, batch_size=self.data.batch_size, shuffle=False, pin_memory=True, num_workers=self.data.num_workers)
            return train_loader, valid_loader, test_loader

        elif task == 'regression':
            train_loader = DataLoader(dataset=train_dataset, batch_size=self.bs, shuffle=True, pin_memory=True, num_workers=self.num_workers)
            valid_loader = DataLoader(dataset=valid_dataset, batch_size=self.bs, shuffle=False, pin_memory=True, num_workers=self.num_workers)
            test_loader_future = DataLoader(dataset=test_set_future, batch_size=self.data.batch_size, shuffle=False, pin_memory=True, num_workers=self.data.num_workers)
            test_loader_domain = DataLoader(dataset=test_set_domain, batch_size=self.data.batch_size, shuffle=False, pin_memory=True, num_workers=self.data.num_workers)
            return train_loader, valid_loader, test_loader_future, test_loader_domain

        else:
            raise ValueError("Unsupported task type. Supported types: 'classification', 'regression'.")



