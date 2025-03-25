import numpy as np
import torch
from torch.utils.data import Dataset
import pickle as pkl

def load_amat_file(file_path):
    """
    Load a .amat file and split it into features and labels.

    Parameters:
    file_path (str): Path to the .amat file.

    Returns:
    tuple: Features (X) and labels (y).
    """
    data = np.loadtxt(file_path)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

class RotMNIST(Dataset):
    """
    A custom Dataset class for the Rotated MNIST dataset.
    """
    def __init__(self, data, labels):
        """
        Initialize the dataset with data and labels.

        Parameters:
        data (np.ndarray): The data samples.
        labels (np.ndarray): The corresponding labels.
        """
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def split_train_val(train_X, train_Y, val_fraction, train_fraction=None):
    """
    Split the training data into training and validation sets.

    Parameters:
    train_X (torch.Tensor): The training data.
    train_Y (torch.Tensor): The training labels.
    val_fraction (float): The fraction of data to use for validation.
    train_fraction (float, optional): The fraction of data to use for training.

    Returns:
    tuple: Training and validation data and labels.
    """
    # Shuffle
    idx = np.arange(train_X.shape[0])
    np.random.shuffle(idx)
    train_X = train_X[idx, :]
    train_Y = train_Y[idx, :]

    # Compute validation set size
    val_size = int(val_fraction * train_X.shape[0])

    # Downsample for sample complexity experiments
    if train_fraction is not None:
        train_size = int(train_fraction * train_X.shape[0])
        assert val_size + train_size <= train_X.shape[0]
    else:
        train_size = train_X.shape[0] - val_size

    # Shuffle X
    idx = np.arange(train_X.shape[0])
    np.random.shuffle(idx)

    train_idx = idx[:train_size]
    val_idx = idx[-val_size:]
    val_X = train_X[val_idx, :]
    val_Y = train_Y[val_idx, :]
    train_X = train_X[train_idx, :]
    train_Y = train_Y[train_idx, :]

    print('train_X: ', train_X.shape)
    print('train_Y: ', train_Y.shape)
    print('val_X: ', val_X.shape)
    print('val_Y: ', val_Y.shape)

    return train_X, train_Y, val_X, val_Y

def get_bg_rot_data(train_loc, test_loc):
    """
    Load mnist-bg-rot data from pickle files.

    Parameters:
    train_loc (str): Path to the training data pickle file.
    test_loc (str): Path to the testing data pickle file.

    Returns:
    tuple: Training and testing data and labels as torch.FloatTensors.
    """
    with open(train_loc, 'rb') as f:
        train_data = pkl.load(f)
    train_X = train_data['X']
    train_Y = train_data['Y']

    with open(test_loc, 'rb') as f:
        test_data = pkl.load(f)
    test_X = test_data['X']
    test_Y = test_data['Y']

    return torch.FloatTensor(train_X), torch.FloatTensor(train_Y), torch.FloatTensor(test_X), torch.FloatTensor(test_Y)

class EqDataset(Dataset):
    """
    A custom Dataset class for PhiFlow and JetFlow data.
    """
    def __init__(self, input_length, mid, output_length, direc, task_list, sample_list, stack=False):
        """
        Initialize the dataset with parameters and load data.

        Parameters:
        input_length (int): The length of the input sequence.
        mid (int): The midpoint of the sequence.
        output_length (int): The length of the output sequence.
        direc (str): The directory containing the data files.
        task_list (list): The list of tasks.
        sample_list (list): The list of samples.
        stack (bool): Whether to stack the data.
        """
        self.input_length = input_length
        self.mid = mid
        self.output_length = output_length
        self.direc = direc
        self.task_list = task_list
        self.sample_list = sample_list
        self.stack = stack
        try:
            self.data_lists = [torch.load(f"{self.direc}/raw_data_{idx[0]}_{idx[1]}.pt") for idx in task_list]
        except:
            self.data_lists = [torch.load(f"{self.direc}/raw_data_{idx}.pt") for idx in task_list]

    def __len__(self):
        return len(self.task_list) * len(self.sample_list)

    def __getitem__(self, index):
        task_idx = index // len(self.sample_list)
        sample_idx = index % len(self.sample_list)
        y = self.data_lists[task_idx][(self.sample_list[sample_idx] + self.mid):(self.sample_list[sample_idx] + self.mid + self.output_length)]
        if not self.stack:
            x = self.data_lists[task_idx][(self.mid - self.input_length + self.sample_list[sample_idx]):(self.mid + self.sample_list[sample_idx])]
        else:
            x = self.data_lists[task_idx][(self.mid - self.input_length + self.sample_list[sample_idx]):(self.mid + self.sample_list[sample_idx])].reshape(-1, y.shape[-2], y.shape[-1])
        return x.float(), y.float()