
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle as pkl

def load_amat_file(file_path):
    data = np.loadtxt(file_path)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


class RotMNIST(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def split_train_val(train_X, train_Y, val_fraction, train_fraction=None):
    """
    Input: training data as a torch.Tensor
    """
    # Shuffle
    idx = np.arange(train_X.shape[0])
    np.random.shuffle(idx)
    train_X = train_X[idx,:]
    train_Y = train_Y[idx,:]

    # Compute validation set size
    val_size = int(val_fraction*train_X.shape[0])

    # Downsample for sample complexity experiments
    if train_fraction is not None:
        train_size = int(train_fraction*train_X.shape[0])
        assert val_size + train_size <= train_X.shape[0]
    else:
        train_size = train_X.shape[0] - val_size

    # Shuffle X
    idx = np.arange(0, train_X.shape[0])
    np.random.shuffle(idx)

    train_idx = idx[0:train_size]
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

    train_data = pkl.load(open(train_loc, 'rb'))
    train_X = train_data['X']
    train_Y = train_data['Y']
    test_data = pkl.load(open(test_loc, 'rb'))
    test_X = test_data['X']
    test_Y = test_data['Y']

    return torch.FloatTensor(train_X), torch.FloatTensor(train_Y), torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
