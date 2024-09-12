import torch.nn as nn
import numpy as np

from gmcnn.gm_convolution.gmconv import GMConvBase
from gmcnn.gm_convolution.gmconv_classification import GMConvCls
from gmcnn.gm_convolution.gmconv_regression import GMConvReg
from gmcnn.gm_pooling.gmpool import GMPool
from utils import generate_elements, get_group_matrix, kronecker_product

class GMCNNBase(nn.Module):
    """
    Base class for GMCNN models.
    Initializes the dataset-specific parameters.
    """
    def __init__(self, config):
        super().__init__()

        self.dataset = config.exp.model.dataset
        self.num_classes = config.exp.model.num_classes

        # Set vec_size and in_channels based on the dataset
        dataset_params = {
            'rot': (784, 1),
            'mnist-bg-rot': (784, 1),
            'mnist-noise': (784, 1),
            'smallnorb': (576, 1),
            'norb': (1024, 1)
        }
        self.vec_size, self.in_channels = dataset_params.get(self.dataset, (1024, 3)) # default cifar10 case

    def forward(self, x):
        """
        Forward pass for the base model.
        Reshapes the input tensor based on the dataset.
        """
        if self.dataset in ['rot', 'mnist', 'norb', 'smallnorb', 'mnist-noise', 'mnist-bg-rot']:
            x = x.view(-1, 1, 1, x.size(-1))
        else:
            x = x.view(-1, x.size(1), 1, x.size(-1) * x.size(-1))
        return x

class GMCNN(GMCNNBase):
    """
    GMCNN model class.
    Initializes the layers and defines the forward pass.
    """
    def __init__(self, config):
        super().__init__(config)

        self.order = config.exp.model.order
        self.group = config.exp.model.group
        self.nbr = config.exp.model.nbr
        self.lr = config.exp.model.lr
        self.dropout_rate = config.exp.model.dropout
        self.blocks = config.exp.model.blocks

        # Generate group elements and matrix
        elements = generate_elements(self.group, self.order)
        M1 = get_group_matrix(elements)
        group_matrix = np.array(kronecker_product(M1, M1))

        # Initialize layers
        self.gm_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.conv1x1_layers = nn.ModuleList()

        in_channels = self.in_channels
        for block in self.blocks:
            for out_channels in block.out_channels:
                self.gm_layers.append(GMConvCls(self.group, self.order, self.nbr, group_matrix, out_channels))
                self.norm_layers.append(nn.LayerNorm([out_channels, 1, self.vec_size], elementwise_affine=False))
                if in_channels != out_channels:
                    self.conv1x1_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))
                in_channels = out_channels

        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.relu = nn.PReLU()
        self.fc1 = nn.Linear(in_channels, self.num_classes)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        """
        Forward pass for the GMCNN model.
        """
        x = super().forward(x)

        idx = 0
        for block_id, block in enumerate(self.blocks):
            res = x
            for _ in range(block.num_layers):
                x = self.gm_layers[idx](x)
                x = self.relu(self.norm_layers[idx](x))
                idx += 1

            if block_id < len(self.conv1x1_layers):
                res = self.conv1x1_layers[block_id](res)
            x = x + res

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logits = self.fc1(x)

        return logits