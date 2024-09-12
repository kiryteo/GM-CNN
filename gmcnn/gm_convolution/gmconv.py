import torch
import torch.nn as nn
import numpy as np
import math

from conv_utils import generate_neighborhood, get_nbrhood_elements
from abc import ABC, abstractmethod

class GMConvBase(nn.Module):
    """
    Base class for Group Matrix Convolution (GMConv) layers.
    Contains common functionality for both classification and regression tasks.
    """
    def __init__(self, group, order, nbr_size, group_matrix, out_channels, error=False):
        """
        Initializes the GMConvBase layer.

        Args:
            group (str): The group type.
            order (int): The order of the group.
            nbr_size (int): The size of the neighborhood.
            group_matrix (np.ndarray): The group matrix.
            out_channels (int): The number of output channels.
        """
        super().__init__()

        self.out_channels = out_channels
        self.error = error

        # Generate the neighborhood
        self.nbrhood = generate_neighborhood(group, order, nbr_size)

        # Get the target elements via direct product of the neighborhood elements
        self.target_elements = get_nbrhood_elements(self.nbrhood)

        # Get the indices of the target elements in the group matrix
        self.index_matrix = np.array([np.where(group_matrix.T == element)[1] for element in self.target_elements])

        # Initialize the bias
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Initialize the weight coefficients
        self.weight_coeff = nn.Parameter(torch.empty(self.out_channels, 1, len(self.target_elements)))
        nn.init.kaiming_uniform_(self.weight_coeff, a=math.sqrt(5))

    @abstractmethod
    def forward(self, x):
        """
        Forward pass for the GMConvBase layer.
        To be overridden by subclasses.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        pass