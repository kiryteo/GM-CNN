import math
import numpy as np
import torch
import torch.nn as nn

from .conv_utils import generate_neighborhood, get_nbrhood_elements

class GMConv(nn.Module):
    def __init__(self, group, order, nbr_size, group_matrix, out_channels):
        """
        Initializes a GMConv module.

        Args:
            group (str): The type of group. Can be 'cyclic' or 'dihedral'.
            order (int): The order of the group.
            nbr_size (int): The size of the neighborhood.
            group_matrix (numpy.ndarray): The group matrix.
            out_channels (int): The number of output channels.
        """
        super().__init__()

        self.out_channels = out_channels

        if group == 'cyclic':
            vec_size = order * order
        elif group == 'dihedral':
            vec_size = order * order * 4

        # Generate the neighborhood 
        nbrhood = generate_neighborhood(group, order, nbr_size)

        # Get the target elements via direct product of the neighborhood elements
        target_elements = get_nbrhood_elements(nbrhood)

        # Get the indices of the target elements in the group matrix
        self.index_matrix = np.array([np.where(group_matrix.T == element)[1] for element in target_elements])

        # Initialize the bias
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Initialize the weight coefficients
        self.weight_coeff = nn.Parameter(torch.empty(self.out_channels, 1, len(target_elements)))
        nn.init.kaiming_uniform_(self.weight_coeff, a=math.sqrt(5))

        # Initialize the error vector
        self.err_vector = nn.Parameter(torch.empty(1, vec_size))
        nn.init.kaiming_uniform_(self.err_vector, a=math.sqrt(5))

    def forward(self, x):
        """
        Performs forward pass of the GMConv module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Normalize the error vector
        self.err_vector.data /= torch.norm(self.err_vector.data)

        # Compute the adjusted input tensor and 'scale' it with the error vector
        adj_input_tensor = x[:, :, 0, self.index_matrix] * self.err_vector[None, None, :, :]

        # Normalize the weight coefficients
        self.weight_coeff.data /= torch.norm(self.weight_coeff.data, dim=-1, keepdim=True)
        res = torch.einsum('ojk, bikl -> bojl', self.weight_coeff, adj_input_tensor)

        results = res + self.bias[None, :, None, None]
        return results