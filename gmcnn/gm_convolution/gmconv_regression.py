import math
import torch
import torch.nn as nn

from .gmconv import GMConvBase

class GMConvReg(GMConvBase):
    """
    Group Matrix Convolution (GMConv) layer for regression tasks.
    """
    def __init__(self, group, order, nbr_size, group_matrix, out_channels, error=True):
        """
        Initializes the GMConvRegression layer.

        Args:
            group (str): The group type.
            order (int): The order of the group.
            nbr_size (int): The size of the neighborhood.
            group_matrix (numpy.ndarray): The group matrix.
            out_channels (int): The number of output channels.
        """
        super().__init__(group, order, nbr_size, group_matrix, out_channels, error=error)

        # Determine the vector size based on the group type
        if group == 'cyclic':
            vec_size = order * order
        elif group == 'dihedral':
            vec_size = order * order * 4

        if self.error:
            # Initialize the error vector
            self.err_vector = nn.Parameter(torch.empty(1, vec_size))
            nn.init.kaiming_uniform_(self.err_vector, a=math.sqrt(5))

    def forward(self, x):
        """
        Forward pass for the GMConvRegression layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """

        # Ensure the input tensor has the expected shape
        if x.dim() != 4 or x.size(2) != 1:
            raise ValueError("Input tensor must have shape (batch_size, in_channels, 1, vec_size)")

        adj_input_tensor = x[:, :, 0, self.index_matrix]

        if self.error:
            # Normalize the error vector
            self.err_vector.data /= torch.norm(self.err_vector.data)

            # Compute the adjusted input tensor and 'scale' it with the error vector
            adj_input_tensor = adj_input_tensor * self.err_vector[None, None, :, :]

            # Normalize the weight coefficients
            self.weight_coeff.data /= torch.norm(self.weight_coeff.data, dim=-1, keepdim=True) 

        res = torch.einsum('ojk, bikl -> bojl', self.weight_coeff, adj_input_tensor)

        # Add the bias to the result
        results = res + self.bias[None, :, None, None]

        return results