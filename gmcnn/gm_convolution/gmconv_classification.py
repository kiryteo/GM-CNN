import torch
import torch.nn as nn
import numpy as np
import math

from conv_utils import get_central_indices
from .gmconv import GMConvBase

class GMConvCls(GMConvBase):
    """
    Group Matrix Convolution (GMConv) layer for classification tasks.
    """
    def __init__(self, group, order, nbr_size, group_matrix, out_channels, error=False):
        """
        Initializes the GMConvClassification layer.

        Args:
            group (str): The group type.
            order (int): The order of the group.
            nbr_size (int): The size of the neighborhood.
            group_matrix (np.ndarray): The group matrix.
            out_channels (int): The number of output channels.
            error (bool): Flag to enable error correction.
        """
        super().__init__(group, order, nbr_size, group_matrix, out_channels, error=error)

        # Conditional initialization of err_params and col_indices if error correction is enabled
        if self.error:
            central_indices = get_central_indices(order, out_channels)
            self.col_indices = [np.where(group_matrix[0] == point)[0][0] for point in central_indices]

            self.err_params = nn.Parameter(torch.empty(out_channels, 1, len(self.target_elements)))
            nn.init.kaiming_uniform_(self.err_params, a=math.sqrt(5))

    def forward(self, x):
        """
        Forward pass for the GMConvClassification layer.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, in_channels, 1, vec_size).

        Returns:
            torch.Tensor: The output tensor after applying the GMConv operation.
        """
        x = super().forward(x)

        # Ensure the input tensor has the expected shape
        if x.dim() != 4 or x.size(2) != 1:
            raise ValueError("Input tensor must have shape (batch_size, in_channels, 1, vec_size)")

        # Adjust input tensor based on the index matrix
        adj_input_tensor = x[:, :, 0, self.index_matrix]
        res = torch.einsum('ojk, bikl -> bojl', self.weight_coeff, adj_input_tensor)

        # Apply error correction if enabled
        if self.error:
            res = self.apply_error_correction(adj_input_tensor, res)

        results = res + self.bias[None, :, None, None]
        return results

    def apply_error_correction(self, adj_input_tensor, res):
        """
        Applies error correction to the convolution result.

        Args:
            adj_input_tensor (torch.Tensor): The adjusted input tensor.
            res (torch.Tensor): The convolution result tensor.

        Returns:
            torch.Tensor: The tensor after applying error correction.
        """
        ip_mod = adj_input_tensor[:, :, :, self.col_indices]
        err_op = torch.einsum('otk, biko -> bto', self.err_params, ip_mod)

        for i, idx in enumerate(self.col_indices):
            res[:, i, :, idx] += err_op[:, :, i]

        return res
