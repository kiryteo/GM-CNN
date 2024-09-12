import torch
import torch.nn as nn
from itertools import product
import math

from ..utils import generate_elements
from .pool_utils import get_nbr_elements, get_subgroup, get_subgroup_cosets, get_indices

class GMTransposePool(nn.Module):
    """
    Transpose pooling layer based on subgroup and cosets.
    """

    def __init__(self, group, order, in_channels, out_channels):
        """
        Initializes a GMTransposePool module.

        Args:
            group (str): The type of group. Can be 'cyclic' or 'dihedral'.
            order (int): The order of the group.
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
        """
        super().__init__()

        self.group = group
        self.order = order
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Generate group elements
        elements = generate_elements(self.group, self.order)

        # Generate direct product elements
        direct_prod_elements = [(a, b) for a, b in product(elements, repeat=2)]
        direct_prod_strings = [a.__str__() + b.__str__() for a, b in direct_prod_elements]

        # Get neighborhood elements
        nbr_elements = get_nbr_elements(direct_prod_elements, 4)

        # Get subgroup and cosets
        subgroup = get_subgroup(order)
        cosets = get_subgroup_cosets(subgroup, nbr_elements)

        # Initialize upsample parameters
        self.upsample_params = nn.Parameter(torch.empty(self.in_channels, self.out_channels, 4))
        nn.init.kaiming_uniform_(self.upsample_params, a=math.sqrt(5))

        # Compute and store indices
        self.indices = torch.tensor(get_indices(cosets, direct_prod_strings)).flatten()

    def forward(self, x):
        """
        Performs forward pass of the GMTransposePool module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after transpose pooling.
        """
        # Compute the upsampled result
        res = torch.einsum('bitk, iof -> bofk', x, self.upsample_params)

        # Compute full indices for flattening and indexing
        batch_size, out_channels, _, _ = res.size()
        full_index = torch.arange(batch_size * out_channels).reshape(batch_size, out_channels, 1, 1) * len(self.indices) + self.indices

        # Flatten and index the result
        res_flatten = res.flatten()[full_index.flatten()]

        # Reshape the result to the desired output shape
        res = res_flatten.view(batch_size, out_channels, 1, len(self.indices))

        return res
