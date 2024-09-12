import torch
import torch.nn as nn
from itertools import product

from ..utils import generate_elements
from pool_utils import get_nbr_elements, get_dihedral_subgroup, get_subgroup, get_subgroup_cosets, get_indices

class GMPool(nn.Module):
    """
    Group Matrix Pooling (GMPool) layer.
    """
    def __init__(self, group, order):
        """
        Initializes the GMPool layer.

        Args:
            group (str): The group type.
            order (int): The order of the group.
        """
        super().__init__()

        self.group = group
        self.order = order

        # Generate elements of the group
        elements = generate_elements(group, order)

        # Generate Kronecker product elements
        kron_elements = [(a, b) for a, b in product(elements, repeat=2)]
        self.kron_prod_strings = [str(a) + str(b) for a, b in kron_elements]

        # Get neighborhood elements
        nbr_elements = get_nbr_elements(kron_elements, 4)

        # Get subgroup based on the group type
        if group == 'dihedral':
            subgroup = get_dihedral_subgroup(order)
        else:
            subgroup = get_subgroup(order)

        # Get cosets of the subgroup
        self.cosets = get_subgroup_cosets(subgroup, nbr_elements)

        # Get indices for pooling
        self.indices = get_indices(self.cosets, self.kron_prod_strings)

    def forward(self, x):
        """
        Forward pass for the GMPool layer.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: The output tensor after applying the pooling operation.
        """
        # Apply pooling operation
        adj_input_tensor = x[:, :, :, self.indices]
        max_pooled_output = torch.max(adj_input_tensor, -2).values
        return max_pooled_output