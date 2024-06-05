
import torch
import torch.nn as nn
from itertools import product

from .pool_utils import generate_elements, get_nbr_elements, get_subgroup, get_subgroup_cosets, get_indices

class GMPool(nn.Module):
    """
    Pooling layer based on subgroup and cosets.
    """

    def __init__(self, group, order, pool='max'):
        """
        Initializes a GMPool module.

        Args:
            group (str): The type of group. Can be 'cyclic' or 'dihedral'.
            order (int): The order of the group.
            pool (str): The pooling method to use. Can be 'max' or 'avg'.
        """
        super().__init__()

        self.group = group
        self.order = order
        self.pool = pool

        # Generate the group elements
        group_elements = generate_elements(self.group, self.order)

        # Generate direct product of the group elements
        direct_prod_elements = [(a, b) for a, b in product(group_elements, repeat=2)]
        direct_prod_strings = [a.__str__() + b.__str__() for a, b in direct_prod_elements]

        # Get the neighborhood elements
        neighborhood_elements = get_nbr_elements(direct_prod_elements, 4)

        # Get the subgroup
        subgroup = get_subgroup(self.order)

        # Get the cosets corresponding to the neighborhood elements
        cosets = get_subgroup_cosets(subgroup, neighborhood_elements)

        # Get the indices of the coset elements in the Kronecker product strings
        self.indices = get_indices(cosets, direct_prod_strings)

    def forward(self, x):
        """
        Performs forward pass of the GMPool module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The pooled output tensor.
        """
        # Extract the relevant indices from the input tensor
        pooled_input = x[:, :, :, self.indices]

        # Perform the pooling operation
        if self.pool == 'max':
            pooled_output = torch.max(pooled_input, dim=-2).values
        elif self.pool == 'avg':
            pooled_output = torch.mean(pooled_input, dim=-2)
        
        return pooled_output

