
import torch
import torch.nn as nn
from itertools import product

from pool_utils import generate_elements, get_nbr_elements, get_subgroup, get_subgroup_cosets, get_indices

class GMPool(nn.Module):
    def __init__(self, group, order):
        super().__init__()

        #self.group = group
        #self.order = order

        elements = generate_elements(group, order)
        kron_elements = [(a, b) for a, b in product(elements, repeat=2)]
        # self.kron_prod_strings = [a.__str__() + b.__str__() for a, b in product(self.elements, repeat=2)]
        kron_prod_strings = [a.__str__() + b.__str__() for a, b in kron_elements]

        nbr_elements = get_nbr_elements(kron_elements, 4)
        subgroup = get_subgroup(order)
        cosets = get_subgroup_cosets(subgroup, nbr_elements)
        self.indices = get_indices(cosets, kron_prod_strings)

    def forward(self, x):
        #indices = get_indices(self.cosets, self.kron_prod_strings)
        indices = self.indices
        op = x[:, :, :, indices]
        mp = torch.max(op, -2).values
        return mp