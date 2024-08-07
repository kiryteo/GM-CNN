
import torch
import torch.nn as nn
import numpy as np
from itertools import product
import math

from group_element.cyclic_group_element import CyclicGroupElement
from group_element.dihedral_group_element import DihedralElement
from gm_convolution.gmconv import GMConv

from utils import generate_elements, get_group_matrix, kronecker_product


class GMNet(nn.Module):
    def __init__(self, group, order, vec_size):
        super().__init__()
        self.order = order

        elements = generate_elements(group, order)
        M1 = get_group_matrix(elements)
        M2 = get_group_matrix(elements)

        group_matrix = np.array(kronecker_product(M1, M2))

        self.gm1 = GMConv(group, order, 3, group_matrix, 120)
        self.gm2 = GMConv(group, order, 3, group_matrix, 120)

        self.gmdown = GMConv(group, order, 3, group_matrix, 2)

        self.norm1 = nn.LayerNorm([120, 1, vec_size], elementwise_affine=False)

        self.relu = nn.PReLU()

        self.conv1x1_1 = nn.Conv2d(20, 120, kernel_size=1, stride=1, padding=0)

        # self.pool = nn.AdaptiveMaxPool2d((1, 1))
        # self.fc1 = nn.Linear(120, num_classes)

    def forward(self, x):
        # if self.dataset in ['rot', 'mnist', 'norb', 'smallnorb', 'mnist-bg-rot']:
        #     x = x.view(-1, 1, 1, x.size(-1))
        # else:
        #     x = x.view(-1, x.size(1), 1, x.size(-1) * x.size(-1))
        x = x.view(-1, x.size(1), 1, x.size(-1) * x.size(-1))

        res = x

        x = self.gm1(x)
        x = self.relu(self.norm1(x))
        x = x + self.conv1x1_1(res)

        res = x
        x = self.gm2(x)
        x = self.relu(self.norm1(x))
        x = x + res

        x = self.gmdown(x)
        # x = self.pool(x)

        # x = x.view(x.size(0), -1)

        # logits = self.fc1(x)
        logits = x.view(x.size(0), x.size(1), 64, 64)

        return logits