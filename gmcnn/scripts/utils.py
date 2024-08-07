
import numpy as np
import math

from group_element.cyclic_group_element import CyclicGroupElement
from group_element.dihedral_group_element import DihedralElement


def generate_elements(group, order):
    if group == 'dihedral':
        elements_rotations = [DihedralElement(r, 0, order) for r in range(order)]
        elements_flips = [DihedralElement(r, 1, order) for r in range(order)]
        return elements_rotations + elements_flips
    elif group == 'cyclic':
        return [CyclicGroupElement(k, order) for k in range(order)]


def get_group_matrix(elements):
    # Precompute inverses of all elements
    inverses = [e.inverse() for e in elements]

    # Use list comprehensions to build the matrix
    mat = np.array([[f'{inverses[k].product(elements[j])}' for j in range(len(elements))] for k in range(len(elements))])

    return mat

def kronecker_product(M1, M2):

    return [
        [entry1 + entry2 for entry1 in row1 for entry2 in row2]
        for row1 in M1 for row2 in M2
    ]




