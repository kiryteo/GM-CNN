
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

def generate_neighborhood(group, n, t):
    neighborhood = []

    if group == 'dihedral':
        for i in range(-t, t + 1):
            r_element = DihedralElement(i, 0, n)
            f_element = DihedralElement(i, 1, n)
            neighborhood.append(r_element.__str__())
            neighborhood.append(f_element.__str__())
    elif group == 'cyclic':
        for i in range(-t, t+1):
            element = CyclicGroupElement(i, n)
            neighborhood.append(element.__str__())

    return neighborhood


def get_nbrhood_elements(nbrhood):
    return [e1 + e2 for e1 in nbrhood for e2 in nbrhood]


def get_nbr_distances(group, order):
    if group == 'cyclic':
        center = (order//2, order//2)
        valid_range = range(order)
    elif group == 'dihedral':
        center = (order, order)
        valid_range = range(order*2)
    else:
        print('Currently supported groups: dihedral, permutation, and cyclic')
        return None

    distances = []
    for x in valid_range:
        for y in valid_range:
            distance = math.sqrt((x - center[0])**2 + (y - center[1])**2)
            distances.append((distance, (x, y)))

    distances.sort()

    return distances

def get_central_indices(group, order, out_channels):
    
    elements = generate_elements(group, order)
    distances = get_nbr_distances(group, order)

    if group == 'cyclic':
        central_indices = [f'{point[0]} (mod {order}){point[1]} (mod {order})' for _, point in distances[:out_channels]]
    elif group == 'dihedral':
        central_indices = [f'{elements[point[0]].__str__()}{elements[point[1]].__str__()}' for _, point in distances[:out_channels]]

    return central_indices