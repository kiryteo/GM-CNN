import math

from group_element.cyclic_group_element import CyclicGroupElement
from group_element.dihedral_group_element import DihedralElement

from utils import generate_elements

def generate_neighborhood(group, n, t):
    """
    Generates a neighborhood of group elements.

    Args:
        group (str): The type of group ('dihedral' or 'cyclic').
        n (int): The order of the group.
        t (int): The range of elements to generate around zero.

    Returns:
        list of str: The string representations of the neighborhood elements.
    """
    neighborhood = []

    if group == 'dihedral':
        for i in range(-t, t + 1):
            r_element = DihedralElement(i, 0, n)
            f_element = DihedralElement(i, 1, n)
            neighborhood.append(r_element.__str__())
            neighborhood.append(f_element.__str__())
    elif group == 'cyclic':
        for i in range(-t, t + 1):
            element = CyclicGroupElement(i, n)
            neighborhood.append(element.__str__())

    return neighborhood

def get_nbrhood_elements(nbrhood):
    """
    Computes the direct product of neighborhood elements.

    Args:
        nbrhood (list of str): The neighborhood elements.

    Returns:
        list of str: The direct product of neighborhood elements.
    """
    return [e1 + e2 for e1 in nbrhood for e2 in nbrhood]


def get_neighbor_distances(group, order):
    """
    Computes the Euclidean distances of elements in the group from the center.

    Args:
        group (str): The type of group. Can be 'cyclic' or 'dihedral'.
        order (int): The order of the group.

    Returns:
        list of tuple: A sorted list of tuples containing distances and their corresponding coordinates.
    """
    if group == 'cyclic':
        center = (order // 2, order // 2)
        valid_range = range(order)
    elif group == 'dihedral':
        center = (order, order)
        valid_range = range(order * 2)
    else:
        print('Currently supported groups: dihedral and cyclic')
        return None

    distances = []
    for x in valid_range:
        for y in valid_range:
            distance = math.sqrt((x - center[0])**2 + (y - center[1])**2)
            distances.append((distance, (x, y)))

    # Sort distances
    distances.sort()

    return distances

def get_central_indices(group, order, out_channels):
    """
    Retrieves the central indices based on the sorted distances for the specified group.
    Current implementation uses the first `out_channels` elements - providing a single
    element per channel.
    TODO: Add the restricted neighborhood option.

    Args:
        group (str): The type of group. Can be 'cyclic' or 'dihedral'.
        order (int): The order of the group.
        out_channels (int): The number of output channels.

    Returns:
        list of str: The central indices of the group elements.
    """
    # Generate group elements
    elements = generate_elements(group, order)

    # Compute distances from the center
    distances = get_neighbor_distances(group, order)

    # Get the central indices based on the distances
    if group == 'cyclic':
        central_indices = [
            f'{point[0]} (mod {order}) {point[1]} (mod {order})' for _, point in distances[:out_channels]
        ]
    elif group == 'dihedral':
        central_indices = [
            f'{elements[point[0]].__str__()} {elements[point[1]].__str__()}' for _, point in distances[:out_channels]
        ]

    return central_indices
