import numpy as np
import math
from gmcnn.group_element.dihedral_group_element import DihedralElement
from gmcnn.group_element.cyclic_group_element import CyclicGroupElement

def generate_elements(group, order):
    """
    Generate elements of a specified group and order.
    
    Parameters:
    group (str): The type of group ('dihedral' or 'cyclic').
    order (int): The order of the group.
    
    Returns:
    list: A list of group elements.
    """
    if group == 'dihedral':
        elements_rotations = [DihedralElement(r, 0, order) for r in range(order)]
        elements_flips = [DihedralElement(r, 1, order) for r in range(order)]
        return elements_rotations + elements_flips
    elif group == 'cyclic':
        return [CyclicGroupElement(k, order) for k in range(order)]
    else:
        raise ValueError("Unsupported group type. Supported types: 'dihedral', 'cyclic'.")

def get_group_matrix(elements):
    """
    Generate the group matrix for a list of elements.
    
    Parameters:
    elements (list): A list of group elements.
    
    Returns:
    np.ndarray: A matrix representing the group.
    """
    inverses = [e.inverse() for e in elements]
    mat = np.array([[f'{inverses[k].product(elements[j])}' for j in range(len(elements))] for k in range(len(elements))])
    return mat

def kronecker_product(M1, M2):
    """
    Compute the Kronecker product of two matrices.
    
    Parameters:
    M1 (list): The first matrix.
    M2 (list): The second matrix.
    
    Returns:
    list: The Kronecker product of M1 and M2.
    """
    result = []
    for row1 in M1:
        for row2 in M2:
            result_row = [entry1 + entry2 for entry1 in row1 for entry2 in row2]
            result.append(result_row)
    return result

def generate_neighborhood(group, n, t):
    """
    Generate the neighborhood of elements for a specified group.
    
    Parameters:
    group (str): The type of group ('dihedral' or 'cyclic').
    n (int): The order of the group.
    t (int): The range of the neighborhood.
    
    Returns:
    list: A list of neighborhood elements.
    """
    neighborhood = []
    if group == 'dihedral':
        for i in range(-t, t + 1):
            r_element = DihedralElement(i, 0, n)
            f_element = DihedralElement(i, 1, n)
            neighborhood.append(str(r_element))
            neighborhood.append(str(f_element))
    elif group == 'cyclic':
        for i in range(-t, t + 1):
            element = CyclicGroupElement(i, n)
            neighborhood.append(str(element))
    else:
        raise ValueError("Unsupported group type. Supported types: 'dihedral', 'cyclic'.")
    return neighborhood

def select_elements_around_middle(lst, k):
    """
    Select elements around the middle of a list.
    
    Parameters:
    lst (list): The list of elements.
    k (int): The number of elements to select.
    
    Returns:
    list: A list of selected elements.
    """
    identity_index = 0
    half_k = (k - 1) // 2
    start_index = identity_index - half_k
    end_index = identity_index + half_k + 1
    selected_elements = lst[:end_index] + lst[start_index:]
    return selected_elements

def get_nbrhood_elements(nbrhood):
    """
    Generate neighborhood elements.
    
    Parameters:
    nbrhood (list): The list of neighborhood elements.
    
    Returns:
    list: A list of combined neighborhood elements.
    """
    return [e1 + e2 for e1 in nbrhood for e2 in nbrhood]

def get_nbr_distances(group, order):
    """
    Calculate distances in the neighborhood for a specified group.
    
    Parameters:
    group (str): The type of group ('dihedral' or 'cyclic').
    order (int): The order of the group.
    
    Returns:
    list: A list of distances and their corresponding coordinates.
    """
    if group == 'cyclic':
        center = (order // 2, order // 2)
        valid_range = range(order)
    elif group == 'dihedral':
        center = (order, order)
        valid_range = range(order * 2)
    else:
        raise ValueError("Unsupported group type. Supported types: 'dihedral', 'cyclic'.")
    
    distances = []
    for x in valid_range:
        for y in valid_range:
            distance = math.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            distances.append((distance, (x, y)))
    
    distances.sort()
    return distances