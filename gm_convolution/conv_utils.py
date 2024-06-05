from group_element.cyclic_group_element import CyclicGroupElement
from group_element.dihedral_group_element import DihedralElement

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
