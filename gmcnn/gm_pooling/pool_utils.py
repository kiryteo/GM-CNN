from itertools import product

from ..group_element.cyclic_group_element import CyclicGroupElement
from ..group_element.dihedral_group_element import DihedralElement

def get_nbr_elements(kron_elements, k):
    """
    Get neighborhood elements from the Kronecker product elements.

    Args:
        kron_elements (list): List of Kronecker product elements.
        k (int): Number of neighborhood elements to retrieve.

    Returns:
        list: List of neighborhood elements.
    """
    nbr_elements = [kron_elements[0]]

    for idx in range(k - 1):
        if idx % 2 == 0:
            nbr_elements.append(kron_elements[1:][idx // 2])
        else:
            nbr_elements.append(kron_elements[1:][-(idx // 2 + 1)])

    return nbr_elements

def get_generator_elements(order):
    """
    Get generator elements for cyclic groups.

    Args:
        order (int): The order of the group.

    Returns:
        tuple: Two lists of generator elements.
    """
    gen1_elements = []
    gen2_elements = []

    for i in range(order):
        if i % 2 == 0:
            gen1_elements.append((CyclicGroupElement(0, order), CyclicGroupElement(i, order)))
            gen2_elements.append((CyclicGroupElement(i, order), CyclicGroupElement(0, order)))

    return gen1_elements, gen2_elements

def get_dihedral_generators(order):
    """
    Get generator elements for dihedral groups.

    Args:
        order (int): The order of the group.

    Returns:
        list: List of generator elements.
    """
    generators = []
    for idx in range(order):
        if idx % 2 == 0:
            generators.append(DihedralElement(idx, 0, order))
    for idx in range(order):
        if idx % 2 == 0:
            generators.append(DihedralElement(idx, 1, order))

    return generators

def get_subgroup(order):
    """
    Get subgroup elements for cyclic groups.

    Args:
        order (int): The order of the group.

    Returns:
        list: List of subgroup elements.
    """
    subgroup_elements = []
    gen1_elements, gen2_elements = get_generator_elements(order)

    for e1 in gen1_elements:
        for e2 in gen2_elements:
            subgroup_elements.append((e1[0].product(e2[0]), e1[1].product(e2[1])))

    return subgroup_elements

def get_dihedral_subgroup(order):
    """
    Get subgroup elements for dihedral groups.

    Args:
        order (int): The order of the group.

    Returns:
        list: List of subgroup elements.
    """
    generators = get_dihedral_generators(order)
    return [(a, b) for a, b in product(generators, repeat=2)]

def get_subgroup_cosets(subgroup_elements, nbr_elements):
    """
    Get cosets of the subgroup.

    Args:
        subgroup_elements (list): List of subgroup elements.
        nbr_elements (list): List of neighborhood elements.

    Returns:
        list: List of cosets.
    """
    cosets = []
    for element in nbr_elements:
        coset = []
        for subgroup_element in subgroup_elements:
            coset.append((subgroup_element[0].product(element[0]).__str__() + 
                          subgroup_element[1].product(element[1]).__str__()))
        cosets.append(coset)

    return cosets

def get_indices(cosets, kron_prod_strings):
    """
    Get indices of coset elements in the Kronecker product strings.

    Args:
        cosets (list): List of cosets.
        kron_prod_strings (list): List of Kronecker product strings.

    Returns:
        list: List of indices.
    """
    indices = []
    for coset in cosets:
        coset_indices = []
        for element in coset:
            index = kron_prod_strings.index(element)
            coset_indices.append(index)
        indices.append(coset_indices)

    return indices