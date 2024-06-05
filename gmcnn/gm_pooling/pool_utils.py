
from itertools import product

from group_element.cyclic_group_element import CyclicGroupElement
from group_element.dihedral_group_element import DihedralElement

    
def get_nbr_elements(kron_elements, k):
    nbr_elements = [kron_elements[0]]

    for i in range(k-1):
        if i % 2 == 0:
            nbr_elements.append(kron_elements[1:][i//2])
        else:
            nbr_elements.append(kron_elements[1:][-(i//2+1)])

    return nbr_elements

def get_generator_elements(order):
    gen1_elements = []
    gen2_elements = []

    for i in range(order):
        if i % 2 == 0:
            gen1_elements.append((CyclicGroupElement(0, order), CyclicGroupElement(i, order)))
            gen2_elements.append((CyclicGroupElement(i, order), CyclicGroupElement(0, order)))

    return gen1_elements, gen2_elements

def get_dihedral_generators(order):
    generators = []
    for i in range(order):
        if i % 2 == 0:
            generators.append(DihedralElement(i, 0, order))
    for i in range(order):
        if i % 2 == 0:
            generators.append(DihedralElement(i, 1, order))

    return generators

def get_subgroup(order):
    subgroup_op = []
    # subgroup_str = []
    gen1_elements, gen2_elements = get_generator_elements(order)

    for e1 in gen1_elements:
        for e2 in gen2_elements:
            subgroup_op.append((e1[0].product(e2[0]), e1[1].product(e2[1])))
            # subgroup_str.append(e1[0].product(e2[0]).__str__() + e1[1].product(e2[1]).__str__())

    return subgroup_op

def get_dihedral_subgroup(order):
    generators = get_dihedral_generators(order)
    return [(a, b) for a, b in product(generators, repeat=2)]

def get_subgroup_cosets(subgroup_op, nbr_elements):
    cosets = []
    for element in nbr_elements:
        coset = []
        for subgroup_element in subgroup_op:
            coset.append((subgroup_element[0].product(element[0]).__str__() + subgroup_element[1].product(element[1]).__str__()))
        cosets.append(coset)

    return cosets

def get_indices(cosets, kron_prod_strings):
    indices = []
    for coset in cosets:
        coset_indices = []
        for element in coset:
            index = kron_prod_strings.index(element)
            coset_indices.append(index)
        indices.append(coset_indices)

    return indices