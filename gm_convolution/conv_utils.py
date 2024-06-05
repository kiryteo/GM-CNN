
from group_element.cyclic_group_element import CyclicGroupElement
from group_element.dihedral_group_element import DihedralElement

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