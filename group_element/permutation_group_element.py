from group_element.group_element import GroupElement

class PermutationElement(GroupElement):
    def __init__(self, permutation):
        self.permutation = permutation

    def product(self, other):
        product = [self.permutation[i] for i in other.permutation]
        return PermutationElement(product)

    def inverse(self):
        inverse = [self.permutation.index(i) for i in range(len(self.permutation))]
        return PermutationElement(inverse)

    def __str__(self):
        return str(self.permutation)