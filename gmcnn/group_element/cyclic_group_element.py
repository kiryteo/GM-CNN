from .group_element import GroupElement

class CyclicGroupElement(GroupElement):
    """
    Class representing an element of a cyclic group.

    Attributes:
        k (int): The element value.
        N (int): The order of the cyclic group.
    """

    def __init__(self, k, N):
        """
        Initializes a CyclicGroupElement.

        Args:
            k (int): The element value.
            N (int): The order of the cyclic group.
        """
        self.k = k % N
        self.N = N

    def product(self, other):
        """
        Computes the product of this element with another CyclicGroupElement.

        Args:
            other (CyclicGroupElement): The other CyclicGroupElement to multiply with.

        Returns:
            CyclicGroupElement: The product of the two elements.
        """
        k_hat = (self.k + other.k) % self.N
        return CyclicGroupElement(k_hat, self.N)

    def inverse(self):
        """
        Computes the inverse of this CyclicGroupElement.

        Returns:
            CyclicGroupElement: The inverse of this element.
        """
        k_inv = (self.N - self.k) % self.N
        return CyclicGroupElement(k_inv, self.N)

    def __str__(self):
        """
        Returns a string representation of the CyclicGroupElement.

        Returns:
            str: The string representation.
        """
        return f'{self.k} (mod {self.N})'
