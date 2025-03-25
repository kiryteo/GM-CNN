from .group_element import GroupElement

class DihedralElement(GroupElement):
    """
    Class representing an element of a dihedral group.

    Attributes:
        r (int): The rotation part of the element.
        f (int): The flip part of the element (0 or 1).
        n (int): The order of the dihedral group.
    """

    def __init__(self, r, f, n):
        """
        Initializes a DihedralElement.

        Args:
            r (int): The rotation part of the element.
            f (int): The flip part of the element (0 or 1).
            n (int): The order of the dihedral group.
        """
        self.n = n
        self.r = r % n
        self.f = f % 2

    def product(self, other):
        """
        Computes the product of this element with another DihedralElement.

        Args:
            other (DihedralElement): The other DihedralElement to multiply with.

        Returns:
            DihedralElement: The product of the two elements.
        """
        if other.f == 0:
            return DihedralElement(self.r + other.r, self.f, self.n)
        else:
            return DihedralElement(self.n - self.r + other.r, 1 - self.f, self.n)

    def inverse(self):
        """
        Computes the inverse of this DihedralElement.

        Returns:
            DihedralElement: The inverse of this element.
        """
        if self.f == 0:
            return DihedralElement(-self.r, 0, self.n)
        else:
            return DihedralElement(self.r, 1, self.n)

    def __str__(self):
        """
        Returns a string representation of the DihedralElement.

        Returns:
            str: The string representation.
        """
        return f'f^{self.f} * r^{self.r}' if self.f else f'r^{self.r}'
