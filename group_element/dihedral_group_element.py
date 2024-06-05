from .group_element import GroupElement

class DihedralElement(GroupElement):

    def __init__(self, r, f, n):
        self.n = n
        self.r = r % n
        self.f = f % 2

    def product(self, other):
        if other.f == 0:
            return DihedralElement(self.r + other.r, self.f, self.n)
        else:
            return DihedralElement(self.n - self.r + other.r, 1 - self.f, self.n)

    def inverse(self):
        if self.f == 0:
            return DihedralElement(-self.r, 0, self.n)
        else:
            return DihedralElement(self.r, 1, self.n)

    def __str__(self):
        return f'f^{self.f} * r^{self.r}' if self.f else f'r^{self.r}'