from .group_element import GroupElement

class CyclicGroupElement(GroupElement):
    def __init__(self, k, N):
        self.k = k % N
        self.N = N

    def product(self, other):
        k_hat = (self.k + other.k) % self.N
        return CyclicGroupElement(k_hat, self.N)

    def inverse(self):
        k_inv = (self.N - self.k) % self.N
        return CyclicGroupElement(k_inv, self.N)

    def nbrhood(self, t):
        # select the nbrhood using powers of k^(0 to t) and k(N-t to N)
        pass

    def __str__(self):
        return f'{self.k} (mod {self.N})'
