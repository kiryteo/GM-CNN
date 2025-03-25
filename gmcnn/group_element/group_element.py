class GroupElement:
    """
    Abstract base class representing a generic group element.
    """

    def __init__(self, *args):
        """
        Initializes a GroupElement.

        This constructor is meant to be overridden by subclasses
        with specific parameters.
        """
        pass

    def product(self, other):
        """
        Computes the product of this element with another GroupElement.

        Args:
            other (GroupElement): The other GroupElement to multiply with.

        Returns:
            GroupElement: The product of the two elements.
        """
        pass

    def inverse(self):
        """
        Computes the inverse of this GroupElement.

        Returns:
            GroupElement: The inverse of this element.
        """
        pass

    def __str__(self):
        """
        Returns a string representation of the GroupElement.

        Returns:
            str: The string representation.
        """
        pass
