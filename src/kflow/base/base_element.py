from .session import Session
from .. import base
import kflow

import numpy as np

class BaseElement:
    """
    The superclass for the Constant, Operation, AdvancedOperation, Variable and Placeholder. It includes
    a value, gradient and a shape.
    """
    def __init__(self, name="basic element", add_to_flow=True):
        """
        Initializes a new BaseElement
        :param name: String, The name for the class, e.g. Constant, Variable, Placeholder or any operation.
        :param add_to_flow: Whether or not to add the
        :post If add_to_flow is true, the new Element will be added to the flow of the current open session.
        """
        assert isinstance(name, str)
        assert isinstance(add_to_flow, bool)
        # The name of the element, string
        self.name = name
        if add_to_flow:
            # If the element is added to a session, its name can change by adding a suffix at the end
            Session.open_session.add_to_flow(self, self.name)

        # The value the element holds, Float or np.array
        self.value = None
        # The gradient of the element, Float or np.array
        self.gradient = 0
        # Shape is the shape of the output, tuple
        self.shape = None

    def forward(self):
        """
        Executes a forward pass through the element. For this basic element, this only includes resetting the
        gradient to zero.
        """
        self.gradient = 0

    def backward(self):
        """
        Executes a backward pass through the element.
        """
        pass

    def get_value(self):
        """
        :return: The value the element is currently holding
        """
        return self.value

    def set_value(self, value):
        """
        :param value: The new value of the element
        :post Sets the value of the element to this new value
        """
        self.value = value

    def add_to_value(self, add_value):
        """
        Adds the given value to the value of the element.
        """
        self.value += add_value

    def get_gradient(self):
        """
        Returns the gradient of the element
        """
        return self.gradient

    def set_gradient(self, gradient):
        """
        Sets the gradient of the element to the given gradient
        """
        if np.isscalar(gradient):
            self.gradient = float(gradient)
        else:
            self.gradient = gradient.astype(np.float64)

    def add_to_gradient(self, add_gradient):
        """
        Adds the given gradient to the gradient of the element.
        """
        self.gradient += add_gradient

    def get_shape(self):
        """
        Returns the shape of the operation
        :return: the shape of ther operation
        """
        return self.shape

    def __eq__(self, other):
        """
        Checks whether two Basic Elements are the same
        :param other: another object
        :return: True if the other object is a BaseElement and if the names match
        """
        if isinstance(other, BaseElement):
            return other.name == self.name
        return False

    def __add__(self, other):
        """
        Adds two BaseElements, or adds a BaseElement with a scalar
        :param other: BaseElement or scalar
        :return: An operation that adds the two together
        """
        if isinstance(other, BaseElement) or np.isscalar(other):
            return kflow.operations.Add(self, other, name="add")
        else:
            raise ValueError("Trying to add something to a BaseElement.")

    def __iadd__(self, other):
        """
        Adds two BaseElements, or adds a BaseElement with a scalar
        :param other: BaseElement or scalar
        :return: An operation that adds the two together
        """
        return self.__add__(other)

    def __radd__(self, other):
        """
        Adds two BaseElements, or adds a BaseElement with a scalar
        :param other: BaseElement or scalar
        :return: An operation that adds the two together
        """
        return self.__add__(other)

    def __mul__(self, other):
        """
        Multiplies two BaseElements, or multiplies a BaseElement with a scalar
        :param other: BaseElement or scalar
        :return: An operation that multiplies the two
        """
        if isinstance(other, BaseElement) or np.isscalar(other):
            return kflow.operations.Multiply(self, other, name="multiply")
        else:
            raise ValueError("Trying to multiply something with BaseElement.")

    def __imul__(self, other):
        """
        Multiplies two BaseElements, or multiplies a BaseElement with a scalar
        :param other: BaseElement or scalar
        :return: An operation that multiplies the two
        """
        return self.__mul__(other)

    def __rmul__(self, other):
        """
        Multiplies two BaseElements, or multiplies a BaseElement with a scalar
        :param other: BaseElement or scalar
        :return: An operation that multiplies the two
        """
        return self.__mul__(other)

    def __sub__(self, other):
        """
        Subtracts two BaseElements, or subtracts a BaseElement with a scalar
        :param other: BaseElement or scalar
        :return: An operation that subtracts the two
        """
        if isinstance(other, BaseElement) or np.isscalar(other):
            return kflow.operations.Subtract(self, other, name="subtract")
        else:
            raise ValueError("Trying to subtract something to a BaseElement.")

    def __isub__(self, other):
        """
        Subtracts two BaseElements, or subtracts a BaseElement with a scalar
        :param other: BaseElement or scalar
        :return: An operation that subtracts the two
        """
        return self.__sub__(other)

    def __rsub__(self, other):
        """
        Subtracts two BaseElements, or subtracts a scalar with a BaseElement
        :param other: BaseElement or scalar
        :return: An operation that subtracts the two
        """
        return self.__neg__().__add__(other)

    def __abs__(self):
        """
        Takes the absolute vaue of this BaseElement
        :return: An operation that takes the absolute value of this BaseElement
        """
        return kflow.operations.Abs(self, name="abs")

    def __truediv__(self, other):
        """
        Divides this BaseElement with other two BaseElements, or divides this BaseElement with a scalar
        :param other: BaseElement or scalar
        :return: An operation that divides the two
        """
        if isinstance(other, BaseElement) or np.isscalar(other):
            return kflow.operations.Divide(self, other, name="divide")
        else:
            raise ValueError("Trying to divide something to a BaseElement.")

    def __itruediv__(self, other):
        """
        Divides this BaseElement with other two BaseElements, or divides this BaseElement with a scalar
        :param other: BaseElement or scalar
        :return: An operation that divides the two
        """
        return self.__truediv__(other)

    def __rtruediv__(self, other):
        """
        Divides this BaseElement with other two BaseElements, or divides this BaseElement with a scalar
        :param other: BaseElement or scalar
        :return: An operation that divides the two
        """
        return base.Constant(other).__truediv__(self)

    def __neg__(self):
        """
        Negates this BaseElement
        :return: An operation that negates this BaseElement
        """
        return kflow.operations.Multiply(self, -1, name="negate")

    def __pow__(self, power, modulo=None):
        """
        Takes a power of this BaseElement
        :param power: the power to which to raise this BaseElement, must be a scalar
        :param modulo: Always None
        :return: An operation that takes the power of this BaseElement
        """
        return kflow.operations.Power(self, power, name="power")

    def __ipow__(self, other):
        """
        Takes a power of this BaseElement
        :param power: the power to which to raise this BaseElement, must be a scalar
        :param modulo: Always None
        :return: An operation that takes the power of this BaseElement
        """
        return self.__pow__(other)

    def __rpow__(self, base, modulo=None):
        """
        Takes the exponential of this BaseElement
        :param base: the power to which to raise this BaseElement, must be a scalar
        :param modulo: Always None
        :return: An operation that takes the exp of this BaseElement
        """
        return kflow.operations.Exp(self, base, name="exp")

    def __matmul__(self, other):
        """
        Performs dot product on two BaseElements
        :param other: BaseElement
        :return: An operation that performs the dot product on self and other
        """
        if not isinstance(other, BaseElement):
            raise ValueError("Trying to matmul something to a BaseElement.")
        if len(self.get_shape()) == 2:
            return kflow.operations.Matmul(self, other, name="power")
        else:
            return kflow.operations.Dot(self, other, name="power")

    def __imatmul__(self, other):
        """
        Performs dot product on two BaseElements
        :param other: BaseElement
        :return: An operation that performs the dot product on self and other
        """
        return self.__matmul__(other)