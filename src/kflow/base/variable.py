from .base_element import BaseElement
import numpy as np

class Variable(BaseElement):
    """
    A Variable is special kind of a basic element that can be optimized by an Optimizer to maje the output of the neural
    net as good as possible.
    """
    def __init__(self, initial_value, name="variable"):
        """
        Initializes the variable with a given shape, an initializer function and a name
        :param initial_value: The initial value of the variable
        :param name: The name of the variable
        :post Initializes the value with the output of the initializer(shape).
        """
        super(Variable, self).__init__(name)
        if isinstance(initial_value, type(np.array([]))):
            self.value = initial_value
        else:
            self.value = np.array([initial_value], dtype=np.float64)

        self.shape = self.value.shape

    def __str__(self):
        """
        Overwrites the string function of the object.
        :return: 'variable'
        """
        return 'variable'