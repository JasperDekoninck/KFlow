from .base_element import BaseElement
import numpy as np

class Constant(BaseElement):
    """
    Constant is an element of the computational graph that remains constant during all calculations. Hence, its initial
    value will remain the shape, as does its shape.
    """
    def __init__(self, value, name="constant", add_to_flow=True):
        """
        Initializes a new constant with a given value, name and whether or not to add the constant to the flow.
        :param value: The value of the constant, numpy array or scalar
        :param name: The name of the constant, string
        :param add_to_flow: Boolean indicating whether or not to add the constant to the flow
        :post Calls the initializer of the basic element with the given name and whether or not to add it to the flow
        :post If the given value is a numpy array, the shape of the constant is the shape of the value, otherwise
              the shape will be 1.
        """
        assert np.isscalar(value) or isinstance(value, type(np.array([]))), \
            "{0} isn't a scalar or a numpy array".format(value)

        BaseElement.__init__(self, name, add_to_flow=add_to_flow)
        if isinstance(value, type(np.array([]))):
            self.value = value
        else:
            self.value = np.array([value], dtype=np.float64)

        self.shape = self.value.shape

    def __str__(self):
        """
        Overwrites the string function of the object class
        :return: 'constant'
        """
        return 'constant'