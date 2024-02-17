from .base_element import BaseElement
import numpy as np

class Placeholder(BaseElement):
    """
    Placeholder is a special class that just holds a certain value for a forward pass. This value can always change and
    most of the times, serves as input to a neural network.
    """
    def __init__(self, shape, name="Placeholder", add_to_flow=True):
        """
        Initializes a new placeholder with a given shape, name and whether or not to add it to the flow
        :param shape: The shape of the placeholder, list, tuple or scalar
        :param name: The name of the placeholder, str
        :param add_to_flow: Boolean indicating whether or not to add the placeholder to the flow
        :post The initializer of the basic element is called
        :post The value is initialized with zeros in the shape of the given shape
        """
        assert isinstance(shape, (list, tuple)) or np.isscalar(shape)
        super(Placeholder, self).__init__(name, add_to_flow=add_to_flow)
        if isinstance(shape, (list, tuple)):
            self.shape = shape
        else:
            self.shape = np.array([shape])
        self.value = np.zeros(self.shape)

    def __str__(self):
        """
        Overwrites the __str__ of the object.
        :return: 'placeholder'
        """
        return 'placeholder'