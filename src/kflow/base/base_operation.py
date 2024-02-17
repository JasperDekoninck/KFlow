from .base_element import BaseElement
from .constant import Constant
import numpy as np

class BaseOperation(BaseElement):
    """
    A BaseOperation is the base class for a normal operation on BaseElements.
    """
    def __init__(self, x, shape=None, name="Basic operation", add_to_flow=True):
        """
        Initializes a new BaseOperation
        :param x: The input for the Operation, basic element. If it is a scalar, the function will pretend it is a
                  Constant.
        :param shape: the output shape of the operation, if None it is calculated using the operation of the Operation
        :param name: the name of the operation
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :post the initializer of the BaseElement is called
        :post self.x will hold the given x if it is a basic element otherwise self.x will hold a Constant with value x
        :post self.shape will hold the given shape, if the given shape is None, it will calculate the output shape
              by performing the operation
        """
        super(BaseOperation, self).__init__(name, add_to_flow=add_to_flow)

        # adding functionality in case of integer or float given in as x
        if isinstance(x, BaseElement):
            self.x = x
        else:
            self.x = Constant(x, add_to_flow=False)

        if shape is not None:
            self.shape = shape
        else:
            try:
                self.shape = self.operation().shape
            except TypeError:
                raise ValueError("The forward operation of the operation '{}' "
                                 "should be defined with np.zeros if shape isn't given in as argument"
                                 .format(self.name))

        self.value = np.zeros(self.shape)

    def operation(self):
        """
        Performs the operation on the given input. (self.x and / or self.y)
        :return: Returns the output of the operation
        """
        pass

    def reversed_operation(self, gradient):
        """
        Calculates the partial derivative of the operation and returns the new gradients to the input data.
        :param gradient: The gradient that is set on the operation, aka the gradient that is given to the operation
                         by the output.
        :return: Returns the gradients this operation gives to its inputs.
        """
        pass

    def forward(self):
        """
        Performs a forward pass on the operation
        :post Sets the current gradient to 0
        :post sets the value to the output of the operation.
        """
        self.set_gradient(0)
        self.value = self.operation()
        self.shape = self.value.shape

    def add_gradient_element(self, element, gradient):
        """
        Adds the given gradient to the element, keeping into account that the gradient might be a bit misshaped.
        :param element: BaseElement to which the gradient must be added
        :param gradient: The gradient to be added
        :post If value of the element is a scalar, the gradient that is added to the element is the sum of the input
              gradient along all axis.
        :post Else if the shape of the given gradient matches the shape of the element, the given gradient will be added
              to the gradient of the element
        :post Else if the shapes don't match, some operations will be performed to make the shapes match correctly
              (i.e taking the sum along the axis where the shapes don't match)
        """
        if not np.all(gradient.shape == element.get_shape()):
            for axis in range(len(gradient.shape) - 1, -1, -1):
                try:
                    if gradient.shape[axis] != element.get_shape()[axis]:
                        gradient = gradient.sum(axis=axis, keepdims=True)
                except IndexError:
                    # This can happen when gradient_x has shape (3, 1) and x has shape (3,)
                    gradient = gradient.sum(axis=axis, keepdims=False)
        element.add_to_gradient(gradient)

    def __str__(self):
        """
        ...
        :return: 'operation'
        """
        return 'operation'
    



