from .base_element import BaseElement
import numpy as np

class AdvancedOperation(BaseElement):
    """
    A Base Class for Advanced Operation. These are operations that consist out of multiple simpeler operations.
    """
    def __init__(self, operations, name="advanced operation", add_to_flow=True):
        """
        Initializes a new AdvancedOperations
        :param operations: The list of operations this advanced operation consists of. Every element of the list
               can be a BaseOperation or an AdvancedOperation
        :param name: The name of the advanced operation
        :param add_to_flow: Boolean indidicating whether or not to add the operation to the flow
        :post The initializer of the superclass is called
        :post The shape of the advanced operation is the shape of the last operation in the given operations.
        """
        super(AdvancedOperation, self).__init__(name, add_to_flow)
        assert isinstance(operations, (list, tuple))
        assert np.all([isinstance(operation, (AdvancedOperation, BaseElement)) for operation in operations])
        self.operations = operations

        self.shape = operations[-1].get_shape()
        self.value = np.zeros(self.shape)

    def __str__(self):
        """
        ...
        :return: 'advanced operation'
        """
        return "advanced operation"

    def forward(self):
        """
        Performs a forward pass on the advanced operation
        :post In turn, calls the forward function on each op the operations in self.operations.
        """
        for operation in self.operations:
            operation.forward()

    def backward(self):
        """
        Performs a backward pass on the advanced operation.
        :post In turn, calls the backward function on each of the operations in self.operations (in reverse order).
        """
        for operation in reversed(self.operations):
            operation.backward()

    def get_gradient(self):
        """
        Gets the gradient of the last operation
        :return: Returns the gradient of the last operation
        """
        gradient = self.operations[-1].get_gradient()
        return gradient

    def set_gradient(self, gradient):
        """
        Sets the gradient of the last operation
        :param gradient: the new gradient for the last operation
        :post Sets the gradient of the last operation to the given gradient
        """
        self.operations[-1].set_gradient(gradient)

    def add_to_gradient(self, add_gradient):
        """
        Adds the given gradient to the last operation
        :param add_gradient: the new gradient for the last operation
        :post Add the given gradient to the gradient of the last operation
        """
        self.operations[-1].add_to_gradient(add_gradient)

    def get_value(self):
        """
        Gets the value of the last operation.
        :return: Returns the value of the last operation
        """
        return self.operations[-1].get_value()

    def set_value(self, value):
        """
        Sets the value of the last operation to the given value
        :param value: The new value for the last operation
        :post Sets the value of the last operation to the given value.
        """
        self.operations[-1].set_value(value)

    def get_shape(self):
        return self.operations[-1].get_shape()
