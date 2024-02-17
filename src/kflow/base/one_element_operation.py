from .base_operation import BaseOperation

class OneElementOperation(BaseOperation):
    """
    A simple operation on just one basic element.
    """
    def __init__(self, x, shape=None, name="One element operation", add_to_flow=True):
        """
        Initializes a new OneElementOperation
        :param x: The input for the Operation, basic element. If it is a scalar, the function will pretend it is a
                  Constant.
        :param shape: the output shape of the operation, if None it is calculated using the operation of the Operation
        :param name: the name of the operation
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :post Calls the initializer of the superclass
        """
        BaseOperation.__init__(self, x, shape, name, add_to_flow)

    def backward(self):
        """
        Performs a backward pass on the OneElementOperation
        :post adds the gradient of the element to the output of the reversed operation on self.gradient
        """
        gradient_x = self.reversed_operation(self.gradient)
        self.add_gradient_element(self.x, gradient_x)