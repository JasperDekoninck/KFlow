from .base_operation import BaseOperation
from .constant import Constant
from .base_element import BaseElement

class TwoElementOperation(BaseOperation):
    """
    A simple operation between two basic elements.
    """
    def __init__(self, x, y, shape=None, name="two element operation", add_to_flow=None):
        """
        Initializes a new TwoElementOperation
        :param x: The input for the Operation, basic element. If it is a scalar, the function will pretend it is a
                  Constant.
        :param y: The input for the Operation, basic element. If it is a scalar, the function will pretend it is a
                  Constant.
        :param shape: the output shape of the operation, if None it is calculated using the operation of the Operation
        :param name: the name of the operation
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :post Calls the initializer of the superclass
        :post self.y will hold the given y if it is a basic element otherwise self.y will hold a Constant with value y
        """
        self.y = None
        if y is not None:
            if isinstance(y, BaseElement):
                self.y = y
            else:
                self.y = Constant(y, add_to_flow=False)

        BaseOperation.__init__(self, x, shape, name, add_to_flow)

    def backward(self):
        """
        Performs a backward pass on the TwoElementOperation
        :post adds the gradient to the self.x of the element to the first output of the reversed operation on
              self.gradient
        :post adds the gradient to the self.y of the element to the second output of the reversed operation on
              self.gradient
        """
        gradient_x, gradient_y = self.reversed_operation(self.gradient)
        self.add_gradient_element(self.x, gradient_x)
        self.add_gradient_element(self.y, gradient_y)