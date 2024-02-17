from .. import base
from .. import operations as ops


class Variance(base.AdvancedOperation):
    """Returns the variance of the input x along a certain axis"""
    def __init__(self, x, axis=None, name="variance", add_to_flow=True):
        """
        Initializes a new Variance operation
        :param x: The input, must be a baseElement
        :param axis: The axis along which to compute the variance.
        :param name: The name of the operation, string
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :pre axis must be None or an integer
        :pre x must be a baseElement
        :post Calls the initializer of the superclass with the following operations:
                Mean(Power(Subtract(x, Mean(x), 2))
        """
        assert axis is None or isinstance(axis, int)
        assert isinstance(x, base.baseElement)

        mean = ops.Mean(x, axis=axis, keepdims=True, add_to_flow=False)
        subtract = ops.Subtract(x, mean, add_to_flow=False)
        square = ops.Power(subtract, 2, add_to_flow=False)
        mean2 = ops.Mean(square, axis=axis, keepdims=True, add_to_flow=False)

        operations = [mean, subtract, square, mean2]

        base.AdvancedOperation.__init__(self, operations, name=name, add_to_flow=add_to_flow)


class Std(base.AdvancedOperation):
    """Returns the standard deviation of x along a certain axis"""
    def __init__(self, x, axis=None, name="std", add_to_flow=True):
        """
        Initializes a new Std operation
        :param x: The input, must be a baseElement
        :param axis: The axis along which to compute the variance.
        :param name: The name of the operation, string
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :pre axis must be None or an integer
        :pre x must be a baseElement
        :post Calls the initializer of the superclass with the following operations:
                Power(Variance(x, axis), 0.5)
        """
        var = Variance(x, axis=axis, add_to_flow=False)
        root = ops.Power(var, 0.5, add_to_flow=False)

        operations = [var, root]

        base.AdvancedOperation.__init__(self, operations, name=name, add_to_flow=add_to_flow)
