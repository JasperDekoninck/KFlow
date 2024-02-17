from .. import base
from .. import operations as ops
import numpy as np

class Loss(base.AdvancedOperation):
    """
    Basic class for a loss function for a neural net.
    """
    def __init__(self, operations, name="loss", add_to_flow=True):
        """
        Initializes a new loss advanced operation
        :param operations: The operations to perform in order in the advanced operation, list of baseElements
        :param name: The name of the loss, string
        :param add_to_flow: Boolean indicating whether or not to add the loss to the flow
        :post the initializer of AdvancedOperation is called.
        """
        base.AdvancedOperation.__init__(self, operations, name=name, add_to_flow=add_to_flow)

    def backward(self):
        """
        Performs a backward operation on the loss function
        :post If the gradient is equal to 0, set the gradient equal to 1.
        :post Calls the backward function of the AdvancedOperation
        """
        if self.get_gradient() == 0:
            self.set_gradient(np.array([1.0]))

        super(Loss, self).backward()

    def __str__(self):
        """
        Returns a string describing the loss function
        :return: "loss operation"
        """
        return "loss operation"


class LossMaker(Loss):
    """
    A class that makes any baseElement whatsoever to a Loss.
    """
    def __init__(self, input_element, name="loss", add_to_flow=True):
        """
        Initializes a new LossMaker
        :param input_element: The element which needs to be the loss
        :param name: The name of the loss, string
        :param add_to_flow: Boolean indicating whether or not to add the loss to the flow
        :post calls the initializer of the loss class with the DoNothing operation
        """
        do_nothing = ops.DoNothing(input_element, add_to_flow=False)
        Loss.__init__(self, [do_nothing], name, add_to_flow)


class Mse(Loss):
    """
    A class implementing the mean squared error loss function.
    """
    def __init__(self, y_true, y_pred, name="mse", add_to_flow=True):
        """
        Initializes a new MSE.
        :param y_true: A baseElement (or scalar) that should have been the predicted value
        :param y_pred: A baseElement (or scalar) that is the predicted value.
        :param name: The name of the loss, string
        :param add_to_flow: Boolean indicating whether or not to add the loss to the flow
        :post Calls the initializer of the loss function with the operations describing the following function:
        Mean((y_true - y_pred) ** 2)
        """
        minus = ops.Subtract(y_pred, y_true, add_to_flow=False)
        square = ops.Power(minus, 2, add_to_flow=False)
        mean = ops.Mean(square, add_to_flow=False)

        operations = [minus, square, mean]

        Loss.__init__(self, operations, name=name, add_to_flow=add_to_flow)


class Mae(Loss):
    """
    A class implementing the mean absolute error loss function.
    """
    def __init__(self, y_true, y_pred, name="mae", add_to_flow=True):
        """
        Initializes a new MAE.
        :param y_true: A baseElement (or scalar) that should have been the predicted value
        :param y_pred: A baseElement (or scalar) that is the predicted value.
        :param name: The name of the loss, string
        :param add_to_flow: Boolean indicating whether or not to add the loss to the flow
        :post Calls the initializer of the loss function with the operations describing the following function:
        Mean(|y_true - y_pred|)
        """
        minus = ops.Subtract(y_pred, y_true, add_to_flow=False)
        absol = ops.Abs(minus, add_to_flow=False)
        mean = ops.Mean(absol, add_to_flow=False)

        operations = [minus, absol, mean]

        Loss.__init__(self, operations, name=name, add_to_flow=add_to_flow)


class BinaryCrossentropy(Loss):
    """
    A class implementing the binary cross entropy loss function.
    """
    def __init__(self, y_true, y_pred, name="binary crossentropy", add_to_flow=True, epsilon=1e-7):
        """
        Initializes a new BinaryCrossentropy.
        :param y_true: A baseElement (or scalar) that should have been the predicted value
        :param y_pred: A baseElement (or scalar) that is the predicted value.
        :param name: The name of the loss, string
        :param add_to_flow: Boolean indicating whether or not to add the loss to the flow
        :param epsilon: the epsilon used in the log operation to act as some kind of smoothing
        :post Calls the initializer of the loss function with the operations describing the following function:
        Mean(- y_true * log(y_pred + epsilon) - (1 - y_true) * log(1 - y_pred + epsilon))
        """
        one_minus_x = ops.Subtract(1, y_pred, add_to_flow=False)
        log_one_minus_x = ops.Log(one_minus_x, epsilon=epsilon, add_to_flow=False)
        log_x = ops.Log(y_pred, add_to_flow=False, epsilon=epsilon)
        one_minus_y = ops.Subtract(1, y_true, add_to_flow=False)
        first_mult = ops.Multiply(y_true, log_x, add_to_flow=False)
        second_mult = ops.Multiply(one_minus_y, log_one_minus_x, add_to_flow=False)
        all_ = ops.Add(first_mult, second_mult, add_to_flow=False)
        all_minus_one = ops.Multiply(all_, -1, add_to_flow=False)
        output = ops.Mean(all_minus_one, add_to_flow=False)

        operations = [one_minus_x, log_one_minus_x, log_x, one_minus_y, first_mult, second_mult, all_, all_minus_one,
                      output]
        Loss.__init__(self, operations, name=name, add_to_flow=add_to_flow)


class CategoricalCrossentropy(Loss):
    """
    A class implementing the categorical cross entropy loss function.
    """
    def __init__(self, y_true, y_pred, name="categorical crossentropy", add_to_flow=True, epsilon=1e-7):
        """
        Initializes a new CategoricalCrossentropy.
        :param y_true: A baseElement (or scalar) that should have been the predicted value
        :param y_pred: A baseElement (or scalar) that is the predicted value.
        :param name: The name of the loss, string
        :param add_to_flow: Boolean indicating whether or not to add the loss to the flow
        :param epsilon: the epsilon used in the log operation to act as some kind of smoothing
        :post Calls the initializer of the loss function with the operations describing the following function:
        Mean(- Sum(y_true * log(y_pred + epsilon), axis=1))
        """
        log_y = ops.Log(y_pred, epsilon=epsilon, add_to_flow=False)
        entropy = ops.Multiply(y_true, log_y, add_to_flow=False)
        minus_output = ops.Sum(entropy, axis=1, keepdims=True, add_to_flow=False)
        multiply = ops.Multiply(-1, minus_output, add_to_flow=False)
        output = ops.Mean(multiply, axis=0, add_to_flow=False)

        operations = [log_y, entropy, minus_output, multiply, output]

        Loss.__init__(self, operations, name=name, add_to_flow=add_to_flow)
