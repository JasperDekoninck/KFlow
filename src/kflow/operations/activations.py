from .. import base
from .. import operations as ops
import numpy as np


class ReLU(base.OneElementOperation):
    """Performs the ReLU activation function on its input"""
    def __init__(self, x, name="ReLU", add_to_flow=True):
        """
        Initializes a new ReLU activation
        :param x: baseElement, the input of the ReLU
        :param name: The name of the operation, string
        :param add_to_flow: Boolean indicating whether or not to add the activation to the flow
        :pre x must be a baseElement
        :post the Initializer of the OneElementOperation is called
        """
        assert isinstance(x, base.baseElement)
        base.OneElementOperation.__init__(self, x, name=name, add_to_flow=add_to_flow)

    def operation(self):
        """
        Performs the operation on the input
        :return: np.maximum(self.x.get_value(), 0)
        """
        return np.maximum(self.x.get_value(), 0)

    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: The partial derivative of the operation to x evaluated in the given gradient
        """
        dx = np.copy(np.broadcast_to(gradient, self.x.shape))
        dx[0 > self.x.get_value()] = 0

        return dx


class Sigmoid(base.OneElementOperation):
    """Performs the sigmoid activation function on its input"""
    def __init__(self, x, name="sigmoid", add_to_flow=True):
        """
        Initializes a new Sigmoid activation
        :param x: baseElement, the input of the Sigmoid
        :param name: The name of the operation, string
        :param add_to_flow: Boolean indicating whether or not to add the activation to the flow
        :pre x must be a baseElement
        :post the Initializer of the OneElementOperation is called
        """
        assert isinstance(x, base.baseElement)
        base.OneElementOperation.__init__(self, x, name=name, add_to_flow=add_to_flow)

    def operation(self):
        """
        Performs the operation on the input
        In order to avoid overflow, the incoming value is clipped!
        :return: 1 / (1 + e ** (-x))
        """
        overflow_manager = np.clip(self.x.get_value(), -500, 500)
        return 1 / (1 + np.exp(-overflow_manager))

    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: The partial derivative of the operation to x evaluated in the given gradient
        """
        op = self.operation()
        dx = gradient * op * (1 - op)
        return dx


class Tanh(base.OneElementOperation):
    """Performs the tanh activation function on its input."""
    def __init__(self, x, name="tanh", add_to_flow=True):
        """
        Initializes a new Tanh activation
        :param x: baseElement, the input of the Tanh
        :param name: The name of the operation, string
        :param add_to_flow: Boolean indicating whether or not to add the activation to the flow
        :pre x must be a baseElement
        :post the Initializer of the OneElementOperation is called
        """
        assert isinstance(x, base.baseElement)
        base.OneElementOperation.__init__(self, x, name=name, add_to_flow=add_to_flow)

    def operation(self):
        """
        Performs the operation on the input
        :return: (e ** x - e ** (-x)) / (e ** x + e ** (-x))
        """
        return (np.exp(self.x.get_value()) - np.exp(-self.x.get_value())) / (
                    np.exp(self.x.get_value()) + np.exp(-self.x.get_value()))

    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: The partial derivative of the operation to x evaluated in the given gradient
        """
        return gradient * 4 / np.square(np.exp(self.x.get_value()) + np.exp(-self.x.get_value()))


class Softmax(base.AdvancedOperation):
    """returns
    Note: softmax can only be used for 2D arrays x: (batch_size, input_size)
    """
    def __init__(self, x, name="softmax", add_to_flow=True):
        """
        Initializes a new Softmax activation
        :param x: baseElement, the input of the Softmax
        :param name: The name of the operation, string
        :param add_to_flow: Boolean indicating whether or not to add the activation to the flow
        :pre x must be a baseElement
        :post the Initializer of the AdvancedOperation is called with the following equation:
              e ** x / sum(e ** x, axis=1)
        Note: we implement a more stable algorithm that does exactly the same though (the exponent is (x - mean(x)))
        """
        assert isinstance(x, base.baseElement)
        mean_value = ops.Mean(x, axis=1, keepdims=True, add_to_flow=False)
        input_exponent = ops.Subtract(x, mean_value, add_to_flow=False)
        exp_x = ops.Exp(input_exponent, add_to_flow=False)
        sum_exp = ops.Sum(exp_x, axis=1, keepdims=True, add_to_flow=False)
        output = ops.Divide(exp_x, sum_exp, add_to_flow=False)
        operations = [mean_value, input_exponent, exp_x, sum_exp, output]

        base.AdvancedOperation.__init__(self, operations, name=name, add_to_flow=add_to_flow)


class ELU(base.OneElementOperation):
    """Performs the ELU activation function on its input"""
    def __init__(self, x, alpha=1.0, name="elu", add_to_flow=True):
        """
        Initializes a new ELU activation
        :param x: baseElement, the input of the ELU
        :param alpha: the smoothing parameter of the ELU function, scalar
        :param name: The name of the operation, string
        :param add_to_flow: Boolean indicating whether or not to add the activation to the flow
        :pre x must be a baseElement
        :post the Initializer of the OneElementOperation is called
        """
        assert isinstance(x, base.baseElement)
        assert np.isscalar(alpha)
        self.alpha = alpha
        base.OneElementOperation.__init__(self, x, name=name, add_to_flow=add_to_flow)

    def operation(self):
        """
        Performs the operation on the input
        :return: The ELU operation, x if x >= 0 else alpha * (e ** x - 1)
        """
        y = self.x.get_value().copy()
        neg_indices = self.x.get_value() < 0
        y[neg_indices] = self.alpha * (np.exp(y[neg_indices]) - 1)
        return y

    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: The partial derivative of the operation to x evaluated in the given gradient
        """
        neg_indices = self.x.get_value() < 0
        dx = np.copy(np.broadcast_to(gradient, self.x.shape))
        dx[neg_indices] *= self.alpha * np.exp(self.x.get_value()[neg_indices])
        return dx


class LeakyReLU(base.OneElementOperation):
    """Performs the Leaky ReLU activation function on its input"""
    def __init__(self, x, alpha=0.01, name="Leaky ReLU", add_to_flow=True):
        """
        Initializes a new LeakyReLU activation
        :param x: baseElement, the input of the LeakyReLU
        :param alpha: the smoothing parameter of the LeakyReLU function, scalar
        :param name: The name of the operation, string
        :param add_to_flow: Boolean indicating whether or not to add the activation to the flow
        :pre x must be a baseElement
        :post the Initializer of the OneElementOperation is called
        """
        assert isinstance(x, base.baseElement)
        assert np.isscalar(alpha)
        self.alpha = alpha
        self.x = x
        base.OneElementOperation.__init__(self, x, name=name, add_to_flow=add_to_flow)

    def operation(self):
        """
        Performs the operation on the input
        :return: The LeakyReLU operation, x if x >= 0 else alpha * x
        """
        y = self.x.get_value().copy()
        neg_indices = self.x.get_value() < 0
        y[neg_indices] = self.alpha * y[neg_indices]
        return y

    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: The partial derivative of the operation to x evaluated in the given gradient
        """
        dx = np.copy(np.broadcast_to(gradient, self.x.shape))
        dx[self.x.get_value() < 0] *= self.alpha
        return dx


class PReLU(base.AdvancedOperation):
    """Performs the PReLU activation function on its input"""
    def __init__(self, x, array=True, initial_value=0, name="prelu",
                 add_to_flow=True):
        """
        Initializes a new PReLU activation
        :param x: baseElement, the input of the PReLU
        :param array: Boolean indicating whether or not to use the array variant of PReLU or not
        :param initial_value: the initial value for the alpha parameter
        :param name: The name of the operation, string
        :param add_to_flow: Boolean indicating whether or not to add the activation to the flow
        :pre x must be a baseElement
        :pre initial_value is a scalar
        :post Creates a variable alpha that is used in the formula of the PReLU acitivation, if array is True, then
              this variable will be an array, otherwise it is a scalar.
        :post the Initializer of AdvancedOperation is called with the following equation:
              max(0, x) + alpha * min(0, x)
        """
        assert isinstance(x, base.baseElement)
        assert np.isscalar(initial_value)
        # initialize the Variable
        if array:
            self.alpha = base.Variable(np.full(x.shape[1:], float(initial_value)), name=name + "/alpha")
        else:
            self.alpha = base.Variable(initial_value, name=name + "/alpha")

        maximum = ops.Maximum(x, 0, add_to_flow=False)
        minimum = ops.Minimum(x, 0, add_to_flow=False)
        multiply = ops.Multiply(self.alpha, minimum, add_to_flow=False)
        add = ops.Add(maximum, multiply, add_to_flow=False)

        operations = [maximum, minimum, multiply, add]

        base.AdvancedOperation.__init__(self, operations, name=name, add_to_flow=add_to_flow)


class Swish(base.AdvancedOperation):
    """Performs the Swish activation function on its input"""
    def __init__(self, x, trainable=True, initial_value=1, name="swish", add_to_flow=True):
        """
        Initializes a new Swish activation
        :param x: baseElement, the input of the Swish
        :param trainable: Boolean indicating whether or not the beta parameter of the swish activation function
                          is trainable.
        :param initial_value: the initial value for the beta parameter
        :param name: The name of the operation, string
        :param add_to_flow: Boolean indicating whether or not to add the activation to the flow
        :pre x must be a baseElement
        :pre initial_value is a scalar
        :post Creates a variable beta that is used in the formula of the Swish acitivation, if trainable is True, then
              this will be a Variable of shape 1, otherwise it is a Constant.
        :post the Initializer of AdvancedOperation is called with the following equation:
              x * Sigmoid(beta * x)
        """
        if trainable:
            self.beta = base.Variable(initial_value=initial_value, name=name + "/Beta")
        else:
            self.beta = base.Constant(initial_value, add_to_flow=False)

        multiply = ops.Multiply(x, self.beta, add_to_flow=False)
        sigmoid = Sigmoid(multiply, add_to_flow=False)
        multiply2 = ops.Multiply(x, sigmoid, add_to_flow=False)

        operations = [multiply, sigmoid, multiply2]

        base.AdvancedOperation.__init__(self, operations, name=name, add_to_flow=add_to_flow)
