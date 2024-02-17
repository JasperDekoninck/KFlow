from .. import base
import numpy as np


class DoNothing(base.OneElementOperation):
    """
    A simple one element operation that does not perform any operation on its input.
    """
    def __init__(self, x, name="do nothing", add_to_flow=True, *args, **kwargs):
        """
        Initializes a new DoNothing operation
        :param x: The input for the operation, must be BaseElement
        :param name: The name of the operation, string
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :param args: ...
        :param kwargs: ...
        :post calls the Initializer of the OneElementOperation
        """
        base.OneElementOperation.__init__(self, x, name=name, add_to_flow=add_to_flow)
    
    def operation(self):
        """
        Performs the operation on the input
        :return: Returns the value of its input
        """
        return self.x.get_value()
    
    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: gradient
        """
        return gradient


class Add(base.TwoElementOperation):
    """An operation that adds two BaseElements together"""
    def __init__(self, x, y, name="add", add_to_flow=True):
        """
        Initializes a new Add operation
        :param x: The input for the operation, must be BaseElement
        :param y: the second input for the operation, must be BaseElement
        :param name: The name of the operation, string
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :post calls the Initializer of the TwoElementOperation
        """
        base.TwoElementOperation.__init__(self, x, y, name=name, add_to_flow=add_to_flow)
        
    def operation(self):
        """
        Performs the operation on the input
        :return: The sum of the two inputs values
        """
        return np.add(self.x.get_value(), self.y.get_value())
    
    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: gradient, gradient
        """
        return gradient, gradient


class Subtract(base.TwoElementOperation):
    """Performs np.subtract on function e.g. x - y"""
    def __init__(self, x, y, name="subtract", add_to_flow=True):
        """
        Initializes a new Subtract operation
        :param x: The input for the operation, must be BaseElement
        :param y: the second input for the operation, must be BaseElement
        :param name: The name of the operation, string
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :post calls the Initializer of the TwoElementOperation
        """
        base.TwoElementOperation.__init__(self, x, y, name=name, add_to_flow=add_to_flow)
        
    def operation(self):
        """
        Performs the operation on the input
        :return: Returns the value of x subtracted with the value of y
        """
        return np.subtract(self.x.get_value(), self.y.get_value())
    
    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: gradient, -gradient
        """
        return gradient, -gradient


class Dot(base.TwoElementOperation):
    """Calculates the dot product of two numpy arrays"""
    def __init__(self, x, y, name="dot", add_to_flow=True):
        """
        Initializes a new Dot operation
        :param x: The input for the operation, must be BaseElement
        :param y: the second input for the operation, must be BaseElement
        :param name: The name of the operation, string
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :pre The first element of the shape of the second input must be equal to the last element of the shape of
             the first input.
        :pre Both inputs must be BaseElements
        :post calls the Initializer of the TwoElementOperation
        """
        assert isinstance(y, base.BaseElement)
        assert isinstance(x, base.BaseElement)
        assert y.get_shape()[1] == x.get_shape()[-1]
        base.TwoElementOperation.__init__(self, x, y, name=name, add_to_flow=add_to_flow)
        
    def operation(self):
        """
        Performs the operation on the input
        :return: Returns the dot product of x and y
        """
        return np.dot(self.x.get_value(), self.y.get_value())
    
    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: The partial derivative of the operation to x, the partial derivative of the operation to y
                 evaluated in the given gradient
        """
        dx = gradient.dot(self.y.get_value().T)
        dy = self.x.get_value().T.dot(gradient)
        return dx, dy


class Matmul(base.TwoElementOperation):
    def __init__(self, x, y, name="dot", add_to_flow=True):
        """
        Initializes a new Matmul operation, only works for arrays of len(shape)=2, but a bit faster than dot.
        :param x: The input for the operation, must be BaseElement
        :param y: the second input for the operation, must be BaseElement
        :param name: The name of the operation, string
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :pre The length of both the shapes of the x input as the y input must be 2
        :pre The first element of the shape of the second input must be equal to the last element of the shape of
             the first input.
        :pre Both inputs must be BaseElements
        :post calls the Initializer of the TwoElementOperation
        """
        assert isinstance(y, base.BaseElement)
        assert isinstance(x, base.BaseElement)
        assert len(x.get_shape()) == 2
        assert len(y.get_shape()) == 2
        assert y.get_shape()[0] == x.get_shape()[1]
        base.TwoElementOperation.__init__(self, x, y, name=name, add_to_flow=add_to_flow)
        
    def operation(self):
        """
        Performs the operation on the input
        :return: Returns the matrix multiplication of x and y
        """
        return np.matmul(self.x.get_value(), self.y.get_value())
    
    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: The partial derivative of the operation to x, the partial derivative of the operation to y
                 evaluated in the given gradient
        """
        dx = np.matmul(gradient, self.y.get_value().T)
        dy = np.matmul(self.x.get_value().T, gradient)
        return dx, dy


class Maximum(base.TwoElementOperation):
    """Performs element wise max operation"""
    def __init__(self, x, y, name="maximum", add_to_flow=True):
        """
        Initializes a new Maximum operation
        :param x: The input for the operation, must be BaseElement
        :param y: the second input for the operation, must be BaseElement
        :param name: The name of the operation, string
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :post calls the Initializer of the TwoElementOperation
        """
        base.TwoElementOperation.__init__(self, x, y, name=name, add_to_flow=add_to_flow)
        
    def operation(self):
        """
        Performs the operation on the input
        :return: Returns the element wise max operation
        """
        return np.maximum(self.x.get_value(), self.y.get_value())

    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: The partial derivative of the operation to x, the partial derivative of the operation to y
                 evaluated in the given gradient
        """
        value = self.x.get_value() - self.y.get_value()
        grad = np.broadcast_to(gradient, self.x.get_shape())
        dx = np.copy(grad)
        dx[value < 0] = 0

        dy = np.copy(grad)
        dy[value >= 0] = 0

        return dx, dy


class Minimum(base.TwoElementOperation):
    """Performs element wise min operation"""
    def __init__(self, x, y, name="minimum", add_to_flow=True):
        """
        Initializes a new Minimum operation
        :param x: The input for the operation, must be BaseElement or scalar
        :param y: the second input for the operation, must be BaseElement or scalar
        :param name: The name of the operation, string
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :post calls the Initializer of the TwoElementOperation
        """
        base.TwoElementOperation.__init__(self, x, y, name=name, add_to_flow=add_to_flow)
        
    def operation(self):
        """
        Performs the operation on the input
        :return: Returns the element wise min operation in the values of x and y
        """
        return np.minimum(self.x.get_value(), self.y.get_value())

    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: The partial derivative of the operation to x, the partial derivative of the operation to y
                 evaluated in the given gradient
        """
        value = self.x.get_value() - self.y.get_value()
        dx = np.copy(gradient)
        dx[value > 0] = 0

        dy = np.copy(gradient)
        dy[value <= 0] = 0

        return dx, dy


class Power(base.OneElementOperation):
    """returns x ** pow"""
    def __init__(self, x, power, name="power", add_to_flow=True):
        """
        Initializes a new Power operation
        :param x: The input for the operation, must be BaseElement
        :param name: The name of the operation, string
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :pre The given power must be an integer
        :post calls the Initializer of the OneElementOperation
        """
        assert np.isscalar(power)
        # Variable registering the power.
        self.power = power
        base.OneElementOperation.__init__(self, x, name=name, add_to_flow=add_to_flow)
        
    def operation(self):
        """
        Performs the operation on the input
        :return: Returns the element wise power of the input
        """
        return np.power(self.x.get_value(), self.power)
    
    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: The partial derivative of the operation to x evaluated in the given gradient
        """
        dx = self.power * np.power(self.x.get_value(), self.power - 1) * gradient
        return dx


class Exp(base.OneElementOperation):
    """returns base ** x"""
    def __init__(self, x, base=np.e, name="exp", add_to_flow=True):
        """
        Initializes a new Exp operation
        :param x: The input for the operation, must be BaseElement
        :param base: The base of the exponential operation, scalar
        :param name: The name of the operation, string
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :pre The given base must be a scalar and must be bigger or equal to 0
        :post calls the Initializer of the OneElementOperation
        """
        assert np.isscalar(base) and base >= 0
        # Variable registering the base.
        self.base = base
        base.OneElementOperation.__init__(self, x, name=name, add_to_flow=add_to_flow)

    def operation(self):
        """
        Performs the operation on the input
        :return: Returns the element wise exponential function of the value of the input with the given base
        """
        return np.power(self.base, self.x.get_value())

    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: The partial derivative of the operation to x evaluated in the given gradient
        """
        return np.power(self.base, self.x.get_value()) * np.log(self.base) * gradient


class Log(base.OneElementOperation):
    """returns log(x + epsilon) / log(base)
     Epsilon is necessary for smoothing"""
    def __init__(self, x, base=np.e, epsilon=1e-7, name="log", add_to_flow=True):
        """
        Initializes a new Exp operation
        :param x: The input for the operation, must be BaseElement
        :param base: The base of the logarithmic operation
        :param epsilon: A smoothing parameter that is necessary when your input contains zeros, scalar
        :param name: The name of the operation, string
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :pre The given base must be a scalar
        :pre the given epsilon must be a scalar
        :post calls the Initializer of the OneElementOperation
        """
        assert np.isscalar(base)
        assert np.isscalar(epsilon)
        # Variable registering the base of the operation
        self.base = base
        # Variable registering the epsilon of the operation
        self.epsilon = epsilon
        base.OneElementOperation.__init__(self, x, name=name, add_to_flow=add_to_flow)

    def operation(self):
        """
        Performs the operation on the input
        :return: Returns the element wise logarithm of the input (+ self.epsilon) in the given base
        """
        return np.log(self.x.get_value() + self.epsilon) / np.log(self.base)

    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: The partial derivative of the operation to x evaluated in the given gradient
        """
        return gradient / ((self.x.get_value() + self.epsilon) * np.log(self.base))


class Multiply(base.TwoElementOperation):
    """Returns element wise multiplication"""
    def __init__(self, x, y, name="multiply", add_to_flow=True):
        """
        Initializes a new Multiply operation
        :param x: The input for the operation, must be BaseElement or scalar
        :param y: the second input for the operation, must be BaseElement or scalar
        :param name: The name of the operation, string
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :post calls the Initializer of the TwoElementOperation
        """
        base.TwoElementOperation.__init__(self, x, y, name=name, add_to_flow=add_to_flow)

    def operation(self):
        """
        Performs the operation on the input
        :return: Returns the element wise multiplication of x and y
        """
        return np.multiply(self.x.get_value(), self.y.get_value())

    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: The partial derivative of the operation to x, the partial derivative of the operation to y
                 evaluated in the given gradient
        """
        dx = gradient * self.y.get_value()
        dy = gradient * self.x.get_value()
        return dx, dy


class Divide(base.TwoElementOperation):
    """returns element wise division: x/(y+epsilon)"""
    def __init__(self, x, y, epsilon=1e-7, name="divide", add_to_flow=True):
        """
        Initializes a new Divide operation
        :param x: The input for the operation, must be BaseElement
        :param y: the second input for the operation, must be BaseElement
        :param epsilon: A smoothing parameter that is necessary when your input contains zeros, scalar
        :param name: The name of the operation, string
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :pre the given epsilon must be a scalar
        :post calls the Initializer of the TwoElementOperation
        """
        assert np.isscalar(epsilon)
        # Variable registering the epsilon
        self.epsilon = epsilon
        base.TwoElementOperation.__init__(self, x, y, name=name, add_to_flow=add_to_flow)

    def operation(self):
        """
        Performs the operation on the input
        :return: Returns the element wise division of x with y + self.epsilon
        """
        return np.divide(self.x.get_value(), self.y.get_value() + self.epsilon)

    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: The partial derivative of the operation to x, the partial derivative of the operation to y
                 evaluated in the given gradient
        """
        dx = gradient / (self.y.get_value() + self.epsilon)
        dy = - gradient * self.x.get_value() / (np.square(self.y.get_value() + self.epsilon))
        return dx, dy


class Abs(base.OneElementOperation):
    def __init__(self, x, name="abs", add_to_flow=True):
        """
        Initializes a new Abs operation
        :param x: The input for the operation, must be BaseElement or scalar
        :param name: The name of the operation, string
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :post calls the Initializer of the OneElementOperation
        """
        base.OneElementOperation.__init__(self, x, name=name, add_to_flow=add_to_flow)

    def operation(self):
        """
        Performs the operation on the input
        :return: Returns the absolute value of the value of x
        """
        return np.abs(self.x.get_value())

    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: The partial derivative of the operation to x evaluated in the given gradient
        """
        return np.sign(self.x.get_value()) * gradient


class Sum(base.OneElementOperation):
    """returns the sum of x along a certain axis"""
    def __init__(self, x, axis=None, keepdims=False, add_to_flow=True, name="sum"):
        """
        Initializes a new Sum operation
        :param x: The input for the operation, must be BaseElement
        :param name: The name of the operation, string
        :param axis: The axis along which the sum operation is going to be performed, integer
        :param keepdims: boolean indicating whether or not to keep the dimension along the axis.
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :pre keepdims must be a boolean
        :pre x must be a BaseElement
        :pre axis must be an int and must be lower than the length of the shape of x or it must be None
        :post calls the Initializer of the OneElementOperation
        """
        assert isinstance(x, base.BaseElement)
        assert isinstance(keepdims, bool)
        assert axis is None or (isinstance(axis, int) and 0 <= axis < len(x.get_shape()))
        self.axis = axis
        self.keepdims = keepdims
        base.OneElementOperation.__init__(self, x, name=name, add_to_flow=add_to_flow)

    def operation(self):
        """
        Performs the operation on the input
        :return: Returns the sum of x along the given axis
        """
        output = np.sum(self.x.get_value(), axis=self.axis, keepdims=self.keepdims)
        if np.isscalar(output):
            output = np.array([output])
        return output

    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: The partial derivative of the operation to x evaluated in the given gradient
        """
        if self.axis is not None:
            dx = np.broadcast_to(gradient, self.x.get_shape())
        else:
            dx = gradient * np.ones(self.x.get_shape())

        return dx


class Mean(base.OneElementOperation):
    """returns the mean of x along a certain axis"""
    def __init__(self, x, axis=None, keepdims=False, name="mean", add_to_flow=True):
        """
        Initializes a new Mean operation
        :param x: The input for the operation, must be BaseElement
        :param name: The name of the operation, string
        :param axis: The axis along which the mean operation is going to be performed, integer
        :param keepdims: boolean indicating whether or not to keep the dimension along the axis.
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :pre keepdims must be a boolean
        :pre x must be a BaseElement
        :pre axis must be an int and must be lower than the length of the shape of x or it is None
        :post calls the Initializer of the OneElementOperation
        """
        assert isinstance(x, base.BaseElement)
        assert isinstance(keepdims, bool)
        assert axis is None or (isinstance(axis, int) and 0 <= axis < len(x.get_shape()))
        self.axis = axis
        self.keepdims = keepdims
        base.OneElementOperation.__init__(self, x, name=name, add_to_flow=add_to_flow)

    def operation(self):
        """
        Performs the operation on the input
        :return: Returns the mean of x along the given axis
        """
        output = np.mean(self.x.get_value(), axis=self.axis, keepdims=self.keepdims)
        if np.isscalar(output):
            output = np.array([output])
        return output

    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: The partial derivative of the operation to x evaluated in the given gradient
        """
        if self.axis is not None:
            dx = np.broadcast_to(gradient, self.x.get_shape()) / self.x.get_shape()[self.axis]
        else:
            dx = gradient * np.ones(self.x.get_shape()) / np.prod(self.x.get_shape())

        return dx


class BroadcastTo(base.OneElementOperation):
    """Broadcasts x into shape"""
    def __init__(self, x, shape, name="broadcast_to", add_to_flow=True):
        """
        Initializes a new BroadcastTo operation
        :param x: The input for the operation, must be BaseElement
        :param name: The name of the operation, string
        :param shape: The shape to which the input must be broadcasted
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :pre x must be a BaseElement
        :pre shape must be list or tuple containing only integers
        :post calls the Initializer of the OneElementOperation
        """
        assert isinstance(x, base.BaseElement)
        assert isinstance(shape, (list, tuple))
        assert np.all([isinstance(i, int) for i in shape])
        # Variable registering the output shape
        self.shape_broadcast = shape
        base.OneElementOperation.__init__(self, x, name=name, add_to_flow=add_to_flow)

    def operation(self):
        """
        Performs the operation on the input
        :return: Returns the input x broadcasted to the given shape
        """
        return np.broadcast_to(self.x.get_value(), self.shape_broadcast)

    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: The partial derivative of the operation to x evaluated in the given gradient
        """
        for axis in range(len(gradient.shape)):
            if len(self.x.get_shape()) <= axis:
                gradient = gradient.sum(axis=axis, keepdims=False)
            elif self.x.get_shape()[axis] == 1:
                gradient = gradient.sum(axis=axis)

        return gradient


class Newaxis(base.OneElementOperation):
    """Adds axis of size 1 in place you want (axis)
    e.g. x.shape = (3, 2, 3) and axis=1 => output.shape = (3, 1, 2, 3)
    """
    def __init__(self, x, axis=-1, name="newaxis", add_to_flow=True):
        """
        Initializes a new Newaxis operation
        :param x: The input for the operation, must be BaseElement
        :param name: The name of the operation, string
        :param axis: The axis along which the mean operation is going to be performed, integer
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :pre x must be a BaseElement
        :pre axis must be an integer
        :post calls the Initializer of the OneElementOperation
        """
        assert isinstance(x, base.BaseElement)
        assert isinstance(axis, int)
        # Variable registering the axis
        self.axis = axis
        base.OneElementOperation.__init__(self, x, name=name, add_to_flow=add_to_flow)

    def operation(self):
        """
        Performs the operation on the input
        :return: Returns the input axis with a new axis at the given axis position
        """
        return np.expand_dims(self.x.get_value(), self.axis)

    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: The partial derivative of the operation to x evaluated in the given gradient
        """
        return gradient.reshape(self.x.get_shape())


class Reshape(base.OneElementOperation):
    def __init__(self, x, shape, name="reshape", add_to_flow=True):
        """
        Initializes a new Reshape operation
        :param x: The input for the operation, must be BaseElement
        :param name: The name of the operation, string
        :param shape: The shape to which the input must be reshaped
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :pre x must be a BaseElement
        :pre shape must be list or tuple containing only integers
        :post calls the Initializer of the OneElementOperation
        """
        assert isinstance(x, base.BaseElement)
        assert isinstance(shape, (list, tuple))
        assert np.all([isinstance(i, int) for i in shape])
        # Variable registering the output shape
        self.output_shape = shape
        base.OneElementOperation.__init__(self, x, name=name, add_to_flow=add_to_flow)

    def operation(self):
        """
        Performs the operation on the input
        :return: Returns the reshaped input x
        """
        return self.x.get_value().reshape(self.output_shape)

    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: The partial derivative of the operation to x evaluated in the given gradient
        """
        return gradient.reshape(self.x.get_shape())


class Flatten(base.OneElementOperation):
    """flattens array into shape (batch_size, inputs)"""
    def __init__(self, x, name="flatten", add_to_flow=True):
        """
        Initializes a new Flatten operation
        :param x: The input for the operation, must be BaseElement
        :param name: The name of the operation, string
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :pre x must be a BaseElement
        :post calls the Initializer of the OneElementOperation
        """
        assert isinstance(x, base.BaseElement)
        base.OneElementOperation.__init__(self, x, name=name, add_to_flow=add_to_flow)

    def operation(self):
        """
        Performs the operation on the input
        :return: Returns the input x flattened to a 2D array
        """
        return self.x.get_value().reshape(self.x.get_shape()[0], np.prod(self.x.get_shape()[1:]))

    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: The partial derivative of the operation to x evaluated in the gradient
        """
        return gradient.reshape(self.x.get_shape())


class Concatenate(base.TwoElementOperation):
    """Concatenates two arrays along certain axis, axis cannot be None"""
    def __init__(self, x, y, axis, name="concatination", add_to_flow=True):
        """
        Initializes a new Divide operation
        :param x: The input for the operation, must be BaseElement
        :param y: the second input for the operation, must be BaseElement
        :param axis: The axis along which to concatenate the two inputs, integer
        :param name: The name of the operation, string
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :pre x and y must be BaseElements
        :pre axis must be an integer
        :post calls the Initializer of the TwoElementOperation
        """
        assert isinstance(x, base.BaseElement) and isinstance(y, base.BaseElement)
        assert isinstance(axis, int)
        # Variable registering the axis.
        self.axis = axis
        base.TwoElementOperation.__init__(self, x, y, name=name, add_to_flow=add_to_flow)

    def operation(self):
        """
        Performs the operation on the input
        :return: Returns the concatenation of x and y along the given axis
        """
        return np.concatenate([self.x.get_value(), self.y.get_value()], axis=self.axis)

    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: The partial derivative of the operation to x, the partial derivative of the operation to y
                 evaluated in the given gradient
        """
        gradients_x, gradients_y = np.split(gradient, [self.x.get_shape()[self.axis]], axis=self.axis)
        return gradients_x, gradients_y


class Pad(base.OneElementOperation):
    """Pads an  image of shape (batch_size, n_channels, width, height) with zeros"""
    def __init__(self, x, pad, name="zero padding", add_to_flow=True):
        """
        Initializes a new Flatten operation
        :param x: The input for the operation, must be BaseElement
        :param pad: Integer that registers how much to pad the input.
        :param name: The name of the operation, string
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :pre x must be a BaseElement
        :pre pad must be an int
        :post calls the Initializer of the OneElementOperation
        """
        assert isinstance(x, base.BaseElement)
        assert isinstance(pad, int)
        # Variable registering the pad.
        self.pad = pad
        base.OneElementOperation.__init__(self, x, name=name, add_to_flow=add_to_flow)

    def operation(self):
        """
        Performs the operation on the input
        :return: Returns the padded input.
        """
        return np.pad(self.x.get_value(), ((0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode="constant",
                      constant_values=0)

    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: The partial derivative of the operation to x evaluated in the given gradient
        """
        return gradient[:, self.pad:-self.pad, self.pad:-self.pad, :]
