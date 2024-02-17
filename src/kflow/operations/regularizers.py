from .. import base
from .. import operations as ops
import numpy as np


class BasicRegularizer(base.AdvancedOperation):
    """Basic class for some regularizers"""
    def __init__(self, operations, name="regularizer", add_to_flow=True):
        """
        Initializes a new BasicRegularizer
        :param operations: The list of operations this advanced operation consists of.
        :param name: The name of the advanced operation
        :param add_to_flow: Boolean indidicating whether or not to add the operation to the flow
        :post The initializer of the superclass is called
        """
        super(BasicRegularizer, self).__init__(operations, name=name, add_to_flow=add_to_flow)

    def backward(self):
        """
        Performs a backward pass on the regularizer
        :post If the gradient is 0, than sets it to 1
        :post calls the backward function of the superclass
        """
        if self.get_gradient() == 0:
            self.set_gradient(np.array([1.0]))
        super(BasicRegularizer, self).backward()


class NoRegularization(base.OneElementOperation):
    """Class that does no regularization whatsoever"""
    def __init__(self, W, name="no regularization", add_to_flow=True, *args, **kwargs):
        """
        Initializes a new NoRegularization
        :param operations: The list of operations this advanced operation consists of.
        :param name: The name of the advanced operation
        :param add_to_flow: Boolean indidicating whether or not to add the operation to the flow
        :post The initializer of the superclass is called
        """
        super(NoRegularization, self).__init__(W, name=name, add_to_flow=add_to_flow)

    def operation(self):
        """
        Gets the output of the regularizer
        :return: An array containing zeros
        """
        return np.zeros(self.x.shape)

    def reversed_operation(self, gradient):
        """
        Gets the gradient of the regularizer
        :return: An array containing zeros
        """
        return np.zeros(self.x.shape)


class L1Regularizer(BasicRegularizer):
    """Class that implements the L1 regularization loss."""
    def __init__(self, W, l1_weight=0.01, name="l1 regularization", add_to_flow=True):
        """
        Initializes a new L1Regularizer
        :param W: The weights on which to put the regularization
        :param l1_weight: A scalar that is used in the computations of the regularizer (lower -> regularization has
                          less effect)
        :param name: The name of the advanced operation
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :post The initializer of the superclass is called with the following function:
              sum(|W|) * l1_weight
        """
        assert np.isscalar(l1_weight)
        abs_ = ops.Abs(W, add_to_flow=False)
        som = ops.Sum(abs_, axis=None, add_to_flow=False)
        output = ops.Multiply(som, l1_weight, add_to_flow=False)

        operations = [abs_, som, output]
        super(L1Regularizer, self).__init__(operations, name=name, add_to_flow=add_to_flow)


class L2Regularizer(BasicRegularizer):
    """Class that implements the L1 regularization loss."""
    def __init__(self, W, l2_weight=0.01, name="l2_regularization", add_to_flow=True):
        """
        Initializes a new L2Regularizer
        :param W: The weights on which to put the regularization
        :param l2_weight: A scalar that is used in the computations of the regularizer (lower -> regularization has
                          less effect)
        :param name: The name of the advanced operation
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :post The initializer of the superclass is called with the following function:
              sum(W ** 2) * l2_weight
        """
        assert np.isscalar(l2_weight)
        square = ops.Power(W, 2, add_to_flow=False)
        som = ops.Sum(square, axis=None, add_to_flow=False)
        output = ops.Multiply(som, l2_weight, add_to_flow=False)

        operations = [square, som, output]
        super(L2Regularizer, self).__init__(operations, name=name, add_to_flow=add_to_flow)

class L1L2Regularizer(BasicRegularizer):
    """Class that implements a combination of the l1 and l2 loss."""

    def __init__(self, W, l1_weight=0.01, l2_weight=0.01, name="l1_l2_regularization", add_to_flow=True):
        """
        Initializes a new L1L2regularizer
        :param W: The weights on which to put the regularization
        :param l1_weight: A scalar that is used in the computations of the l1 regularizer (lower -> regularization has
                          less effect)
        :param l2_weight: A scalar that is used in the computations of the l2 regularizer (lower -> regularization has
                          less effect)
        :param name: The name of the advanced operation
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :post The initializer of the superclass is called with the following function:
              sum(W ** 2) * l2_weight + sum(|W|) * l1_weight
        """

        # NOTE TO SELF: it doesn't matter that the program sets the gradients of the seperate losses to 1, because
        # the add operator only distributes the 1 gradient this operation has to the two.
        l1_loss = L1Regularizer(W, l1_weight, add_to_flow=False)
        l2_loss = L2Regularizer(W, l2_weight, add_to_flow=False)
        output = ops.Add(l1_loss, l2_loss, add_to_flow=False)

        operations = [l1_loss, l2_loss, output]
        super(L1L2Regularizer, self).__init__(operations, name=name, add_to_flow=add_to_flow)

class Dropout(base.OneElementOperation):
    """Implements the dropout operation for a layer."""
    def __init__(self, x, training, dropout_rate=0.5, name="dropout", add_to_flow=True):
        """
        Initializes a new Dropout operation.
        :param x: The input for the operation, must be BaseElement
        :param training: Boolean Placeholder indicating whether or not the neural net is in training phase or in
                         testing phase
        :param dropout_rate: Scalar between 0 and 1 indicating what fraction of the neurons output must be set to 0.
        :param name: The name of the advanced operation
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :post The initializer of the superclass is called
        :post a variable registering the dropout rate, fallout (which neurons were set to 0 in the previous operation)
             , keep rate (contains a fraction indicating exactly how many neurons were kept in the previous
              operation) and training is created.
        """
        assert np.isscalar(dropout_rate) and 0 <= dropout_rate <= 1
        assert isinstance(training, base.Placeholder) and (training.get_value() is None or
                                                           isinstance(training.get_value(), bool))
        assert isinstance(x, base.BaseElement)
        self.dropout_rate = dropout_rate
        self.fallout = None
        self.keep_rate = None
        self.training = training
        super(Dropout, self).__init__(x, name=name, add_to_flow=add_to_flow)

    def operation(self):
        """
        Performs the dropout operation.
        :return: During testing, just returns the value of the input x
        :return: If training is True, selects some of the neurons (the fallout) that will be set to 0 (using the
                 function np.random.binomial), then defines the keep rate as the number of neurons that are kept
                 divided by the total number of zeros. After that, it returns the value of x divided by keep rate
                 multiplied by the fallout.
        """
        z = self.x.get_value().copy()
        if self.training.get_value():
            self.fallout = np.random.binomial(1, 1 - self.dropout_rate, size=self.x.shape[1:])
            # We dont want everything to be zero, otherwise we're going to get ZeroDivisionErrors
            while np.all(self.fallout == 0):
                self.fallout = np.random.binomial(1, 1 - self.dropout_rate, size=self.x.shape[1:])
            self.keep_rate = np.count_nonzero(self.fallout) / np.prod(self.fallout.shape)
            self.fallout = np.expand_dims(self.fallout, axis=0)
            self.fallout = np.broadcast_to(self.fallout, self.x.shape)
            # We scale to let the output match the output during test time
            z *= self.fallout / self.keep_rate

        return z

    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the given gradient
        :param gradient: The gradient on which to perform the reversed operation
        :return: The partial derivative of the operation to x evaluated in the given gradient
        """
        doutput = gradient * self.fallout / self.keep_rate
        return doutput


class BatchNormalization(base.AdvancedOperation):
    """A class that implements the batch normalization operation.
    To increase the stability of a neural network, batch normalization normalizes the output of a previous activation
    layer by subtracting the batch mean and dividing by the batch standard deviation."""
    def __init__(self, x, training, axis=1, momentum=0.9, epsilon=1e-7, name="Batch normalization", add_to_flow=True):
        """
        Initializes a new BatchNormalization.
        :param x: The input for the operation, must be BaseElement
        :param training: Boolean Placeholder indicating whether or not the neural net is in training phase or in
                         testing phase
        :param axis: The axis along which to take the batch normalization
        :param momentum: The momentum used in the batch normalization
        :param epsilon: A smoothing parameter that needs to be used in several functions
        :param name: The name of the advanced operation
        :param add_to_flow: Boolean indicating whether or not to add the operation to the flow
        :post Creates two variables gamma and beta that are respectively initialized with 1 and 0 that are
              used in the formula of the batch normalization
        :post Initializes the variable running mean, keeping track of the running mean of the inputs (base on momentum)
        :post Initializes the variable running variance, keeping track of the running variance of the inputs
              (base on momentum)
        :post Creates the variables training, epsilon and momentum keeping track of their values.
        :post Calls the initializer of the superclass with the following equation:
              (x - mean(x, axis)) / (var(x, axis) + epsilon) ** 0.5 * gamma + beta
        """
        assert isinstance(training, base.Placeholder) and (training.get_value() is None or
                                                           isinstance(training.get_value(), bool))
        assert isinstance(x, base.BaseElement)
        assert np.isscalar(momentum)
        assert np.isscalar(epsilon)
        self.gamma = base.Variable(1, name=name + "/gamma")
        self.beta = base.Variable(0, name=name + "/beta")
        self.x = x

        self.mean = ops.Mean(x, axis=axis, keepdims=True, add_to_flow=False)
        self.variance = ops.Variance(x, axis=axis, add_to_flow=False)
        add = ops.Add(self.variance, epsilon, add_to_flow=False)  # Variance could be 0 and this gives a problem
        divisor = ops.Power(add, 0.5, add_to_flow=False)
        subtract = ops.Subtract(x, self.mean, add_to_flow=False)
        normalized = ops.Divide(subtract, divisor, add_to_flow=False)
        multiply = ops.Multiply(normalized, self.gamma, add_to_flow=False)
        output = ops.Add(multiply, self.beta, add_to_flow=False)

        operations = [self.mean, self.variance, add, divisor, subtract, normalized, multiply, output]

        self.running_mean = 0
        self.running_variance = 0
        self.momentum = momentum
        self.training = training
        self.epsilon = epsilon
        super(BatchNormalization, self).__init__(operations, name=name, add_to_flow=add_to_flow)

    def forward(self):
        """
        Performs a forward operation on the input.
        :post If training is true, than calls the forward operation on each operation and updates the running mean
              and running variance with the following formula:
              new = momentum * old + (1 -momentum) * current_(mean, var)
        :post If training is false, set the value of the operation to:
              gamma * (x - running_mean) / (running_variance + epsilon) + beta
        """

        if self.training.get_value():
            for operat in self.operations:
                operat.forward()
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * np.mean(self.mean.get_value())
            self.running_variance = self.momentum * self.running_variance + \
                                    (1 - self.momentum) * np.mean(self.variance.get_value())
        else:
            self.set_value(self.beta.get_value() + self.gamma.get_value() * (self.x.get_value() - self.running_mean) /
                           (self.running_variance + self.epsilon) ** 0.5)
