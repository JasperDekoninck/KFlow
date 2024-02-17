from .. import operations as ops
from .. import base
from ..initializations import *
from . import Layer 


class Dense(Layer):
    """Class implementing the dense layer of the neural network"""
    def __init__(self, x, n_outputs, name="dense", activation=None,
                 weight_initializer=he_normal_initialization, bias_initializer=constant_initialization,
                 weight_regularizer=None, bias_regularizer=None, dropout_rate=0, batch_norm=None, training=None,
                 add_to_flow=True):
        """
        Initializes a new dense layer
        :param x: BaseElement that serves as input to the dense layer.
        :param n_outputs: The number of outputs the dense layer needs to have
        :param name: The name of the layer, string
        :param activation: The activation to add, if None, no activation will be added
                           (needs to be class of BaseElement)
        :param weight_initializer: The initialization function that needs to be used for the weights
        :param bias_initializer: The initialization function that needs to be used for the biases
        :param weight_regularizer: A regularizer class that puts a regularization on the weights
        :param bias_regularizer: A regularizer class that puts a regularization on the biases
        :param dropout_rate: Scalar between 0 and 1 indicating the dropout rate for the layer
        :param batch_norm: The batch normalization to add, if None, no activation will be added
                           (needs to be class of BaseElement)
        :param training: Placeholder indicating whether or not the layer is currently training, can only be None
                         if the dropout rate is 0 and batch_norm is None
        :param add_to_flow: Boolean indicating whether or not to add the layer to the flow
        :post A variable W, the weights, is created for use in the operations of the dense layer
        :post Regularizations for the weights and biases are added
        :post The dense operation is performed with the weights W and the biases b, x @ W + b
        :post The superclass initializer is called with all the operations mentioned before and the parameters
              given in here.
        """

        W = base.Variable(weight_initializer((x.shape[-1], n_outputs)), name="W")
        b = base.Variable(bias_initializer((1, n_outputs)), name="b")
        self.b = b
        self.W = W

        if weight_regularizer is None:
            weight_regularizer = ops.NoRegularization
        if bias_regularizer is None:
            bias_regularizer = ops.NoRegularization

        weight_regul = weight_regularizer(W, add_to_flow=False)
        bias_regul = bias_regularizer(b, add_to_flow=False)

        matmul = ops.Matmul(x, W, add_to_flow=False)
        add = ops.Add(matmul, b, add_to_flow=False)
        operations = [weight_regul, bias_regul, matmul, add]

        Layer.__init__(self, operations, activation, dropout_rate, batch_norm, name, add_to_flow, training)
