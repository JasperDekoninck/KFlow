from .. import base
from .. import operations as ops
"""Need to add: getters and setters for all variables of the layer (each object seperately)"""


class Layer(base.AdvancedOperation):
    """The base class of a layer that handles some things like adding batchNorm, dropout and an activation."""
    def __init__(self, operations, activation=None, dropout_rate=0,
                 batch_norm=None, name="layer", add_to_flow=True, training=None):
        """
        Initializes a new layer
        :param operations: The operations the layer currently consists of
        :param activation: The activation to add, if None, no activation will be added
                           (needs to be class of BaseElement)
        :param dropout_rate: Scalar between 0 and 1 indicating the dropout rate for the layer
        :param batch_norm: The batch normalization to add, if None, no activation will be added
                           (needs to be class of BaseElement)
        :param name: The name of the layer, string
        :param add_to_flow: Boolean indicating whether or not to add the layer to the flow
        :param training: Placeholder indicating whether or not the layer is currently training, can only be None
                         if the dropout rate is 0 and batch_norm is None
        :post If batch_norm is not None, adds a batchNormalization layer with the parameters of teh given batch_norm
              class
        :post If activation is not None, adds the activation layer that is given as a parameter
        :post if dropout_rate is not 0, than adds a dropout layer.
        :post Calls the superclass initializer with the given operations, supplemented with the batch normalization,
              activation and dropout.
        """
        assert (dropout_rate == 0 and batch_norm is None and training is None) or training is not None
        if activation is None:
            activation = ops.DoNothing
        if batch_norm is None:
            batch_norm = ops.DoNothing

        batchnorm = batch_norm(operations[-1], training=training, add_to_flow=False)
        activation = activation(batchnorm, add_to_flow=False)
        operations.append(batchnorm)
        operations.append(activation)

        if dropout_rate > 0:
            dropout = ops.Dropout(activation, training, dropout_rate=dropout_rate, add_to_flow=False)
            operations.append(dropout)

        super(Layer, self).__init__(operations, name=name, add_to_flow=add_to_flow)
