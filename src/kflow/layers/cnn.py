from .. import base
from .. import operations as ops
from ..initializations import he_normal_initialization, constant_initialization
from . import Layer
import numpy as np


class VanillaConvOperation(base.TwoElementOperation):
    """
    Implement the convolutional operation, an operation that is used quite a lot in a CNN.
    Note: this is a vanilla implementation of the convolutional operation, meaning that it is written in easy
    to understand code, but it's extremely computationally inefficient.

    In a convolutional layer, there is an "eye" that slides across an image. It's slides over it with a certain
    size (the kernel size) and the next frame it sees, is some number of pixels to the left or down (the strides).
    For each frame and each channel, the input of shape (kernel_size[0], kernel_size[1]) is multiplied element-wise
    with some weights and then summed. This is what this operation does, but then for an all channels of the input and
    for each frame of these channels. Thus it outputs one channel.
    """
    def __init__(self, x, y, b, strides, name="conv operation", add_to_flow=True):
        """
        Initializes a new convolutional operation.
        :param x: BaseElement, the input of this convolutional operation, must have shape
                 (batch_size, width, height, n_channels)
        :param y: BaseElement, the weights with which to multiply one frame. Must have shape
                 (kernel_size[0], kernel_size[1], n_channels)
        :param b: BaseElement, The bias to add at the end of the multiplication. Must have shape (1,)
        :param strides: The distance between each frame, must be a tuple of two elements: the x-stride and the
               y-stride
        :param name: String, the name of the layer.
        :param add_to_flow: Boolean, boolean indicating whether or not to add the operation to the flow.
        @post Calls the initializer of the TwoElementOperation.
        """
        self.strides = strides
        self.b = b
        super(VanillaConvOperation, self).__init__(x, y, name=name, add_to_flow=add_to_flow)

    def operation(self):
        """
        Performs the forward pass on the operation.
        @post First, some variable registering the kernel size (y.get_shape()[:2]), the output widht and output height
              are created. Then the output of shape (batch_size, output_width, output_height, 1) is initialized with
              zeros.
        :return: The output has shape (batch_size, output_width, output_height, 1).
                 The output at [:, i, j, 0] is calculated as the sum along axis 1, 2 an 3 of the
                 element wise multiplication between
              x[:, i * self.strides[0]: i * self.strides[0] + self.kernel_size[0],
                     j * self.strides[1]: j * self.strides[1] + self.kernel_size[1], :] (this is one "frame" of the
                     input) and the weights plus the bias.
        """
        self.kernel_size = self.y.get_shape()[:2]
        self.output_width = np.int32((self.x.get_shape()[1] - self.kernel_size[0]) / self.strides[0] + 1)
        self.output_height = np.int32((self.x.get_shape()[2] - self.kernel_size[1]) / self.strides[1] + 1)

        w = np.expand_dims(self.y.get_value(), 0)
        output = np.zeros((self.x.get_shape()[0], self.output_width, self.output_height, 1))

        for i in range(self.output_width):
            for j in range(self.output_height):
                x_ = self.x.get_value()[:, i * self.strides[0]: i * self.strides[0] + self.kernel_size[0],
                     j * self.strides[1]: j * self.strides[1] + self.kernel_size[1], :]
                output[:, i, j, 0] = np.sum(np.multiply(x_, w), axis=(1, 2, 3)) + self.b.get_value()

        return output

    def reversed_operation(self, gradient):
        """
        Performs the backward pass on the convolutional operation.
        :param gradient: The gradient on which to perform the reversed operation
        @post Initializes the x gradient and y gradient at 0.
        @post Adds the sum of the parameter gradient along all axes to the gradient of the bias.
        :return: The output consists of the x gradients (the gradient of the inputs) and the y gradients (the
                 gradients of the weights). To calculate this gradient, we loop over i between 0 and output_width
                 and over j between 0 and output height and select the frame for this i and j (see code).
                 To calculate the gradient for the weights, we add something to the gradient that was already
                 calculated, namely the sum along the 0-th axis of the parameter gradient at [:, i, j 0] multiplied
                 element wise with the selected frame. The x gradient at the frame position, is then calculated
                 by computing the element wise multiplication between the gradient at [:, i, j, 0] and the
                 value of the weights.
        """
        gradients_x = np.zeros(self.x.get_shape())
        gradients_y = np.zeros(self.y.get_shape())
        gradients_bias = np.array([gradient.sum()])
        self.b.add_to_gradient(gradients_bias)

        for i in range(self.output_width):
            for j in range(self.output_height):
                x_ = self.x.get_value()[:, i * self.strides[0]: i * self.strides[0] + self.kernel_size[0],
                     j * self.strides[1]: j * self.strides[1] + self.kernel_size[1], :]

                gradients_y += np.sum(gradient[:, i, j, 0][:, np.newaxis, np.newaxis, np.newaxis] * x_, 0)
                gradients_x[:, i * self.strides[0]: i * self.strides[0] + self.kernel_size[0],
                j * self.strides[1]: j * self.strides[1] + self.kernel_size[1], :] = \
                    gradient[:, i, j, 0][:, np.newaxis, np.newaxis, np.newaxis] * self.y.get_value()

        return gradients_x, gradients_y


class VanillaConv2D(Layer):
    """
    Implements the vanilla 2D convolutional layer.
    """
    def __init__(self, x, n_filters, kernel_size, strides=(1, 1), padding=0, activation=None, dropout_rate=0,
                 batch_norm=None, training=None,
                 weight_initializer=he_normal_initialization, bias_initializer=constant_initialization
                 , name="conv2D", add_to_flow=True):
        """
        Initializes a new convolutional layer.
        :param x: BaseElement, the input of this convolutional operation, must have shape
                 (batch_size, width, height, n_channels)
        :param n_filters: The number of filters the output has (output has shape (batch_size, ?, ?, n_filters)
        :param kernel_size: tuple of two ints, The kernel size of the convolutional layer.
        :param strides: tuple of two ints, The strides to use for the convolutional layer.
        :param padding: int, the zero padding to use at the side of the input
        :param activation: Operation, the activation function to use.
        :param dropout_rate: Float,
               The dropout rate for the output of the layer: note this is only added to the end of the
               layer, so it is not possible to add dropout between timesteps.
        :param batch_norm: Basic Element of BatchNormalization or None, Indicates whether or not to add batch
               normalization to the output of the layer.
        :param training: Placeholder, placeholder indicating whether or not the layer is currently training,
                         can only be None if the dropout rate is 0 and batch_norm is None
        :param weight_initializer: initializer function, Serves as initialization of the weights for the matmul with
                                   the input x.
        :param bias_initializer: initializer function. Serves as initialization of the biases.
        :param name: String, the name of the layer.
        :param add_to_flow: Boolean, boolean indicating whether or not to add the operation to the flow.
        @post Creates a variable x_padded (Operation), which is the input x but padded at the sides with the given
              padding size.
        @post Initializes the weights: for each filter a new Variable of size
              (kernel_size[0], kernel_size[1], x.get_shape()[-1]) is created with the given initializer
        @post Initializes the biases: for each filter a new Variable of size
              (1, ) is created with the given initializer
        @post For each filter, performs the VanillaConvOperation on the padded input with the weights and biases
              for that filter.
        @post Concatenates all these outputs of the VanillaConvOperation together to get one output
        @post Calls the initializer of the layer function with all previously mentioned operations.
        """
        operations = []
        weights = []
        biases = []
        self.kernel_size = kernel_size
        self.strides = strides
        self.x = x

        if padding > 0:
            x_padded = ops.Pad(x, pad=padding, add_to_flow=False)
        else:
            x_padded = ops.DoNothing(x, add_to_flow=False)

        for _ in range(n_filters):
            weights.append(base.Variable(weight_initializer((kernel_size[0], kernel_size[1], x.get_shape()[-1])), name="W"))
            biases.append(base.Variable(bias_initializer((1, )), name="b"))

        for i in range(n_filters):
            operations.append(VanillaConvOperation(x_padded, weights[i], biases[i], strides=strides, add_to_flow=False))

        operations.append(ops.Concatenate(operations[0], operations[1], axis=-1, add_to_flow=False))

        for i in range(2, n_filters):
            operations.append(ops.Concatenate(operations[i], operations[-1], axis=-1, add_to_flow=False))

        operations.insert(0, x_padded)

        super(VanillaConv2D, self).__init__(operations, activation=activation, dropout_rate=dropout_rate, batch_norm=batch_norm,
                                training=training, name=name, add_to_flow=add_to_flow)

class ConvOperation(base.TwoElementOperation):
    """
    Implement the convolutional operation, an operation that is used quite a lot in a CNN.
    Note: this operation does not the same thing as the VanillaConvOperation: instead of separating all
          output channels, this operation does this all in one.

    In a convolutional layer, there is an "eye" that slides across an image. It's slides over it with a certain
    size (the kernel size) and the next frame it sees, is some number of pixels to the left or down (the strides).
    For each frame and each channel, the input of shape (kernel_size[0], kernel_size[1]) is multiplied element-wise
    with some weights and then summed. This is what this operation does, but then for an all channels of the input and
    for each frame of these channels.
    """
    def __init__(self, x, W, b, strides, kernel_size, name="conv operation", add_to_flow=True):
        """
        Initializes a new convolutional operation.
        :param x: BaseElement, the input of this convolutional operation, must have shape
                 (batch_size, width, height, n_channels)
        :param W: BaseElement, the weights with which to multiply one frame. Must have shape
                 (n_filters, kernel_size[0], kernel_size[1], n_channels)
        :param b: BaseElement, The bias to add at the end of the multiplication. Must have shape (n_filters,)
        :param strides: The distance between each frame, must be a tuple of two elements: the x-stride and the
               y-stride
        :param name: String, the name of the layer.
        :param add_to_flow: Boolean, boolean indicating whether or not to add the operation to the flow.
        @post Calls the initializer of the TwoElementOperation.
        @post Some variable registering the kernel size, the output width and output height
              are created
        """
        self.x = x
        self.W = W
        self.b = b
        self.strides = strides
        self.kernel_size = kernel_size
        self.n_filters = self.W.get_shape()[0]
        self.output_width = np.int32((self.x.get_shape()[1] - self.kernel_size[0]) / self.strides[0] + 1)
        self.output_height = np.int32((self.x.get_shape()[2] - self.kernel_size[1]) / self.strides[1] + 1)

        super(ConvOperation, self).__init__(x, W, name=name, add_to_flow=add_to_flow)

    def operation(self):
        """
        Performs the forward pass on the operation.
        :return: The output has shape (batch_size, output_width, output_height, n_filters).
                 The output at [:, i, j, :] is calculated as the sum along axis 1 and 2
                 of the element wise multiplication between
              x[:, i * self.strides[0]: i * self.strides[0] + self.kernel_size[0],
                     j * self.strides[1]: j * self.strides[1] + self.kernel_size[1], :] (this is one "frame" of the
                     input) and the weights plus the bias.
        The documentation of this method is quite a bit easier (and less efficient) than the actual implementation
        though, but it does the same thing.
        """
        self.x_value_reshaped = self.im2col()
        weights_reshaped = self.W.get_value().reshape(self.n_filters, -1)
        out = np.matmul(weights_reshaped, self.x_value_reshaped) + self.b.get_value()
        # because it is quite difficult to reshape and stuff, you first need to put n_filters first
        # and then transpose it.
        out = out.reshape(self.n_filters, self.output_height, self.output_width, self.x.get_shape()[0])
        out = out.transpose(3, 1, 2, 0)
        return out

    def reversed_operation(self, gradient):
        """
        Performs the backward pass on the convolutional operation.
        :param gradient: The gradient on which to perform the reversed operation
        @post Adds the sum of the parameter gradient along the first three axes to the gradient of the bias.
        :return: The gradient of the input x and the weights W, the calculation is quite difficult, mostly because
                 of the col2im function, but the only thing this function does is to reshape the column of gradients
                 back to the shape of x in the correct way.
        """
        gradients_bias = gradient.sum(axis=(0, 1, 2))[:, np.newaxis]
        self.b.add_to_gradient(gradients_bias)

        gradients_reshaped = gradient.transpose(3, 1, 2, 0).reshape(self.n_filters, -1)
        dW = np.matmul(gradients_reshaped, self.x_value_reshaped.T)
        dW = dW.reshape(self.W.get_shape())
        WReshape = self.W.get_value().reshape(self.n_filters, -1)
        dx = np.matmul(WReshape.T, gradients_reshaped)
        dx = self.col2im(dx)
        return dx, dW

    def im2col(self):
        """
        This function reshapes the x value such that it becomes possible to do the operation with one big dot product.
        @post Creates an empty array of shape (np.prod(kernel_size) * x.get_shape()[-1],
                                              x.get_shape()[0] * output_width * output_height)
              This is the output shape.
        :return The output at [:, (output_width * i + j) * x.get_shape()[0] : (output_width * i + j + 1) * x.get_shape()[0])
                is equal to the "frame" at that position, thus it gets the value of x at
                [:, i * strides[0]: i * strides[0] + kernel_size[0], j * strides[1]: j * strides[1] + kernel_size[1], :]
                then transposes it and reshapes it in the correct shape.
        """
        reshaped_value = np.zeros((np.prod(self.kernel_size) * self.x.get_shape()[-1],
                                   self.x.get_shape()[0] * self.output_width * self.output_height))

        for i in range(self.output_height):
            for j in range(self.output_width):
                x_ = self.x.get_value()[:, i * self.strides[0]: i * self.strides[0] + self.kernel_size[0],
                                        j * self.strides[1]: j * self.strides[1] + self.kernel_size[1], :]
                # because it is quite difficult to reshape and stuff, you first need to transpose it
                # and then reshape it.
                x_ = x_.T.reshape(np.prod(self.kernel_size) * self.x.get_shape()[-1], self.x.get_shape()[0])
                n = self.output_width * i + j
                reshaped_value[:, n * self.x.get_shape()[0]: (n + 1) * self.x.get_shape()[0]] = x_

        return reshaped_value

    def col2im(self, reshaped_value):
        """
        This function reshapes the reshaped x value to the original shape such that it becomes possible to do
        the operation with one big dot product. Thus it performs the inverse operation of im2col
        :param reshaped_value: The value of the reshaped shape.
        :return: The output at [:, i * self.strides[0]: i * self.strides[0] + self.kernel_size[0],
            j * self.strides[1]: j * self.strides[1] + self.kernel_size[1], :] is calculated by looping over n, with
            j = n % self.output_width and i = n // self.output_width. Then it selects the reshaped_value at
            [:, n * self.x.get_shape()[0]: (n + 1) * self.x.get_shape()[0]], then transposes it, reshapes it to the correct shape
            (self.x.get_shape()[0], self.kernel_size[0], self.kernel_size[1], self.x.get_shape()[-1])
        """
        x_value = np.zeros(self.x.get_shape())

        for n in range(reshaped_value.shape[1] // self.x.get_shape()[0]):
            j = n % self.output_width
            i = n // self.output_width

            reshaped_part_value = reshaped_value[:, n * self.x.get_shape()[0]: (n + 1) * self.x.get_shape()[0]].T
            reshaped_part_value = reshaped_part_value.reshape(self.x.get_shape()[0],
                                                              self.kernel_size[0], self.kernel_size[1],
                                                              self.x.get_shape()[-1])

            x_value[:, i * self.strides[0]: i * self.strides[0] + self.kernel_size[0],
                    j * self.strides[1]: j * self.strides[1] + self.kernel_size[1], :] += reshaped_part_value

        return x_value


class Conv2D(Layer):
    """
    Implements the vanilla 2D convolutional layer.
    """
    def __init__(self, x, n_filters, kernel_size, strides=(1, 1), padding=0, activation=None, dropout_rate=0,
                 batch_norm=None, training=None,
                 weight_initializer=he_normal_initialization, bias_initializer=constant_initialization,
                 name="conv2D", add_to_flow=True):
        """
        Initializes a new convolutional layer.
        :param x: BaseElement, the input of this convolutional operation, must have shape
                 (batch_size, width, height, n_channels)
        :param n_filters: The number of filters the output has (output has shape (batch_size, ?, ?, n_filters)
        :param kernel_size: tuple of two ints, The kernel size of the convolutional layer.
        :param strides: tuple of two ints, The strides to use for the convolutional layer.
        :param padding: int, the zero padding to use at the side of the input
        :param activation: Operation, the activation function to use.
        :param dropout_rate: Float,
               The dropout rate for the output of the layer: note this is only added to the end of the
               layer, so it is not possible to add dropout between timesteps.
        :param batch_norm: Basic Element of BatchNormalization or None, Indicates whether or not to add batch
               normalization to the output of the layer.
        :param training: Placeholder, placeholder indicating whether or not the layer is currently training,
                         can only be None if the dropout rate is 0 and batch_norm is None
        :param weight_initializer: initializer function, Serves as initialization of the weights for the matmul with
                                   the input x.
        :param bias_initializer: initializer function. Serves as initialization of the biases.
        :param name: String, the name of the layer.
        :param add_to_flow: Boolean, boolean indicating whether or not to add the operation to the flow.
        @post Creates a variable x_padded (Operation), which is the input x but padded at the sides with the given
              padding size.
        @post Initializes the weights: a new Variable of size
              (n_filters, kernel_size[0], kernel_size[1], x.get_shape()[-1]) is created with the given initializer
        @post Initializes the biases: a new Variable of size (n_filters, ) is created with the given initializer
        @post Performs the convolutional operation on the padded input with the weights and the biases.
        @post Calls the initializer of the layer function with all previously mentioned operations.
        """
        self.W = base.Variable(weight_initializer((n_filters, kernel_size[0], kernel_size[1], x.get_shape()[-1])))
        self.b = base.Variable(bias_initializer((n_filters, 1)))

        if padding > 0:
            x_padded = ops.Pad(x, pad=padding, add_to_flow=False)
        else:
            x_padded = ops.DoNothing(x, add_to_flow=False)

        output = ConvOperation(x_padded, self.W, self.b, strides, kernel_size, add_to_flow=False)
        operations = [x_padded, output]

        super(Conv2D, self).__init__(operations, activation=activation, dropout_rate=dropout_rate, batch_norm=batch_norm,
                              training=training, name=name, add_to_flow=add_to_flow)


class VanillaMaxPooling(base.OneElementOperation):
    """
    Implements the max pooling layer.
    Note: this implementation is very computationally expensive.
    """
    def __init__(self, x, strides, name="max pooling", add_to_flow=True):
        """
        Initializes a new max pooling layer.
        :param x: BaseElement, the input of this convolutional operation, must have shape
                 (batch_size, width, height, n_channels)
        :param strides: tuple of two ints, The strides to use for the convolutional layer.
        :param name: String, the name of the layer.
        :param add_to_flow: Boolean, boolean indicating whether or not to add the operation to the flow.
        @post Creates variables registering the output width, output height and the strides.
        @post Calls the initializer of the OneElementOperation
        """
        self.output_width = np.int32((np.round(x.get_shape()[1]) / strides[0]))
        self.output_height = np.int32((np.round(x.get_shape()[2]) / strides[1]))
        self.strides = strides
        super(VanillaMaxPooling, self).__init__(x, name=name, add_to_flow=add_to_flow)

    def operation(self):
        """
        Performs the max pooling operation on the input.
        @post Creates an array called mapping that keeps track of where all the maximal elements in the "frames" are.
              This comes in handy for the reversed operation.
        :return: The output is an array of shape (self.x.get_shape()[0], self.output_width, self.output_height,
                 self.x.get_shape()[-1]) where each element is the maximal element of a certain frame. The element at
                 (i, k, m, j) is the maximal element of x at [i, k * self.strides[0]: (k + 1) * self.strides[0],
                 m * self.strides[1]: (m + 1) * self.strides[1], j].
        """
        output = np.zeros((self.x.get_shape()[0], self.output_width, self.output_height, self.x.get_shape()[-1]))
        self.mappings = np.zeros((self.x.get_shape()[0], self.output_width, self.output_height, self.x.get_shape()[-1], 2))

        for i in range(self.x.get_shape()[0]):
            for j in range(self.x.get_shape()[-1]):
                for k in range(self.output_width):
                    for m in range(self.output_height):
                        x_part = self.x.get_value()[i, k * self.strides[0]: (k + 1) * self.strides[0],
                                                m * self.strides[1]: (m + 1) * self.strides[1], j].reshape(self.strides[0],
                                                                                                        self.strides[1])
                        argmax = np.unravel_index(x_part.argmax(), x_part.get_shape())
                        self.mappings[i, k, m, j, :] = np.array([argmax[0] + k * self.strides[0],
                                                                    argmax[1] + m * self.strides[1]])
                        output[i, k, m, j] = x_part[argmax]

        self.mappings = self.mappings.astype(np.int32)
        return output

    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the max pooling.
        :param gradient: The gradient on which to perform the reversed operation
        :return: Returns an array of the x.get_shape(). For the frame at (i, k, m, j), the maximal element gets the gradient
                 corresponding to this frame.
        """
        gradients_x = np.zeros(self.x.get_shape())

        for i in range(self.x.get_shape()[0]):
            for j in range(self.x.get_shape()[-1]):
                for k in range(self.output_width):
                    for m in range(self.output_height):
                        argmax_index = self.mappings[i, k, m, j, :]
                        gradients_x[i, argmax_index[0], argmax_index[1], j] += gradient[i, k, m, j]

        return gradients_x


# TODO: I can't get this working which is really annoying
class MaxPooling(base.OneElementOperation):
    """
    Implements the max pooling layer.
    """
    def __init__(self, x, kernel_size, strides=None, name="max pooling", add_to_flow=True):
        """
        Initializes a new max pooling layer.
        :param x: BaseElement, the input of this convolutional operation, must have shape
                 (batch_size, width, height, n_channels)
        :param strides: tuple of two ints, The strides to use for the convolutional layer.
        :param name: String, the name of the layer.
        :param add_to_flow: Boolean, boolean indicating whether or not to add the operation to the flow.
        @post Creates variables registering the output width, output height and the strides.
        @post Calls the initializer of the OneElementOperation
        """
        self.kernel_size = kernel_size
        if strides is None:
            self.strides = kernel_size
        else:
            self.strides = strides
        self.output_height = np.int32((np.round(x.get_shape()[1]) / self.strides[0]))
        self.output_width = np.int32((np.round(x.get_shape()[2]) / self.strides[1]))

        super(MaxPooling, self).__init__(x, name=name, add_to_flow=add_to_flow)

    def operation(self):
        """
        Performs the max pooling operation on the input.
        :return: The output is an array of shape (self.x.get_shape()[0], self.output_width, self.output_height,
                 self.x.get_shape()[-1]) where each element is the maximal element of a certain frame. The element at
                 (i, k, m, j) is the maximal element of x at [i, k * self.strides[0]: (k + 1) * self.strides[0],
                 m * self.strides[1]: (m + 1) * self.strides[1], j].
        """
        self.X_col = self.im2col(self.x.get_value().transpose(0, 3, 1, 2).reshape(
                        self.x.get_shape()[0] * self.x.get_shape()[3], self.x.get_shape()[1], self.x.get_shape()[2], 1))
        self.max_idx = np.argmax(self.X_col, axis=0)
        out = self.X_col[self.max_idx, range(self.max_idx.size)]
        out = out.reshape(self.output_height, self.output_width, self.x.get_shape()[3], self.x.get_shape()[0])
        out = out.transpose(3, 0, 1, 2)

        return out

    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the max pooling.
        :param gradient: The gradient on which to perform the reversed operation
        :return: Returns an array of the x.get_shape(). For the frame at (i, k, m, j), the maximal element gets the gradient
                 corresponding to this frame.
        """
        dX_col = np.zeros_like(self.X_col)

        dout_flat = gradient.transpose(1, 2, 0, 3).reshape(-1)
        dX_col[self.max_idx, range(self.max_idx.size)] = dout_flat
        dX = self.col2im(dX_col, (self.x.get_shape()[0] * self.x.get_shape()[3], self.x.get_shape()[1], self.x.get_shape()[2], 1))
        dX = dX.reshape((self.x.get_shape()[0], self.x.get_shape()[3], self.x.get_shape()[1], self.x.get_shape()[2])).transpose(0, 2, 3, 1)

        return dX

    def im2col(self, x_value):
        """
        This function reshapes the x value such that it becomes possible to do the operation with one big max operation.
        @post Creates an empty array of shape (np.prod(kernel_size) * x.get_shape()[-1],
                                              x.get_shape()[0] * output_width * output_height)
              This is the output shape.
        :return The output at [:, (output_width * i + j) * x.get_shape()[0] : (output_width * i + j + 1) * x.get_shape()[0])
                is equal to the "frame" at that position, thus it gets the value of x at
                [:, i * strides[0]: i * strides[0] + kernel_size[0], j * strides[1]: j * strides[1] + kernel_size[1], :]
                then transposes it and reshapes it in the correct shape.
        """
        reshaped_value = np.zeros((np.prod(self.kernel_size) * x_value.get_shape()[-1],
                                   x_value.get_shape()[0] * self.output_width * self.output_height))

        for i in range(self.output_height):
            for j in range(self.output_width):
                x_ = x_value[:, i * self.strides[0]: i * self.strides[0] + self.kernel_size[0],
                     j * self.strides[1]: j * self.strides[1] + self.kernel_size[1], :]
                # because it is quite difficult to reshape and stuff, you first need to transpose it
                # and then reshape it.
                x_ = x_.T.reshape(np.prod(self.kernel_size) * x_value.get_shape()[-1], x_value.get_shape()[0])
                n = self.output_width * i + j
                reshaped_value[:, n * x_value.get_shape()[0]: (n + 1) * x_value.get_shape()[0]] = x_.copy()

        return reshaped_value

    def col2im(self, reshaped_value, shape):
        """
        This function reshapes the reshaped x value to the original shape such that it becomes possible to do
        the operation with one big dot product. Thus it performs the inverse operation of im2col
        :param reshaped_value: The value of the reshaped shape.
        :param shape: shape of x_value
        :return: The output at [:, i * self.strides[0]: i * self.strides[0] + self.kernel_size[0],
            j * self.strides[1]: j * self.strides[1] + self.kernel_size[1], :] is calculated by looping over n, with
            j = n % self.output_width and i = n // self.output_width. Then it selects the reshaped_value at
            [:, n * self.x.get_shape()[0]: (n + 1) * self.x.get_shape()[0]], then transposes it, reshapes it to the correct shape
            (self.x.get_shape()[0], self.kernel_size[0], self.kernel_size[1], self.x.get_shape()[-1])
        """
        x_value = np.zeros(shape)

        for n in range(int(reshaped_value.shape[1] / shape[0])):
            j = n % self.output_width
            i = int(n // self.output_width)

            # because it is quite difficult to reshape and stuff, you first need to reshape it like that
            # and then transpose it.
            reshaped_part_value = reshaped_value[:, n * shape[0]: (n + 1) * shape[0]].T
            reshaped_part_value = reshaped_part_value.reshape(shape[0], shape[-1],
                                                              self.kernel_size[0], self.kernel_size[1])
            reshaped_part_value = reshaped_part_value.transpose(0, 3, 2, 1)

            x_value[:, i * self.strides[0]: i * self.strides[0] + self.kernel_size[0],
            j * self.strides[1]: j * self.strides[1] + self.kernel_size[1], :] += reshaped_part_value

        return x_value


class AveragePooling(base.OneElementOperation):
    """
    Implements the average pooling layer.
    """
    def __init__(self, x, kernel_size, strides=None, name="average pooling", add_to_flow=True):
        """
        Initializes a new average pooling layer.
        :param x: BaseElement, the input of this convolutional operation, must have shape
                 (batch_size, width, height, n_channels)
        :param strides: tuple of two ints, The strides to use for the convolutional layer.
        :param name: String, the name of the layer.
        :param add_to_flow: Boolean, boolean indicating whether or not to add the operation to the flow.
        @post Creates variables registering the output width, output height and the strides.
        @post Calls the initializer of the OneElementOperation
        """
        self.kernel_size = kernel_size
        if strides is None:
            self.strides = kernel_size
        else:
            self.strides = strides
        self.output_height = np.int32((np.round(x.get_shape()[1]) / self.strides[0]))
        self.output_width = np.int32((np.round(x.get_shape()[2]) / self.strides[1]))
        super(AveragePooling, self).__init__(x, name=name, add_to_flow=add_to_flow)

    def operation(self):
        """
        Performs the average pooling operation on the input.
        :return: The output is an array of shape (self.x.get_shape()[0], self.output_width, self.output_height,
                 self.x.get_shape()[-1]) where each element is the average element of a certain frame. The element at
                 (i, k, m, j) is the mean of x at [i, k * self.strides[0]: (k + 1) * self.strides[0],
                 m * self.strides[1]: (m + 1) * self.strides[1], j].
        """
        output = np.zeros((self.x.get_shape()[0], self.output_height, self.output_width, self.x.get_shape()[-1]))

        for i in range(self.output_height):
            for j in range(self.output_width):
                x_part = self.x.get_value()[:, i * self.strides[0]: i * self.strides[0] + self.kernel_size[0],
                                            j * self.strides[1]: j * self.strides[1] + self.kernel_size[1], :].\
                                            reshape(self.x.get_shape()[0], self.strides[0] * self.strides[1],
                                                    self.x.get_shape()[-1])

                output[:, i, j, :] = np.mean(x_part, axis=1)

        return output

    def reversed_operation(self, gradient):
        """
        Performs the reversed operation on the average pooling.
        :param gradient: The gradient on which to perform the reversed operation
        :return: Returns an array of the shape x.get_shape(). For each frame an element belongs to, it gets the gradient of
                 that frame divided by the number of elements in one frame.
        """
        gradients_x = np.zeros(self.x.get_shape())

        for i in range(self.output_height):
            for j in range(self.output_width):
                gradients_part = gradient[:, i, j, :] / (self.strides[0] * self.strides[1])
                gradients_part = gradients_part[:, np.newaxis, np.newaxis, :]
                gradients_x[:, i * self.strides[0]: i * self.strides[0] + self.kernel_size[0],
                            j * self.strides[1]: j * self.strides[1] + self.kernel_size[1], :] += gradients_part

        return gradients_x
