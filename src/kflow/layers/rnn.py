from .. import base
from .. import operations as ops
from ..initializations import *
from . import Layer
import numpy as np

class RNNbaseOperation(base.AdvancedOperation):
    """
    A class that performs a basic operation that happens quite a lot in RNN, namely
    activation(h.dot(Wh) + x.dot(Wx) + b)
    """
    def __init__(self, x, h, Wh, Wx, b, activation, name="RNN basic operation", add_to_flow=True):
        """
        Initializes a new RNN_Basic_operation
        :param x: Basic Element, the input on which to perform the operation (in RNN: the input of the data)
        :param h: Basic Element, the input on which to perform the operation
                  (in RNN: the output of the previous timestep)
        :param Wh: Variable, weights for the operation that can be optimized
        :param Wx: Variable, weights for the operation that can be optimized
        :param b: Variable, biases for the operation that can be optimized
        :param activation: Operation, the activation which needs to performed.
        :param name: String, The name of the operation
        :param add_to_flow: Boolean, boolean indicating whether or not to add the operation to the flow.
        @post Calls the initializer of AdvancedOperation with the following operations:
              activation(h.dot(Wh) + x.dot(Wx) + b)
        """
        dot_product1 = ops.Matmul(h, Wh, add_to_flow=False)
        dot_product2 = ops.Matmul(x, Wx, add_to_flow=False)
        add1 = ops.Add(dot_product1, dot_product2, add_to_flow=False)
        add2 = ops.Add(add1, b, add_to_flow=False)
        output = activation(add2, add_to_flow=False)

        operations = [dot_product1, dot_product2, add1, add2, output]

        base.AdvancedOperation.__init__(self, operations, name=name, add_to_flow=add_to_flow)


class baseRNN(Layer):
    """
    The base class for RNN layers.
    """
    def __init__(self, x, operations, h_values, x_timesteps, return_sequences,
                 dropout_rate=0, batch_norm=None, training=None, name="baseRNN", add_to_flow=True):
        """
        Initializes a new baseRNN
        :param x: Basic Element, the input on which to perform the operation.
        :param operations: List of Basic Elements, the operations that are already performed by the specific
               RNN layer and that need to be added to all operations of the layer.
        :param h_values: List of Basic Elements, the output of each timestep. The first element of this list contains
                         a Constant that is zero.
        :param x_timesteps: List of Placeholders, the input for each timestep.
        :param return_sequences: Boolean, boolean indicating whether or not to return all h_values or only to return
               the last one.
        :param dropout_rate: Float,
               The dropout rate for the output of the layer: note this is only added to the end of the
               layer, so it is not possible to add dropout between timesteps.
        :param batch_norm: Basic Element of BatchNormalization or None, Indicates whether or not to add batch
               normalization to the output of the layer.
        :param training: Placeholder, placeholder indicating whether or not the layer is currently training,
                         can only be None if the dropout rate is 0 and batch_norm is None
        :param name: String, the name of the layer.
        :param add_to_flow: Boolean, boolean indicating whether or not to add the operation to the flow.
        @post if return_sequences is false, the initializer of Layer is called with the given parameters
        @post If return_sequences is True, all h_values, except the first one,
              are concatenated with each other such that the output has dimensions (batch_size, time_steps, n_outputs).
              All these operations are added to the input operations and only then, the initializer of the Layer class
              is called.
        """
        self.operations = operations
        self.h_values = h_values
        self.x_timesteps = x_timesteps
        self.return_sequences = return_sequences
        self.x = x

        if self.return_sequences:
            changes = []
            # in order to concatenate h_values along timestep axis, we need to create a timestep axis
            # => h_values.shape = (batch_size, n_outputs), new_shape = (batch_size, 1, n_outputs)
            for h in self.h_values[1:]:
                change_shape_output = ops.Newaxis(h, axis=1, add_to_flow=False)
                changes.append(change_shape_output)
                self.operations.append(change_shape_output)

            # concatenating everything along timestep axis
            self.operations.append(ops.Concatenate(changes[0], changes[1], axis=1, add_to_flow=False))
            for m in range(2, len(changes)):
                self.operations.append(
                    ops.Concatenate(changes[m], self.operations[-1], axis=1, add_to_flow=False))

        Layer.__init__(self, self.operations, dropout_rate=dropout_rate, batch_norm=batch_norm,
                              training=training, name=name, add_to_flow=add_to_flow)

    def forward(self):
        """
        Overwrites the forward pass of the layer.
        @post Sets the value of the x_timesteps correctly, namely it sets the value of the i-th timestep to the value
        of x[:, i, :].
        @post Runs all the forward passes of the operations in its list of operations.
        """
        # setting the placeholder timesteps values correctly
        for i in range(self.x.shape[1]):
            self.x_timesteps[i].set_value(self.x.get_value()[:, i, :])

        # running the operations
        for op in self.operations:
            op.forward()

    def backward(self):
        """
        Overwrites the backward pass of the layer.
        @post Runs all the backward passes of the operations in reverse.
        @post Sets the gradient of the input x, by adding the gradient of the i-th timestep to x.get_gradient()[:, i, :]
        """
        for op in reversed(self.operations):
            op.backward()

        # setting the gradient of x
        gradient_x = np.zeros(self.x.shape)

        for i, x_timestep in enumerate(self.x_timesteps):
            gradient_x[:, i, :] = x_timestep.get_gradient().reshape(self.x.shape[0], self.x.shape[2])

        self.x.add_to_gradient(gradient_x)


class BasicRNNCell(baseRNN):
    """
    Implements the Basic RNN Cell on an input of shape (batch_size, timesteps, n_inputs).
    """

    # Note that it doesn't perform the second equation given in the Stanford computer vision course. For this, you will
    # need to put a dense layer on top of it.

    def __init__(self, x, n_outputs, activation=ops.Tanh, weight_initializer=he_normal_initialization,
                 hidden_initializer=orthogonal_initialization, bias_hidden_initializer=constant_initialization,
                 return_sequences=True, dropout_rate=0, batch_norm=None, training=None,
                 name="Basic RNN", add_to_flow=True):
        """
        Initializes a new BasicRNNCell
        :param x: Basic Element, the input of the cell, must be of shape (batch_size, timesteps, n_inputs)
        :param n_outputs: Int, the number of outputs for each timestep of the cell.
        :param activation: Operation, the activation to use in between the outputs of the cells.
        :param weight_initializer: initializer function, Serves as initialization of the weights for the matmul with
                                   the input x.
        :param hidden_initializer: initializer function. Serves as initialization of the weights for the matmul
                                   with the output of the previous timestep.
        :param bias_hidden_initializer: initializer function. Serves as initialization of the biases.
        :param return_sequences: Boolean, boolean indicating whether or not to return all timestep outputs
        :param dropout_rate: Float,
               The dropout rate for the output of the layer: note this is only added to the end of the
               layer, so it is not possible to add dropout between timesteps.
        :param batch_norm: Basic Element of BatchNormalization or None, Indicates whether or not to add batch
               normalization to the output of the layer.
        :param training: Placeholder, placeholder indicating whether or not the layer is currently training,
                         can only be None if the dropout rate is 0 and batch_norm is None
        :param name: String, the name of the layer.
        :param add_to_flow: Boolean, boolean indicating whether or not to add the operation to the flow.
        @post Creates x_timesteps which is a list of placeholders holding the input for each timestep.
        @post Creates variables Wxh, Whh, b that can be optimized and perform the basic RNN operation on each
              timestep.
        @post For each timestep (= x.shape[1]) performs a RNN basic operation on the previous output of the
              RNN and the new input for the RNN.
        @post Calls the initializer of the baseRNN with the operations mentioned in the previous @post.
        """

        self.n_outputs = n_outputs
        self.x = x

        # creating all variables
        self.Wxh = base.Variable(weight_initializer((self.x.shape[2], n_outputs)))
        self.Whh = base.Variable(hidden_initializer((n_outputs, n_outputs)))
        self.b = base.Variable(bias_hidden_initializer((1, n_outputs)))

        # creating a constant for the first h_0 and placeholders for the value of x at each timestep
        h_0 = base.Constant(np.zeros((self.x.shape[0], n_outputs)), add_to_flow=False)
        self.x_timesteps = [base.Placeholder((self.x.shape[0], self.x.shape[2]), add_to_flow=False)
                            for _ in range(self.x.shape[1])]

        h_values = [h_0]

        # creating all the operations in the correct order, now all operations are just about calculating h_values
        for i in range(self.x.shape[1]):
            # computing activation(h_{t-1}.dot(Whh) + x.dot(Wxh) + b_h)
            basic_rnn = RNNbaseOperation(self.x_timesteps[i], h_values[-1], self.Whh, self.Wxh,
                                          self.b, activation, add_to_flow=False)
            h_values.append(basic_rnn)

        baseRNN.__init__(self, x, h_values, h_values, self.x_timesteps, return_sequences,
                         dropout_rate=dropout_rate, batch_norm=batch_norm, training=training,
                         name=name, add_to_flow=add_to_flow)


class LSTM(baseRNN):
    """
    Implements the LSTM Cell on an input of shape (batch_size, timesteps, n_inputs).
    """
    def __init__(self, x, n_outputs, activation=ops.Tanh, weight_initializer=he_normal_initialization,
                 hidden_initializer=orthogonal_initialization, bias_hidden_initializer=constant_initialization,
                 return_sequences=True, dropout_rate=0, batch_norm=None, training=None,
                 name="Basic RNN", add_to_flow=True):
        """
        Initializes a new LSTM.
        :param x: Basic Element, the input of the cell, must be of shape (batch_size, timesteps, n_inputs)
        :param n_outputs: Int, the number of outputs for each timestep of the cell.
        :param activation: Operation, the activation to use in between the outputs of the cells.
        :param weight_initializer: initializer function, Serves as initialization of the weights for the matmul with
                                   the input x.
        :param hidden_initializer: initializer function. Serves as initialization of the weights for the matmul
                                   with the output of the previous timestep.
        :param bias_hidden_initializer: initializer function. Serves as initialization of the biases.
        :param return_sequences: Boolean, boolean indicating whether or not to return all timestep outputs
        :param dropout_rate: Float,
               The dropout rate for the output of the layer: note this is only added to the end of the
               layer, so it is not possible to add dropout between timesteps.
        :param batch_norm: Basic Element of BatchNormalization or None, Indicates whether or not to add batch
               normalization to the output of the layer.
        :param training: Placeholder, placeholder indicating whether or not the layer is currently training,
                         can only be None if the dropout rate is 0 and batch_norm is None
        :param name: String, the name of the layer.
        :param add_to_flow: Boolean, boolean indicating whether or not to add the operation to the flow.
        @post Creates Variables for the hidden weights (weights for the previous output), weights (weights for the
              current inputs) and biases for each gate of the LSTM.
        @post Creates x_timesteps which is a list of placeholders holding the input for each timestep.
        @post For each timestep, performs the following operations:
              An RNNbaseOperation for each of the four gates (input gate, forget gate, output gate, gate gate),
              each gate has activation Sigmoid, only the gate gate uses the Tanh activation.
              Multiplies the previous c_value (long term meory) with the current forget gate element wise.
              Multiplies the current gate gate with the current input gate element wise.
              Adds the result of the previous two results and stocks the result in c_value.
              Performs the given activation on teh c_value, with this result it calculates the current h_value, namely
              by multiplying this element wise with the output gate.
        @post Calls the initializer of the baseRNN with all previous operations.
        Using formulas, one timestep update looks like this:
        input_gate = sigmoid(h_{t-1}.dot(Whh_i) + x_t.dot(Wxh_i) + b_i)
        forget_gate = sigmoid(h_{t-1}.dot(Whh_f) + x_t.dot(Wxh_f) + b_f)
        output_gate = sigmoid(h_{t-1}.dot(Whh_o) + x_t.dot(Wxh_o) + b_o)
        gate_gate = tanh(h_{t-1}.dot(Whh_g) + x_t.dot(Wxh_g) + b_g)
        c_t = forget_gate * c_{t-1} + gate_gate * input_gate
        h_t = output_gate * activation(c_t)
        """

        self.x = x
        # creating all variables
        Wxh_i = base.Variable(weight_initializer((self.x.shape[2], n_outputs)))
        Wxh_f = base.Variable(weight_initializer((self.x.shape[2], n_outputs)))
        Wxh_o = base.Variable(weight_initializer((self.x.shape[2], n_outputs)))
        Wxh_g = base.Variable(weight_initializer((self.x.shape[2], n_outputs)))
        self.weights = [Wxh_i, Wxh_f, Wxh_o, Wxh_g]

        Whh_i = base.Variable(hidden_initializer((n_outputs, n_outputs)))
        Whh_f = base.Variable(hidden_initializer((n_outputs, n_outputs)))
        Whh_o = base.Variable(hidden_initializer((n_outputs, n_outputs)))
        Whh_g = base.Variable(hidden_initializer((n_outputs, n_outputs)))
        self.hidden_weights = [Whh_i, Whh_f, Whh_o, Whh_g]

        b_i = base.Variable(bias_hidden_initializer((1, n_outputs)))
        b_f = base.Variable(bias_hidden_initializer((1, n_outputs)))
        b_o = base.Variable(bias_hidden_initializer((1, n_outputs)))
        b_g = base.Variable(bias_hidden_initializer((1, n_outputs)))
        self.biases = [b_i, b_f, b_o,  b_g]

        # creating a constant for the first h_0 and placeholders for the value of x at each timestep
        h_0 = base.Constant(np.zeros((self.x.shape[0], n_outputs)), add_to_flow=False)
        self.x_timesteps = [base.Placeholder((self.x.shape[0], self.x.shape[2]), add_to_flow=False)
                            for _ in range(self.x.shape[1])]

        h_values = [h_0]
        operations = []

        # creating all the operations in the correct order
        for i in range(self.x.shape[1]):
            # computing activation(h_{t-1}.dot(Whh) + x.dot(Wxh) + b_h)
            for W_input, W_hidden, bias, gate in zip(self.weights, self.hidden_weights, self.biases, range(4)):
                # if gate != the gate gate, than applying sigmoid activation, else applying tanh activation
                if gate != 3:
                    current_activation = ops.Sigmoid
                else:
                    current_activation = ops.Tanh

                basic_rnn = RNNbaseOperation(self.x_timesteps[i], h_values[-1], W_hidden, W_input,
                                              bias, current_activation, add_to_flow=False)
                operations.append(basic_rnn)

            if len(operations) >= 7:
                c_multiply1 = ops.Multiply(operations[-3], operations[-7], add_to_flow=False)  # forget_gate * c_{t-1}
            else:
                c_multiply1 = base.Constant(np.zeros((self.x.shape[0], n_outputs)), add_to_flow=False)
            c_multiply2 = ops.Multiply(operations[-4], operations[-1], add_to_flow=False)  # gate_gate * input_gate
            c_value = ops.Add(c_multiply1, c_multiply2, add_to_flow=False)

            activation_h = activation(c_value, add_to_flow=False)  # activation(c_t)
            h_value = ops.Multiply(activation_h, operations[-2], add_to_flow=False)  # output_gate * activation(c_t)
            h_values.append(h_value)

            # appending everything to operations
            operations.append(c_multiply1)
            operations.append(c_multiply2)
            operations.append(c_value)
            operations.append(activation_h)
            operations.append(h_value)

        baseRNN.__init__(self, x, operations, h_values, self.x_timesteps, return_sequences,
                         dropout_rate=dropout_rate, batch_norm=batch_norm, training=training,
                         name=name, add_to_flow=add_to_flow)


class GRU(baseRNN):
    """
    Implements the GRU Cell on an input of shape (batch_size, timesteps, n_inputs).
    """
    def __init__(self, x, n_outputs, activation=ops.Tanh, weight_initializer=he_normal_initialization,
                 hidden_initializer=orthogonal_initialization, bias_hidden_initializer=constant_initialization,
                 return_sequences=True, dropout_rate=0, batch_norm=None, training=None,
                 name="Basic RNN", add_to_flow=True):
        """
        Initializes a new GRU.
        :param x: Basic Element, the input of the cell, must be of shape (batch_size, timesteps, n_inputs)
        :param n_outputs: Int, the number of outputs for each timestep of the cell.
        :param activation: Operation, the activation to use in between the outputs of the cells.
        :param weight_initializer: initializer function, Serves as initialization of the weights for the matmul with
                                   the input x.
        :param hidden_initializer: initializer function. Serves as initialization of the weights for the matmul
                                   with the output of the previous timestep.
        :param bias_hidden_initializer: initializer function. Serves as initialization of the biases.
        :param dropout_rate: Float,
               The dropout rate for the output of the layer: note this is only added to the end of the
               layer, so it is not possible to add dropout between timesteps.
        :param batch_norm: Basic Element of BatchNormalization or None, Indicates whether or not to add batch
               normalization to the output of the layer.
        :param training: Placeholder, placeholder indicating whether or not the layer is currently training,
                         can only be None if the dropout rate is 0 and batch_norm is None
        :param name: String, the name of the layer.
        :param add_to_flow: Boolean, boolean indicating whether or not to add the operation to the flow.
        @post Creates Variables for the hidden weights (weights for the previous output), weights (weights for the
              current inputs) and biases for each gate of the LSTM.
        @post Creates x_timesteps which is a list of placeholders holding the input for each timestep.
        @post For each timestep, performs the following operations:
              For the "r" and "z" gate it performs an RNNbaseOperation with the sigmoid activation.
              For the "e" gate, it first multiplies the current value of the "r" gate element wise with the previous
              output of the RNN, then it performs a RNNbaseOperation with the parameter activation.
              based on the three gates, the new output is given by z * h_{t-1} + (1 - z) * e
        @post Calls the initializer of the baseRNN with all previous operations.
        Using formulas, one timestep update looks like this:
        r_t = sigmoid(x_t.dot(Wxr) + h_{t-1}.dot(Whr) + b_r)
        z_t = sigmoid(x_t.dot(Wxz) + h_{t-1}.dot(Whz) + b_z)
        e_t = tanh(x_t.dot(Wxe) + (r_t * h_{t-1}).dot(Whe) + b_e)
        h_t = z_t * h_{t-1} + (1 - z_t) * e_t
        """

        self.x = x
        # creating all variables
        Wxr = base.Variable(weight_initializer((self.x.shape[2], n_outputs)))
        Wxz = base.Variable(weight_initializer((self.x.shape[2], n_outputs)))
        Wxe = base.Variable(weight_initializer((self.x.shape[2], n_outputs)))
        self.weights = [Wxr, Wxz, Wxe]

        Whr = base.Variable(hidden_initializer((n_outputs, n_outputs)))
        Whz = base.Variable(hidden_initializer((n_outputs, n_outputs)))
        Whe = base.Variable(hidden_initializer((n_outputs, n_outputs)))
        self.hidden_weights = [Whr, Whz, Whe]

        b_r = base.Variable(bias_hidden_initializer((1, n_outputs)))
        b_z = base.Variable(bias_hidden_initializer((1, n_outputs)))
        b_e = base.Variable(bias_hidden_initializer((1, n_outputs)))
        self.biases = [b_r, b_z, b_e]

        # creating a constant for the first h_0 and placeholders for the value of x at each timestep
        h_0 = base.Constant(np.zeros((self.x.shape[0], n_outputs)), add_to_flow=False)
        self.x_timesteps = [base.Placeholder((self.x.shape[0], self.x.shape[2]), add_to_flow=False) for _ in
                            range(self.x.shape[1])]

        h_values = [h_0]
        operations = []

        # creating all the operations in the correct order
        for i in range(self.x.shape[1]):
            # computing activation(h_{t-1}.dot(Whh) + x.dot(Wxh) + b_h)
            for W_input, W_hidden, bias, gate in zip(self.weights, self.hidden_weights, self.biases, range(3)):
                # if gate != the gate gate, than applying sigmoid activation, else applying tanh activation
                if gate != 2:
                    current_activation = ops.Sigmoid
                    basic_rnn = RNNbaseOperation(self.x_timesteps[i], h_values[-1], W_hidden, W_input,
                                                  bias, current_activation, add_to_flow=False)
                    operations.append(basic_rnn)
                else:
                    current_activation = activation
                    multiply1 = ops.Multiply(operations[-2], h_values[-1], add_to_flow=False)
                    basic_rnn = RNNbaseOperation(self.x_timesteps[i], multiply1, W_hidden, W_input,
                                                  bias, current_activation, add_to_flow=False)
                    operations.append(multiply1)
                    operations.append(basic_rnn)

            multiply1 = ops.Multiply(operations[-3], h_values[-1], add_to_flow=False)
            subtract1 = ops.Subtract(1, operations[-3], add_to_flow=False)
            multiply2 = ops.Multiply(subtract1, operations[-1], add_to_flow=False)
            h_value = ops.Add(multiply1, multiply2, add_to_flow=False)
            h_values.append(h_value)

            # appending everything to operations
            operations.append(multiply1)
            operations.append(subtract1)
            operations.append(multiply2)
            operations.append(h_value)

        baseRNN.__init__(self, x, operations, h_values, self.x_timesteps, return_sequences,
                         dropout_rate=dropout_rate, batch_norm=batch_norm, training=training,
                         name=name, add_to_flow=add_to_flow)
