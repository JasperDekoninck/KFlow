import unittest
import kflow.base as Base
import kflow.operations as ops
import numpy as np

class OperationTest(unittest.TestCase):
    def numerical_derivative(self, operation, gradient, X, x, Y=None, y=None, h=1e-7):
        """
        Calculates the numerical derivative for the given gradient
        :param operation: the operation for which to compute the numerical derivative
        :param gradient: the gradient for which to evaluate the partial derivatives
        :param X: Placeholder, first input
        :param x: value for the first input
        :param Y: Placeholder, second input
        :param y: value for the second input
        :param h: small number that is used as divisor in the numerical calculation
        :return: The numerical derivatives for x and y
        """
        if Y is None:
            Y = Base.Placeholder(1)
            y = 0

        if np.isscalar(x):
            self.session.forward([operation], placeholder_values=[[X, x + h], [Y, y]])
            first_term = operation.get_value()
            self.session.forward([operation], placeholder_values=[[X, x - h], [Y, y]])
            second_term = operation.get_value()
            d_num_x = (gradient * (first_term - second_term)).sum() / (2 * h)
        else:
            x = x.astype(np.float64)
            d_num_x = np.zeros(x.shape)
            for element in np.ndindex(x.shape):
                new_x = np.copy(x)
                new_x[element] += h
                self.session.forward([operation], placeholder_values=[[X, new_x], [Y, y]])
                first_term = operation.get_value()
                new_x_2 = np.copy(x)
                new_x_2[element] -= h
                self.session.forward([operation], placeholder_values=[[X, new_x_2], [Y, y]])
                second_term = operation.get_value()
                d_num_x[element] = (gradient * (first_term - second_term)).sum() / (2 * h)
    
        # checking the answer of backward numerically for y:
        if np.isscalar(y):
            self.session.forward([operation], placeholder_values=[[X, x], [Y, y + h]])
            first_term = operation.get_value()
            self.session.forward([operation], placeholder_values=[[X, x], [Y, y - h]])
            second_term = operation.get_value()
            d_num_y = (gradient * (first_term - second_term)).sum() / (2 * h)
        else:
            y = y.astype(np.float64)
            d_num_y = np.zeros(y.shape)
            for element in np.ndindex(y.shape):
                new_y = np.copy(y)
                new_y[element] += h
                self.session.forward([operation], placeholder_values=[[X, x], [Y, new_y]])
                first_term = operation.get_value()
                new_y_2 = np.copy(y)
                new_y_2[element] -= h
                self.session.forward([operation], placeholder_values=[[X, x], [Y, new_y_2]])
                second_term = operation.get_value()
                d_num_y[element] = (gradient * (first_term - second_term)).sum() / (2 * h)
    
        return d_num_x, d_num_y

    def arrayEquals(self, array1, array2, precision=1e-5):
        """
        Checks whether the two given arrays are equal for the given precision
        :param array1: First array
        :param array2: Second array
        :param precision: scalar
        :return: Boolean indidcating whether or not the arrays are the same given the precision. Thus for each
                 element x1 of the first array and the corresponding element x2 of the second array,
                 the following equation must hold: x2 - precision <= x1 <= x2 + precision
        """
        indicator1 = array2 - precision <= array1
        indicator2 = array1 <= array2 + precision
        return np.all(indicator1 & indicator2)

    def basicTwoElement(self, operation, X, value_X, Y, value_Y, expected_value):
        self.session.forward([operation], placeholder_values=[[X, value_X], [Y, value_Y]])
        self.assertTrue(self.arrayEquals(operation.get_value(), expected_value))
        gradients = np.ones(operation.shape)
        d_num_x, d_num_y = self.numerical_derivative(operation, gradients, X, value_X, Y, value_Y)
        self.session.backward([X, Y], placeholder_gradient=[[operation, gradients]])
        self.assertTrue(self.arrayEquals(X.get_gradient(), d_num_x))
        self.assertTrue(self.arrayEquals(Y.get_gradient(), d_num_y))

    def basicOneElement(self, operation, X, value_X, expected_value):
        self.session.forward([operation], placeholder_values=[[X, value_X]])
        self.assertTrue(self.arrayEquals(operation.get_value(), expected_value))
        gradients = np.ones(operation.shape)
        d_num_x, _ = self.numerical_derivative(operation, gradients, X, value_X)
        self.session.backward([X], placeholder_gradient=[[operation, gradients]])
        self.assertTrue(self.arrayEquals(X.get_gradient(), d_num_x))

    def setUp(self):
        self.session = Base.Session()
        self.X = Base.Placeholder((3, 3), name="X")
        self.Y = Base.Placeholder((3, 3), name="Y")
        self.value_X = np.array([[2, -3, 5], [-1, 6, 4], [2, 3, -1]], dtype=np.float64)
        self.value_Y = np.array([[1, 2, -3], [3, -4, 5], [5, -6, 7]], dtype=np.float64)

    def tearDown(self):
        self.session.reset()

    def test_doNothing(self):
        operation = ops.DoNothing(self.X)
        self.basicOneElement(operation, self.X, self.value_X, self.value_X)

    def test_add(self):
        operation = self.X + self.Y
        self.basicTwoElement(operation, self.X, self.value_X, self.Y, self.value_Y, self.value_X + self.value_Y)

    def test_subtract(self):
        operation = self.X - self.Y
        self.basicTwoElement(operation, self.X, self.value_X, self.Y, self.value_Y, self.value_X - self.value_Y)

    def test_matmul(self):
        operation = ops.Matmul(self.X, self.Y)
        self.basicTwoElement(operation, self.X, self.value_X, self.Y, self.value_Y, np.dot(self.value_X, self.value_Y))
        self.session.reset()
        X = Base.Placeholder(shape=(5, 4))
        Y = Base.Placeholder(shape=(4, 3))
        operation = ops.Matmul(X, Y)
        valuex = np.linspace(0, 60, 20).reshape(5, 4)
        valuey = np.linspace(-10, 10, 12).reshape(4, 3)
        self.basicTwoElement(operation, X, valuex, Y, valuey, np.dot(valuex, valuey))

    def test_maximum(self):
        operation = ops.Maximum(self.X, self.Y)
        self.basicTwoElement(operation, self.X, self.value_X, self.Y, self.value_Y, np.maximum(self.value_X,
                                                                                               self.value_Y))

    def test_minimum(self):
        operation = ops.Minimum(self.X, self.Y)
        self.basicTwoElement(operation, self.X, self.value_X, self.Y, self.value_Y, np.minimum(self.value_X,
                                                                                               self.value_Y))

    def test_power(self):
        operation = ops.Power(self.X, 2)
        self.basicOneElement(operation, self.X, self.value_X, self.value_X ** 2)
        operation = ops.Power(self.X, -3)
        self.basicOneElement(operation, self.X, self.value_X, self.value_X ** (-3))

    def test_exp(self):
        operation = ops.Exp(self.X, 2)
        self.basicOneElement(operation, self.X, self.value_X, 2 ** self.value_X)
        operation = ops.Exp(self.X, np.e)
        self.basicOneElement(operation, self.X, self.value_X, np.e ** self.value_X)

    def test_log(self):
        value_x = np.abs(self.value_X)
        operation = ops.Log(self.X, base=2)
        self.basicOneElement(operation, self.X, value_x, np.log(value_x) / np.log(2))

    def test_multiply(self):
        operation = self.X * self.Y
        self.basicTwoElement(operation, self.X, self.value_X, self.Y, self.value_Y, np.multiply(self.value_X,
                                                                                               self.value_Y))

    def test_divide(self):
        operation = self.X / self.Y
        self.basicTwoElement(operation, self.X, self.value_X, self.Y, self.value_Y, np.divide(self.value_X,
                                                                                              self.value_Y))

    def test_abs(self):
        operation = ops.Abs(self.X)
        self.basicOneElement(operation, self.X, self.value_X, np.abs(self.value_X))

    def test_sum(self):
        operation = ops.Sum(self.X, axis=None)
        self.basicOneElement(operation, self.X, self.value_X, np.sum(self.value_X, axis=None))
        operation = ops.Sum(self.X, axis=1, keepdims=True)
        self.basicOneElement(operation, self.X, self.value_X, np.sum(self.value_X, axis=1, keepdims=True))
        operation = ops.Sum(self.X, axis=0, keepdims=False)
        self.basicOneElement(operation, self.X, self.value_X, np.sum(self.value_X, axis=0, keepdims=False))

    def test_mean(self):
        operation = ops.Mean(self.X, axis=None)
        self.basicOneElement(operation, self.X, self.value_X, np.mean(self.value_X, axis=None))
        operation = ops.Mean(self.X, axis=1, keepdims=True)
        self.basicOneElement(operation, self.X, self.value_X, np.mean(self.value_X, axis=1, keepdims=True))
        operation = ops.Mean(self.X, axis=0, keepdims=False)
        self.basicOneElement(operation, self.X, self.value_X, np.mean(self.value_X, axis=0, keepdims=False))

    def test_broadcastTo(self):
        operation = ops.BroadcastTo(self.X, (3, 3, 3))
        self.basicOneElement(operation, self.X, self.value_X, np.broadcast_to(self.value_X, shape=(3, 3, 3)))
        X = Base.Placeholder((3, 1, 3))
        value = np.ones((3, 1, 1))
        operation = ops.BroadcastTo(X, (3, 3, 3))
        self.basicOneElement(operation, X, value, np.broadcast_to(value, shape=(3, 3, 3)))

    def test_newaxis(self):
        operation = ops.Newaxis(self.X, axis=1)
        self.basicOneElement(operation, self.X, self.value_X, np.expand_dims(self.value_X, axis=1))

    def test_reshape(self):
        operation = ops.Reshape(self.X, (1, 9))
        self.basicOneElement(operation, self.X, self.value_X, np.reshape(self.value_X, (1, 9)))
        X = Base.Placeholder((3, 4, 5))
        value = np.ones((3, 4, 5))
        operation = ops.Reshape(X, (10, 2, 3))
        self.basicOneElement(operation, X, value, np.reshape(value, (10, 2, 3)))

    def test_flatten(self):
        X = Base.Placeholder((3, 1, 3))
        value = np.linspace(0, 8, 9).reshape((3, 1, 3))
        operation = ops.Flatten(X)
        self.basicOneElement(operation, X, value,
                             value.reshape(3, 3))

    def test_concatenate(self):
        operation = ops.Concatenate(self.X, self.Y, axis=1)
        self.basicTwoElement(operation, self.X, self.value_X, self.Y, self.value_Y, np.concatenate((self.value_X,
                                                                                              self.value_Y), axis=1))

    def test_pad(self):
        X = Base.Placeholder((3, 3, 3, 1))
        value = np.ones((3, 3, 3, 1))
        operation = ops.Pad(X, 1)
        expected_value = np.zeros((3, 5, 5, 1))
        expected_value[:, 1:4, 1:4, :] += 1
        self.basicOneElement(operation, X, value, expected_value)

    def test_var(self):
        operation = ops.Variance(self.X, axis=None)
        expected_value = np.mean((self.value_X - np.mean(self.value_X)) ** 2)
        self.basicOneElement(operation, self.X, self.value_X, expected_value)
        operation = ops.Variance(self.X, axis=1)
        expected_value = np.mean((self.value_X - np.mean(self.value_X, axis=1, keepdims=True)) ** 2, axis=1,
                                 keepdims=True)
        self.basicOneElement(operation, self.X, self.value_X, expected_value)

    def test_std(self):
        operation = ops.Std(self.X, axis=None)
        expected_value = np.power(np.mean((self.value_X - np.mean(self.value_X)) ** 2), 0.5)
        self.basicOneElement(operation, self.X, self.value_X, expected_value)
        operation = ops.Std(self.X, axis=1)
        expected_value = np.power(np.mean((self.value_X - np.mean(self.value_X, axis=1, keepdims=True)) ** 2, axis=1,
                                          keepdims=True), 0.5)
        self.basicOneElement(operation, self.X, self.value_X, expected_value)

    def test_lossMaker(self):
        operation = ops.Std(self.X, axis=None)
        expected_value = np.power(np.mean((self.value_X - np.mean(self.value_X)) ** 2), 0.5)
        loss = ops.LossMaker(operation)
        self.basicOneElement(loss, self.X, self.value_X, expected_value)

    def test_mse(self):
        operation = ops.Mse(self.X, self.Y)
        expected_value = np.mean((self.value_X - self.value_Y) ** 2)
        self.basicTwoElement(operation, self.X, self.value_X, self.Y, self.value_Y, expected_value)

    def test_mae(self):
        operation = ops.Mae(self.X, self.Y)
        expected_value = np.mean(np.abs(self.value_X - self.value_Y))
        self.basicTwoElement(operation, self.X, self.value_X, self.Y, self.value_Y, expected_value)

    def test_binaryCrossEntropy(self):
        operation = ops.BinaryCrossentropy(self.Y, self.X)
        x = np.abs(self.value_X) / (np.max(self.value_X) + 3)
        y = np.abs(self.value_Y) / (np.max(self.value_Y) + 3)
        expected_value = np.mean(- y * np.log(x) - (1 - y) * np.log(1 - x))
        self.basicTwoElement(operation, self.X, x, self.Y, y, expected_value)

    def test_categoricalCrossEntropy(self):
        operation = ops.CategoricalCrossentropy(self.Y, self.X)
        x = np.abs(self.value_X) / (np.max(self.value_X) + 3)
        y = np.abs(self.value_Y) / (np.max(self.value_Y) + 3)
        expected_value = np.mean(- np.sum(y * np.log(x), axis=1))
        self.basicTwoElement(operation, self.X, x, self.Y, y, expected_value)

    def test_ReLU(self):
        operation = ops.ReLU(self.X)
        expected_value = np.maximum(self.value_X, 0)
        self.basicOneElement(operation, self.X, self.value_X, expected_value)

    def test_sigmoid(self):
        operation = ops.Sigmoid(self.X)
        expected_value = 1 / (1 + np.exp(-self.value_X))
        self.basicOneElement(operation, self.X, self.value_X, expected_value)

    def test_tanh(self):
        operation = ops.Tanh(self.X)
        expected_value = (np.exp(self.value_X) - np.exp(-self.value_X)) / (np.exp(self.value_X) + np.exp(-self.value_X))
        self.basicOneElement(operation, self.X, self.value_X, expected_value)

    def test_softmax(self):
        operation = ops.Softmax(self.X)
        expected_value = np.e ** self.value_X / (np.sum(np.exp(self.value_X), axis=1, keepdims=True))
        self.basicOneElement(operation, self.X, self.value_X, expected_value)

    def test_ELU(self):
        operation = ops.ELU(self.X)
        expected_value = np.maximum(self.value_X, 0) - 1 * (np.exp(self.value_X) - 1) * np.sign(np.minimum(self.value_X, 0))
        self.basicOneElement(operation, self.X, self.value_X, expected_value)

    def test_LeakyReLU(self):
        operation = ops.LeakyReLU(self.X)
        expected_value = np.maximum(self.value_X, 0) + np.minimum(self.value_X, 0) * 0.01
        self.basicOneElement(operation, self.X, self.value_X, expected_value)

    def test_PReLU(self):
        operation = ops.PReLU(self.X, initial_value=0.5)
        expected_value = np.maximum(self.value_X, 0) + np.minimum(self.value_X, 0) * 0.5
        self.basicOneElement(operation, self.X, self.value_X, expected_value)

    def test_swish(self):
        operation = ops.Swish(self.X, initial_value=0.5)
        expected_value = self.value_X * (1 / (1 + np.e ** (-self.value_X * 0.5)))
        self.basicOneElement(operation, self.X, self.value_X, expected_value)

    def test_l1Regularizer(self):
        operation = ops.L1Regularizer(self.X)
        expected_value = np.sum(np.abs(self.value_X)) * 0.01
        self.basicOneElement(operation, self.X, self.value_X, expected_value)

    def test_l2Regularizer(self):
        operation = ops.L2Regularizer(self.X)
        expected_value = np.sum(self.value_X ** 2) * 0.01
        self.basicOneElement(operation, self.X, self.value_X, expected_value)

    def test_l1l2Regularizer(self):
        operation = ops.L1L2Regularizer(self.X, l2_weight=0.02)
        expected_value = np.sum(np.abs(self.value_X)) * 0.01 + np.sum(self.value_X ** 2) * 0.02
        self.basicOneElement(operation, self.X, self.value_X, expected_value)

    def test_dropout(self):
        # because it is random, impossible to give exact expression for expected value
        training = Base.Placeholder(1)
        training.set_value(True)
        operation = ops.Dropout(self.X, training)
        self.session.forward([operation], placeholder_values=[[self.X, self.value_X]])
        for pred, true in np.nditer([self.value_X, operation.get_value()]):
            self.assertTrue(true * operation.keep_rate - 1e-7 <= pred <= true * operation.keep_rate + 1e-7 or
                            -1e-7 <= true <= 1e-7)

        gradients = np.ones(operation.shape)
        self.session.backward([self.X], placeholder_gradient=[[operation, gradients]])
        for pred, true in np.nditer([operation.fallout, self.X.get_gradient()]):
            self.assertTrue(true * operation.keep_rate - 1e-7 <= pred <= true * operation.keep_rate + 1e-7)

        training.set_value(False)
        self.session.forward([operation], placeholder_values=[[self.X, self.value_X]])
        self.assertTrue(self.arrayEquals(operation.get_value(), self.value_X))

    def test_dropout2(self):
        # Checking whether there are indeed about self.dropout_rate neurons dropped every time
        training = Base.Placeholder(1)
        training.set_value(True)
        X = Base.Placeholder(shape=(10, 10))
        valuex = np.ones((10, 10))
        operation = ops.Dropout(X, training)
        zeros = 0
        total = 0
        for i in range(100):
            self.session.forward([operation], placeholder_values=[[X, valuex]])
            total += np.prod(valuex.shape)
            zeros += np.prod(valuex.shape) - np.count_nonzero(operation.get_value())

        self.assertTrue(0.45 * total <= zeros <= 0.55 * total)

    def test_batchnorm(self):
        training = Base.Placeholder(1)
        training.set_value(True)
        operation = ops.BatchNormalization(self.X, training)
        expected_value = (self.value_X - np.mean(self.value_X, axis=1, keepdims=True)) / np.std(self.value_X, axis=1,
                                                                                                keepdims=True)
        self.basicOneElement(operation, self.X, self.value_X, expected_value)
        training.set_value(False)
        self.session.forward([operation], placeholder_values=[[self.X, self.value_X]])
        expected_value = (self.value_X - operation.running_mean) / operation.running_variance ** 0.5
        self.assertTrue(self.arrayEquals(operation.get_value(), expected_value))

if __name__ == '__main__':
    unittest.main()