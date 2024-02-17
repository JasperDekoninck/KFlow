import unittest
import kflow.base as Base
import kflow.operations as ops
import kflow.optimizers as optim


class OptimizersTest(unittest.TestCase):
    def setUp(self):
        self.session = Base.Session()
        self.simple_var = Base.Variable(0)
        self.simple_function = ops.Abs(self.simple_var ** 2 - 2)
        self.simple_function2 = ops.Abs(3 * self.simple_var ** 3 + 2 * self.simple_var ** 2 + self.simple_var + 1)
        self.loss = ops.LossMaker(self.simple_function)
        self.loss2 = ops.LossMaker(self.simple_function2)
        # Solution: -0.78389

    def test_SGD(self):
        optimizer = optim.SGD(learning_rate=0.001)
        optimizer.add_variables()
        for i in range(1000):
            self.session.run([self.loss2, optimizer])

        self.assertTrue(-0.78389 - 0.01 <= self.simple_var.get_value() <= -0.78389 + 0.01)

    def test_Momentum(self):
        optimizer = optim.MomentumOptimizer(learning_rate=0.001)
        optimizer.add_variables()
        for i in range(1000):
            self.session.run([self.loss2, optimizer])

        self.assertTrue(-0.78389 - 0.01 <= self.simple_var.get_value() <= -0.78389 + 0.01)

    def test_NesterovOptimizer(self):
        optimizer = optim.NesterovOptimizer(learning_rate=0.001)
        optimizer.add_variables()
        for i in range(1000):
            self.session.run([self.loss2, optimizer])

        self.assertTrue(-0.78389 - 0.01 <= self.simple_var.get_value() <= -0.78389 + 0.01)

    def test_AdaGrad(self):
        optimizer = optim.AdaGrad(learning_rate=0.01)
        optimizer.add_variables()
        for i in range(2000):
            self.session.run([self.loss2, optimizer])
        self.assertTrue(-0.78389 - 0.01 <= self.simple_var.get_value() <= -0.78389 + 0.01)

    def test_AdaDelta(self):
        optimizer = optim.AdaDelta()
        optimizer.add_variables()
        for i in range(1000):
            self.session.run([self.loss2, optimizer])
        self.assertTrue(-0.78389 - 0.01 <= self.simple_var.get_value() <= -0.78389 + 0.01)

    def test_Adam(self):
        optimizer = optim.Adam(learning_rate=0.01)
        optimizer.add_variables()
        for i in range(1000):
            self.session.run([self.loss2, optimizer])

        self.assertTrue(-0.78389 - 0.01 <= self.simple_var.get_value() <= -0.78389 + 0.01)

    def test_Adamax(self):
        optimizer = optim.AdaMax(learning_rate=0.01)
        optimizer.add_variables()
        for i in range(1000):
            self.session.run([self.loss2, optimizer])
        self.assertTrue(-0.78389 - 0.01 <= self.simple_var.get_value() <= -0.78389 + 0.01)

    def test_Nadam(self):
        optimizer = optim.Nadam(learning_rate=0.01)
        optimizer.add_variables()
        for i in range(1000):
            self.session.run([self.loss2, optimizer])
        self.assertTrue(-0.78389 - 0.01 <= self.simple_var.get_value() <= -0.78389 + 0.01)

    def test_RMSProp(self):
        optimizer = optim.RMSProp(learning_rate=0.01)
        optimizer.add_variables()
        for i in range(1000):
            self.session.run([self.loss2, optimizer])
        self.assertTrue(-0.78389 - 0.01 <= self.simple_var.get_value() <= -0.78389 + 0.01)

    def test_AMSProp(self):
        optimizer = optim.AMSProp(learning_rate=0.01)
        optimizer.add_variables()
        for i in range(1000):
            self.session.run([self.loss2, optimizer])
        self.assertTrue(-0.78389 - 0.01 <= self.simple_var.get_value() <= -0.78389 + 0.01)
