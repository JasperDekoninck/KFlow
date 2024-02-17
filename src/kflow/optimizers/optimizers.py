from .. import base
import numpy as np

"""Good explanations for all optimizers: http://ruder.io/optimizing-gradient-descent/index.html"""


class Optimizer:
    """base class for all optimizers"""
    def __init__(self, name="optimizer", clipvalue=None, variables=None):
        """
        Initializes a new optimizer
        :param name: The name of the optimizer
        :param clipvalue: An array of size two indicating what the maximum gradient and what the minimum gradient is
        :param variables: The variables the optimizer needs to update, None or list of Variables
        :post Sets the session of this optimizers to the open session
        :post Adds the optimizer to the open session
        :post Creates a variable registering the variables and a variable registering the clipvalue
        """
        assert variables is None or isinstance(variables, (list, tuple))
        assert variables is None or np.all([isinstance(variable, base.Variable)] for variable in variables)
        assert clipvalue is None or (isinstance(clipvalue, (list, tuple)) and len(clipvalue) == 2)
        self.session = base.Session.open_session
        if variables is None:
            self.variables = self.session.variables
        else:
            self.variables = variables

        self.name = self.session.add_optimizer(self, name)

        self.clipvalue = clipvalue

    def add_variables(self, variables=None):
        """
        Adds new variables for the optimizer to optimize. Comes in handy when you first define your optimizer
        and then your variables
        :param variables: All variables you want to add, if None, all current variables in the session are added
        :post Adds all given variables to the variables list of the optimizer
        """
        assert variables is None or isinstance(variables, (list, tuple))
        assert variables is None or np.all([isinstance(variable, base.Variable)] for variable in variables)
        if variables is None:
            self.variables = self.session.variables
        else:
            for variable in variables:
                self.variables.append(variable)

    def update_variables(self):
        """
        Basic function in which each optimizer will optimizer its variables.
        :return: The clipped gradients for each variable.
        """
        gradients = [variable.get_gradient() for variable in self.variables]
        if self.clipvalue is not None:
            for i in range(len(gradients)):
                gradients[i] = np.clip(gradients[i], self.clipvalue[0], self.clipvalue[1])

        return gradients

    def __str__(self):
        """
        ...
        :return: "optimizer"
        """
        return "optimizer"


class SGD(Optimizer):
    """Class that implements the stochastic gradient descent optimizer"""
    def __init__(self, learning_rate=1e-3, name="SGD", clipvalue=None, variables=None):
        """
        Initializes a new stochastic gradient descent optimizer
        :param learning_rate: The learning rate parameter for the gradient descent
        :param name: The name of the optimizer
        :param clipvalue: An array of size two indicating what the maximum gradient and what the minimum gradient is
        :param variables: The variables the optimizer needs to update, None or list of Variables
        :post Calls the initializer of the optimizer class
        :post Creates a variable registering the learning rate of the optimizer
        """
        Optimizer.__init__(self, name, clipvalue, variables)
        self.lr = learning_rate

    def update_variables(self):
        """
        Updates the variables for each variable
        :post For each variable, changes the value of the variable to - self.lr times the clipped gradient of the
              variable
        """
        gradients = Optimizer.update_variables(self)
        for var, gradient in zip(self.variables, gradients):
            var.add_to_value(-self.lr * gradient)


class MomentumOptimizer(Optimizer):
    """Class that implements the Momentum optimizer"""
    def __init__(self, learning_rate=1e-3, momentum=0.9, name="Momentum optimizer", clipvalue=None, variables=None):
        """
       Initializes a new MomentumOptimizer optimizer
       :param learning_rate: The learning rate parameter for the gradient descent
       :param momentum: The momentum parameter for the momentum optimizer
       :param name: The name of the optimizer
       :param clipvalue: An array of size two indicating what the maximum gradient and what the minimum gradient is
       :param variables: The variables the optimizer needs to update, None or list of Variables
       :post Calls the initializer of the optimizer class
       :post Creates a variable registering the learning rate of the optimizer
       :post Creates a variable registering the momentum of the optimizer
       :post Creates a variable registering the current optimizing speed for each variable.
       """
        Optimizer.__init__(self, name, clipvalue, variables)

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = [0 for _ in range(len(self.variables))]

    def add_variables(self, variables=None):
        """
        Adds new variables for the optimizer to optimize. Comes in handy when you first define your optimizer
        and then your variables
        :param variables: All variables you want to add, if None, all current variables in the session are added
        :post Adds all given variables to the variables list of the optimizer
        :post Adds zero speed to each variable that is added
        """
        Optimizer.add_variables(self, variables)
        while len(self.v) < len(self.variables):
            self.v.append(0)

    def update_variables(self):
        """
        Updates the variables for each variable
        :post For each variable, changes its current speed to (gradient is the clipped gradient):
              momentum * speed + learning_rate * gradients
        :post For each variable, changes the value of the variable by the following amount:
              -speed
        """
        gradients = Optimizer.update_variables(self)
        for i in range(len(self.variables)):
            self.v[i] = self.momentum * self.v[i] + self.learning_rate * gradients[i]
            self.variables[i].add_to_value(-self.v[i])


class NesterovOptimizer(Optimizer):
    """Class that implements the Nesterov optimizer"""
    def __init__(self, learning_rate=1e-3, momentum=0.9, clipvalue=None, name="Nesterov optimizer", variables=None):
        """
       Initializes a new NesterovOptimizer optimizer
       :param learning_rate: The learning rate parameter for the gradient descent
       :param momentum: The momentum parameter for the momentum optimizer
       :param name: The name of the optimizer
       :param clipvalue: An array of size two indicating what the maximum gradient and what the minimum gradient is
       :param variables: The variables the optimizer needs to update, None or list of Variables
       :post Calls the initializer of the optimizer class
       :post Creates a variable registering the learning rate of the optimizer
       :post Creates a variable registering the momentum of the optimizer
       :post Creates a variable registering the current optimizing speed for each variable.
       """
        Optimizer.__init__(self, name, clipvalue, variables)

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = [0 for _ in range(len(self.variables))]

    def add_variables(self, variables=None):
        """
        Adds new variables for the optimizer to optimize. Comes in handy when you first define your optimizer
        and then your variables
        :param variables: All variables you want to add, if None, all current variables in the session are added
        :post Adds all given variables to the variables list of the optimizer
        :post Adds zero speed to each variable that is added
        """
        Optimizer.add_variables(self, variables)
        while len(self.v) < len(self.variables):
            self.v.append(0)

    def update_variables(self):
        """
        Updates the variables for each variable
        :post For each variable, changes its current speed to (gradient is the clipped gradient):
              momentum * speed - learning_rate * gradients
        :post For each variable, changes the value of the variable by the following amount:
              -momentum * old_speed + (1 + momentum) * new_speed
        """
        gradients = Optimizer.update_variables(self)
        for i in range(len(self.variables)):
            previous_speed = np.copy(self.v[i])
            self.v[i] = self.momentum * self.v[i] - self.learning_rate * gradients[i]
            update = -self.momentum * previous_speed + (1 + self.momentum) * self.v[i]
            self.variables[i].add_to_value(update)


class AdaGrad(Optimizer):
    """Class that implements the AdaGrad optimizer"""
    def __init__(self, learning_rate=1, epsilon=1e-8, clipvalue=None, name="Adagrad", variables=None):
        """
        Initializes a new AdaGrad optimizer
       :param learning_rate: The learning rate parameter for the gradient descent
       :param epsilon: The epsilon parameter for the AdaGrad optimizer
       :param name: The name of the optimizer
       :param clipvalue: An array of size two indicating what the maximum gradient and what the minimum gradient is
       :param variables: The variables the optimizer needs to update, None or list of Variables
       :post Calls the initializer of the optimizer class
       :post Creates a variable registering the learning rate of the optimizer
       :post Creates a variable registering the epsilon of the optimizer
       :post Creates a variable registering the current gradient sqaured for each variable.
       """
        Optimizer.__init__(self, name, clipvalue, variables)

        self.learning_rate = learning_rate
        self.grad_squared = [0 for _ in range(len(self.variables))]
        self.epsilon = epsilon

    def add_variables(self, variables=None):
        """
        Adds new variables for the optimizer to optimize. Comes in handy when you first define your optimizer
        and then your variables
        :param variables: All variables you want to add, if None, all current variables in the session are added
        :post Adds all given variables to the variables list of the optimizer
        :post Adds zero gradient squared to each variable that is added
        """
        Optimizer.add_variables(self, variables)
        while len(self.grad_squared) < len(self.variables):
            self.grad_squared.append(0)

    def update_variables(self):
        """
        Updates the variables for each variable
        :post For each variable, changes its current gradient squared to (gradient is the clipped gradient):
              grad_squared + (new_gradient) ** 2
        :post For each variable, changes the value of the variable by the following amount:
              -learning_rate * gradient / sqrt(grad_squared + epsilon)
        """
        gradients = Optimizer.update_variables(self)
        for i in range(len(self.variables)):
            self.grad_squared[i] += np.square(gradients[i])
            update = -self.learning_rate * gradients[i] / (np.sqrt(self.grad_squared[i]) + self.epsilon)
            self.variables[i].add_to_value(update)


class RMSProp(Optimizer):
    """Class that implements the RMSProp optimizer"""
    def __init__(self, learning_rate=1e-3, momentum=0.9, epsilon=1e-8, clipvalue=None, name="RMSProp", variables=None):
        """
        Initializes a new RMSProp optimizer
       :param learning_rate: The learning rate parameter for the gradient descent
       :param momentum: The momentum parameter for the momentum optimizer
       :param epsilon: The epsilon parameter for the AdaGrad optimizer
       :param name: The name of the optimizer
       :param clipvalue: An array of size two indicating what the maximum gradient and what the minimum gradient is
       :param variables: The variables the optimizer needs to update, None or list of Variables
       :post Calls the initializer of the optimizer class
       :post Creates a variable registering the learning rate of the optimizer
       :post Creates a variable registering the epsilon of the optimizer
       :post Creates a variable registering the current gradient sqaured for each variable.
       :post Creates a variable registering the momentum of the optimizer
       """
        Optimizer.__init__(self, name, clipvalue, variables)

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.grad_squared = [0 for _ in range(len(self.variables))]
        self.epsilon = epsilon

    def add_variables(self, variables=None):
        """
        Adds new variables for the optimizer to optimize. Comes in handy when you first define your optimizer
        and then your variables
        :param variables: All variables you want to add, if None, all current variables in the session are added
        :post Adds all given variables to the variables list of the optimizer
        :post Adds zero gradient squared to each variable that is added
        """
        Optimizer.add_variables(self, variables)
        while len(self.grad_squared) < len(self.variables):
            self.grad_squared.append(0)

    def update_variables(self):
        """
        Updates the variables for each variable
        :post For each variable, changes its current gradient squared to (gradient is the clipped gradient):
              momentum * grad_squared + (1 - momentum) * (new_gradient) ** 2
        :post For each variable, changes the value of the variable by the following amount:
              -learning_rate * gradient / sqrt(grad_squared + epsilon)
        """
        gradients = Optimizer.update_variables(self)
        for i in range(len(self.variables)):
            self.grad_squared[i] = self.momentum * self.grad_squared[i] + (1 - self.momentum) * np.square(gradients[i])
            update = -self.learning_rate * gradients[i] / (np.sqrt(self.grad_squared[i]) + self.epsilon)
            self.variables[i].add_to_value(update)


# NOTE TO SELF: This algorithm seems to optimize things, but only very slowly
class AdaDelta(Optimizer):
    """Class that implements the AdaDelta optimizer"""
    def __init__(self, momentum=0.9, epsilon=1e-5, clipvalue=None, name="AdaDelta", variables=None):
        """
        Initializes a new AdaDelta optimizer
       :param momentum: The momentum parameter for the momentum optimizer
       :param epsilon: The epsilon parameter for the AdaGrad optimizer
       :param name: The name of the optimizer
       :param clipvalue: An array of size two indicating what the maximum gradient and what the minimum gradient is
       :param variables: The variables the optimizer needs to update, None or list of Variables
       :post Calls the initializer of the optimizer class
       :post Creates a variable registering the epsilon of the optimizer
       :post Creates a variable registering the current gradient squared for each variable.
       :post Creates a variable registering the current parameter update squared for each variable.
       :post Creates a variable registering the momentum of the optimizer
       """
        Optimizer.__init__(self, name, clipvalue, variables)
        self.momentum = momentum
        self.grad_squared = [0 for _ in range(len(self.variables))]
        self.parameter_updates_squared = [0 for _ in range(len(self.variables))]
        self.epsilon = epsilon

    def add_variables(self, variables=None):
        """
        Adds new variables for the optimizer to optimize. Comes in handy when you first define your optimizer
        and then your variables
        :param variables: All variables you want to add, if None, all current variables in the session are added
        :post Adds all given variables to the variables list of the optimizer
        :post Adds zero gradient squared to each variable that is added
        :post Adds zero parameter_updates_squared to each variable that is added
        """
        Optimizer.add_variables(self, variables)
        while len(self.grad_squared) < len(self.variables):
            self.grad_squared.append(0)
            self.parameter_updates_squared.append(0)

    def update_variables(self):
        """
        Updates the variables for each variable
        :post For each variable, changes its current gradient squared to (gradient is the clipped gradient):
              momentum * grad_squared + (1 - momentum) * (new_gradient) ** 2
        :post For each variable, it performs an update by the following amount:
              - sqrt(parameter_updates_squared + epsilon) / sqrt(gradients_squared + epsilon)
        :post For each variable, changes its current parameter update squared to:
              momentum * parameter_updates_squared + (1 - momentum) * (update) ** 2
        """
        gradients = Optimizer.update_variables(self)

        for i in range(len(self.variables)):
            self.grad_squared[i] = self.momentum * self.grad_squared[i] + (1 - self.momentum) * np.square(gradients[i])
            update = -(np.sqrt(self.parameter_updates_squared[i]) + self.epsilon) * gradients[i] / \
                      (np.sqrt(self.grad_squared[i]) + self.epsilon)

            self.variables[i].add_to_value(update)

            self.parameter_updates_squared[i] = self.momentum * self.parameter_updates_squared[i] + \
                                                (1 - self.momentum) * np.square(update)


class Adam(Optimizer):
    """Class that implements the Adam optimizer"""
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-7, clipvalue=None, name="Adam",
                 variables=None):
        """
        Initializes a new Adam optimizer
       :param learning_rate: The learning rate parameter for the Adam optimizer
       :param beta1: The beta1 parameter for the Adam optimizer
       :param beta2: The beta2 parameter for the Adam optimizer
       :param epsilon: The epsilon parameter for the Adam optimizer
       :param name: The name of the optimizer
       :param clipvalue: An array of size two indicating what the maximum gradient and what the minimum gradient is
       :param variables: The variables the optimizer needs to update, None or list of Variables
       :post Calls the initializer of the optimizer class
       :post Creates a variable registering the learning_rate of the optimizer
       :post Creates a variable registering the beta1 of the optimizer
       :post Creates a variable registering the beta2 of the optimizer
       :post Creates a variable registering the current parameter update squared for each variable.
       :post Creates a variable registering the epsilon of the optimizer
       :post Creates a variable registering the current optimizing speed for each variable.
       :post Creates a variable registering the current optimizing momentum for each variable.
       :post Creates a variable registering the time.
       """
        Optimizer.__init__(self, name, clipvalue, variables)

        self.learning_rate = learning_rate
        self.time = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = [0 for _ in range(len(self.variables))]
        self.v = [0 for _ in range(len(self.variables))]

    def add_variables(self, variables=None):
        """
        Adds new variables for the optimizer to optimize. Comes in handy when you first define your optimizer
        and then your variables
        :param variables: All variables you want to add, if None, all current variables in the session are added
        :post Adds all given variables to the variables list of the optimizer
        :post Adds zero momentum to each variable that is added
        :post Adds zero speed to each variable that is added
        """
        Optimizer.add_variables(self, variables)
        while len(self.m) < len(self.variables):
            self.m.append(0)
            self.v.append(0)

    def update_variables(self):
        """
        Updates the variables for each variable
        :post For each variable, changes its current momentum to (gradient is the clipped gradient):
              beta1 * momentum + (1 - beta1) * gradient
        :post For each variable, changes its current speed to (gradient is the clipped gradient):
              beta2 * speed + (1 - beta2) * gradient ** 2
        :post For each variable, it performs an update by the following amount:
              - learning_rate * (momentum / (1 - beta1 * time)) / (sqrt(speed / (1 - beta2 * time)) + epsilon)
        """
        gradients = Optimizer.update_variables(self)
        self.time += 1
        for i in range(len(self.variables)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * gradients[i] ** 2
            first_unbias = self.m[i] / (1 - self.beta1 ** self.time)
            second_unbias = self.v[i] / (1 - self.beta2 ** self.time)
            update = - self.learning_rate * first_unbias / (np.sqrt(second_unbias) + self.epsilon)
            self.variables[i].add_to_value(update)


class AdaMax(Optimizer):
    """Class that implements the AdaMax optimizer"""
    def __init__(self, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-7,
                 clipvalue=None, name="AdaMax", variables=None):
        """
        Initializes a new AdaMax optimizer
       :param learning_rate: The learning rate parameter for the AdaMax optimizer
       :param beta1: The beta1 parameter for the AdaMax optimizer
       :param beta2: The beta2 parameter for the AdaMax optimizer
       :param name: The name of the optimizer
       :param epsilon: Smoothign value for the division in the update of the AdaMax optimizer
       :param clipvalue: An array of size two indicating what the maximum gradient and what the minimum gradient is
       :param variables: The variables the optimizer needs to update, None or list of Variables
       :post Calls the initializer of the optimizer class
       :post Creates a variable registering the learning_rate of the optimizer
       :post Creates a variable registering the beta1 of the optimizer
       :post Creates a variable registering the beta2 of the optimizer
       :post Creates a variable registering the current parameter update squared for each variable.
       :post Creates a variable registering the current optimizing speed for each variable.
       :post Creates a variable registering the current optimizing momentum for each variable.
       :post Creates a variable registering the time.
       :post Creates a variable registering epsilon.
       """
        Optimizer.__init__(self, name, clipvalue, variables)

        self.learning_rate = learning_rate
        self.time = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = [0 for _ in range(len(self.variables))]
        self.v = [0 for _ in range(len(self.variables))]

    def add_variables(self, variables=None):
        """
        Adds new variables for the optimizer to optimize. Comes in handy when you first define your optimizer
        and then your variables
        :param variables: All variables you want to add, if None, all current variables in the session are added
        :post Adds all given variables to the variables list of the optimizer
        :post Adds zero momentum to each variable that is added
        :post Adds zero speed to each variable that is added
        """
        Optimizer.add_variables(self, variables)
        while len(self.m) < len(self.variables):
            self.m.append(0)
            self.v.append(0)

    def update_variables(self):
        """
        Updates the variables for each variable
        :post For each variable, changes its current momentum to (gradient is the clipped gradient):
              beta1 * momentum + (1 - beta1) * gradient
        :post For each variable, changes its current speed to (gradient is the clipped gradient):
              maximum(beta2 * speed, |gradient|)
        :post For each variable, it performs an update by the following amount:
              - learning_rate * (momentum / (1 - beta1 * time)) / (speed + epsilon)
        """
        gradients = Optimizer.update_variables(self)
        self.time += 1
        for i in range(len(self.variables)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients[i]
            self.v[i] = np.maximum(self.beta2 * self.v[i], np.abs(gradients[i]))
            first_unbias = self.m[i] / (1 - self.beta1 ** self.time)
            update = - self.learning_rate * first_unbias / (self.v[i] + self.epsilon)
            self.variables[i].add_to_value(update)


class Nadam(Optimizer):
    """Class that implements the Nadam optimizer"""
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-7, clipvalue=None, name="Nadam",
                 variables=None):
        """
        Initializes a new Nadam optimizer
       :param learning_rate: The learning rate parameter for the Nadam optimizer
       :param beta1: The beta1 parameter for the Nadam optimizer
       :param beta2: The beta2 parameter for the Nadam optimizer
       :param name: The name of the optimizer
       :param clipvalue: An array of size two indicating what the maximum gradient and what the minimum gradient is
       :param variables: The variables the optimizer needs to update, None or list of Variables
       :post Calls the initializer of the optimizer class
       :post Creates a variable registering the learning_rate of the optimizer
       :post Creates a variable registering the beta1 of the optimizer
       :post Creates a variable registering the beta2 of the optimizer
       :post Creates a variable registering the epsilon of the optimizer
       :post Creates a variable registering the current parameter update squared for each variable.
       :post Creates a variable registering the current optimizing speed for each variable.
       :post Creates a variable registering the current optimizing momentum for each variable.
       :post Creates a variable registering the time.
       """
        Optimizer.__init__(self, name, clipvalue, variables)

        self.learning_rate = learning_rate
        self.time = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = [0 for _ in range(len(self.variables))]
        self.v = [0 for _ in range(len(self.variables))]

    def add_variables(self, variables=None):
        """
        Adds new variables for the optimizer to optimize. Comes in handy when you first define your optimizer
        and then your variables
        :param variables: All variables you want to add, if None, all current variables in the session are added
        :post Adds all given variables to the variables list of the optimizer
        :post Adds zero momentum to each variable that is added
        :post Adds zero speed to each variable that is added
        """
        Optimizer.add_variables(self, variables)
        while len(self.m) < len(self.variables):
            self.m.append(0)
            self.v.append(0)

    def update_variables(self):
        """
        Updates the variables for each variable
        :post For each variable, changes its current momentum to (gradient is the clipped gradient):
              beta1 * momentum + (1 - beta1) * gradient
        :post For each variable, changes its current speed to (gradient is the clipped gradient):
              beta2 * speed + (1 - beta2) * gradient ** 2
        :post For each variable, it performs an update by the following amount:
              - learning_rate * term1 / term2
              term1 = (beta1 * momentum / (1 - beta1 ** time) + (1 - beta1) * gradients[i] / (1 - beta1 ** time))
              term2 = speed / (1 - beta2 ** time) + epsilon
        """
        gradients = Optimizer.update_variables(self)
        self.time += 1
        for i in range(len(self.variables)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * gradients[i] ** 2
            first_unbias = self.m[i] / (1 - self.beta1 ** self.time)
            second_unbias = self.v[i] / (1 - self.beta2 ** self.time)
            update = - self.learning_rate / (np.sqrt(second_unbias) + self.epsilon) * \
                     (self.beta1 * first_unbias + (1 - self.beta1) * gradients[i] / (1 - self.beta1 ** self.time))
            self.variables[i].add_to_value(update)


class AMSProp(Optimizer):
    """Class that implements the AMSProp optimizer"""
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-7, clipvalue=None, name="AMSProp",
                 variables=None):
        """
        Initializes a new AMSProp optimizer
       :param learning_rate: The learning rate parameter for the AMSProp optimizer
       :param beta1: The beta1 parameter for the AMSProp optimizer
       :param beta2: The beta2 parameter for the AMSProp optimizer
       :param name: The name of the optimizer
       :param clipvalue: An array of size two indicating what the maximum gradient and what the minimum gradient is
       :param variables: The variables the optimizer needs to update, None or list of Variables
       :post Calls the initializer of the optimizer class
       :post Creates a variable registering the learning_rate of the optimizer
       :post Creates a variable registering the beta1 of the optimizer
       :post Creates a variable registering the beta2 of the optimizer
       :post Creates a variable registering the epsilon of the optimizer
       :post Creates a variable registering the current parameter update squared for each variable.
       :post Creates a variable registering the current optimizing speed for each variable.
       :post Creates a variable registering the current optimizing speed2 for each variable.
       :post Creates a variable registering the current optimizing momentum for each variable.
       :post Creates a variable registering the time.
       """
        Optimizer.__init__(self, name, clipvalue, variables)

        self.learning_rate = learning_rate
        self.t = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = [0 for _ in range(len(self.variables))]
        self.v = [0 for _ in range(len(self.variables))]
        self.v2 = [0 for _ in range(len(self.variables))]

    def add_variables(self, variables=None):
        """
        Adds new variables for the optimizer to optimize. Comes in handy when you first define your optimizer
        and then your variables
        :param variables: All variables you want to add, if None, all current variables in the session are added
        :post Adds all given variables to the variables list of the optimizer
        :post Adds zero momentum to each variable that is added
        :post Adds zero speed to each variable that is added
        :post Adds zero speed2 to each variable that is added
        """
        Optimizer.add_variables(self, variables)
        while len(self.m) < len(self.variables):
            self.m.append(0)
            self.v.append(0)
            self.v2.append(0)

    def update_variables(self):
        """
        Updates the variables for each variable
        :post For each variable, changes its current momentum to (gradient is the clipped gradient):
              beta1 * momentum + (1 - beta1) * gradient
        :post For each variable, changes its current speed to (gradient is the clipped gradient):
              beta2 * speed + (1 - beta2) * gradient ** 2
        :post For each variable, changes its current speed2 to (gradient is the clipped gradient):
              maximum(speed2, speed)
        :post For each variable, it performs an update by the following amount:
              - learning_rate * momentum / (sqrt(speed2 + epsilon))
        """
        gradients = Optimizer.update_variables(self)
        self.t += 1
        for i in range(len(self.variables)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * gradients[i] ** 2
            self.v2[i] = np.maximum(self.v2[i], self.v[i])
            update = - self.learning_rate * self.m[i] / (np.sqrt(self.v2[i]) + self.epsilon)
            self.variables[i].add_to_value(update)
