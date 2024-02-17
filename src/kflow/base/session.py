from .. import base
import kflow
import numpy as np

class Session:
    """
    The Session class is the basic class in which everything happens. The optimizers, variables and every other type
    is stored in a session and the flow is controlled from the session. With the forward function, it is possible
    to do a forward pass through the entire flow and with the backward function, it is possible to do a backward pass
    through the entire flow.

    Note: the flow is kept as a list and this causes a little bit of overhead. If for example the user asks to just
    evaluate the third element of the flow and this third element doesn't need the first two elements for its evaluation,
    the session will still evaluate the first two elements of the flow.
    """

    """
    A static variable registering the current open session.
    """
    open_session = None

    def __init__(self):
        """
        Initializes the session.
        """

        # List of BaseElements, contains all variables, placeholders, operations and constants that need to be
        # run in the session.
        self.flow = []
        # List of strings, the i-th element of this corresponds with the i-th element in the flow
        self.names_in_flow = []
        # List of all variables in the flow. These are all the elements that can be optimized
        self.variables = []
        # List of all optimizers used for optimizing the variables
        self.optimizers = []
        # List of the names of the optimizers, the i-th element of this list corresponds to the i-th element of the
        # optimizers list.
        self.names_optimizers = []
        # Sets the open session to this session
        Session.open_session = self

    def add_to_flow(self, element, name):
        """
        Adds a new element to the flow of the session.
        :param element: A BaseElement that needs to be added to the flow
        :param name: The name of the element
        :pre The name must be a string
        :pre the element must be a BaseElement
        :post The element will have a new name, consisting of the given name with a simple suffix behind it.
              This will make the name of the element unique in its session
        :post The element will be added to the flow
        :post The new name of the element will be added to names_in_flow
        :post If the element is a variable, the variable will be added to the variables list of the session.
        """
        assert isinstance(name, str)
        assert isinstance(element, base.BaseElement)

        if name is None:
            name = ""

        suffix_number = 0
        while name + "_" + str(suffix_number) in self.names_in_flow:
            suffix_number += 1

        name += "_" + str(suffix_number)
        self.flow.append(element)
        self.names_in_flow.append(name)

        if isinstance(element, base.Variable):
            self.variables.append(element)

        element.name = name

    def reset(self):
        """
        Resets the session.
        :post The flow, names_in_flow, variables, optimizers and names_optimizers will all be empty
        """
        self.flow = []
        self.names_in_flow = []
        self.variables = []
        self.optimizers = []
        self.names_optimizers = []

    def add_optimizer(self, optimizer, name):
        """
        Adds an optimizer to the list containing all optimizers of the session.
        :param optimizer: The optimizer to add
        :param name: The current name of the optimizer
        :pre The name must be a string
        :pre the optimizer must be an Optimizer
        :post The optimizer will have a new name, consisting of the given name with a simple suffix behind it.
              This will make the name of the optimizer unique in its session
        :post The optimizer will be added to the optimizers
        :post The new name of the optimizer will be added to names_optimizers
        """
        assert isinstance(optimizer, kflow.optimizers.Optimizer)
        assert isinstance(name, str)

        if name is None:
            name = ""

        suffix_number = 0
        while name + "_" + str(suffix_number) in self.names_optimizers:
            suffix_number += 1

        name += "_" + str(suffix_number)
        self.optimizers.append(optimizer)
        self.names_optimizers.append(name)
        optimizer.name = name

    def get_by_name(self, name):
        """
        Gets an element in the flow by the name of that element in the flow.
        :param name: The name of the element, string
        :return: If there exists an element with the given name in the flow, it returns this element. Otherwise
                 the function returns None
        """
        assert isinstance(name, str)
        for element in self.flow:
            if element.name == name:
                return element
        return None

    def forward(self, elements, placeholder_values=None):
        """
        Executes a forward pass through the flow.
        :param elements: A list of elements in the flow that need to be run in this forward pass.
        :param placeholder_values: A list consisting of tuples, each tuple must be of length 2 and consists out of
               a basic element and the value the basic element needs to have in this forward pass
        :pre All given elements must be elements of the BaseElement
        :pre Each element in placeholder_values must have length 2, the first of these two needs to be a BaseElement
        :post The value of each given placeholder is set to the corresponding value
        :post The forward function is called on each element in the flow until all elements in the values are run.
        """
        assert np.all([isinstance(value, base.BaseElement) for value in elements])
        assert placeholder_values is None or np.all(
            [len(element) == 2 and isinstance(element[0], base.BaseElement) for element in placeholder_values])

        if placeholder_values is not None:
            for i in range(len(placeholder_values)):
                placeholder_values[i][0].set_value(placeholder_values[i][1])

        ran = 0
        names_values = [element.name for element in elements]
        current_op = 0

        while ran < len(elements):
            self.flow[current_op].forward()
            if self.flow[current_op].name in names_values:
                ran += 1
            current_op += 1

    def backward(self, elements=None, placeholder_gradient=None):
        """
        Runs a backward pass through the entire flow.
        :param elements: List of all Elements for which the backward pass must be run.
        :param placeholder_gradient: A list consisting of tuples, each tuple must be of length 2 and consists out of
               a BaseElement and the gradient the BaseElement needs to have in this backward pass
        :pre All given values must be elements of the BaseElement
        :pre Each element in placeholder_gradient must have length 2, the first of these two needs to be a BaseElement
        :post The value of each given placeholder is set to the corresponding gradient
        :post The backward function is called on each element in the flow until all elements in the variables are run.
        """
        if elements is None:
            elements = self.flow

        assert np.all([isinstance(value, base.BaseElement) for value in elements])
        assert placeholder_gradient is None or np.all(
            [len(element) == 2 and isinstance(element[0], base.BaseElement) for element in placeholder_gradient])
        if placeholder_gradient is not None:
            for i in range(len(placeholder_gradient)):
                placeholder_gradient[i][0].set_gradient(placeholder_gradient[i][1])

        ran = 0
        names_values = [value.name for value in elements]
        current_op = -1

        while ran < len(elements):
            self.flow[current_op].backward()

            if self.flow[current_op].name in names_values:
                ran += 1
            current_op -= 1

    def run(self, elements, placeholder_values=None, placeholder_gradients=None):
        """
        Runs a forward and a backward pass on all elements in the flow. It also runs the optimizer on all variables.
        :param elements: The elements that need to be run in the forward pass.
        :param placeholder_values: A list consisting of tuples, each tuple must be of length 2 and consists out of
               a placeholder and the value the placeholder needs to have in this forward pass
        :param placeholder_gradients: A list consisting of tuples, each tuple must be of length 2 and consists out of
               a placeholder and the gradient the placeholder needs to have in this backward pass
        :pre Each element in elements must be an optimizer or a basic element
        :post Runs self.forward with the given placeholder_values and the elements in the given elements that are basic
              elements.
        :post Afterwards, calls the backward function of the Session class with the given placeholder_gradients
        :post Afterwards, calls the update_variables() function of each optimizer of the given optimizers
        :return: The value of all elements in the given elements that are BaseElements
        """
        assert np.all([isinstance(element, (base.BaseElement, kflow.optimizers.Optimizer)) for element in elements])
        output = None

        non_optimizers = [element for element in elements if isinstance(element, base.BaseElement)]
        if len(non_optimizers) > 0:
            self.forward(non_optimizers, placeholder_values)
            output = [non_optimizer.get_value() for non_optimizer in non_optimizers]

            optimizers = [optimizer for optimizer in elements if isinstance(optimizer, kflow.optimizers.Optimizer)]

            self.backward(placeholder_gradient=placeholder_gradients)
            for optimizer in optimizers:
                optimizer.update_variables()

        return output