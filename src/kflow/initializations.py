import numpy as np
from scipy.stats import ortho_group


def constant_initialization(shape, constant=0):
    """
    Returns a constant array with the given shape
    :param shape: The shape of the return array
    :param constant: The constant each value of the array needs to have
    :return: constant * np.ones(shape)
    """
    return constant * np.ones(shape)


def uniform_initialization(shape, mini=-1, maxi=1):
    """
    Returns a uniform distribution in the given shape
    :param shape: The shape of the return array
    :param mini: The minimal value that can appear in the array
    :param maxi: The maximal value that can appear in the array
    :return: np.random.uniform(mini, maxi, shape)
    """
    return np.random.uniform(mini, maxi, shape)


def normal_initialization(shape, mean=0, std=1):
    """
    Returns a normal distribution in the given shape
    :param shape: The shape of the return array
    :param mean: The mean of the normal distribution
    :param std: The standard deviation of the normal distribution
    :return: np.random.normal(mean, std, shape)
    """
    return np.random.normal(mean, std, shape)


def xavier_normal_initialization(shape, n_inputs=None, n_outputs=None):
    """
    Returns a Xavier normal distribution in the given shape
    :param shape: The shape of the return array
    :param n_inputs: The number of inputs the array has
    :param n_outputs: The number of outputs the array needs to have
    :return: np.random.normal(0, np.sqrt(2 / (n_inputs + n_outputs)), shape)
    """
    if n_inputs is None:
        n_inputs = shape[0]
    if n_outputs is None:
        n_outputs = shape[1]

    return np.random.normal(0, np.sqrt(2 / (n_inputs + n_outputs)), shape)


def xavier_uniform_initialization(shape, n_inputs=None, n_outputs=None):
    """
    Returns a Xavier uniform distribution in the given shape
    :param shape: The shape of the return array
    :param n_inputs: The number of inputs the array has
    :param n_outputs: The number of outputs the array needs to have
    :return: np.random.uniform(-np.sqrt(6 / (n_inputs + n_outputs)), np.sqrt(6 / (n_inputs + n_outputs)), shape)
    """
    if n_inputs is None:
        n_inputs = shape[0]
    if n_outputs is None:
        n_outputs = shape[1]

    max_range = np.sqrt(6 / (n_inputs + n_outputs))
    return np.random.uniform(-max_range, max_range, shape)


def he_normal_initialization(shape, n_inputs=None, n_outputs=None):
    """
    Returns a He normal distribution in the given shape
    :param shape: The shape of the return array
    :param n_inputs: The number of inputs the array has
    :param n_outputs: The number of outputs the array needs to have
    :return: np.random.normal(0, 2 * np.sqrt(1 / (n_inputs + n_outputs)), shape)
    """
    if n_inputs is None:
        n_inputs = shape[0]
    if n_outputs is None:
        n_outputs = shape[1]

    return np.random.normal(0, 2 * np.sqrt(1 / (n_inputs + n_outputs)), shape)


def he_uniform_initialization(shape, n_inputs=None, n_outputs=None):
    """
    Returns a He uniform distribution in the given shape
    :param shape: The shape of the return array
    :param n_inputs: The number of inputs the array has
    :param n_outputs: The number of outputs the array needs to have
    :return: np.random.uniform(-np.sqrt(12 / (n_inputs + n_outputs)), np.sqrt(12 / (n_inputs + n_outputs)), shape)
    """
    if n_inputs is None:
        n_inputs = shape[0]
    if n_outputs is None:
        n_outputs = shape[1]

    max_range = np.sqrt(12 / (n_inputs + n_outputs))
    return np.random.uniform(-max_range, max_range, shape)


def orthogonal_initialization(shape):
    """
    Returns a orthogonal in the given shape
    :param shape: The shape of the return array
    :return: returns a square orthogonal matrix of size (n_inputs, n_inputs)
    """
    if shape[0] <= 1:
        return np.zeros((1, 1))  # normally this isn't correct, because it isn't orthogonal, but so be it
    return ortho_group.rvs(shape[0])
