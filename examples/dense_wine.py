import kflow.base as Base
import kflow.layers as Layers
import kflow.operations as ops
import kflow.optimizers as optim
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np

# NOTE TO SELF: overflows in Softmax are quite common, only way to avoid them here is by adding batchNormalization
# NOTE TO SELF: Reasonf for the above is just because the data is not changed such that the mean is 0 and std is 1
# NOTE TO SELF: But I get really bad results if I normalize the input.
np.random.seed(42)
data = load_wine()
batch_size = 16
epochs = 300
n_inputs = 13
n_outputs = 3
X = data.data
mean_X = X.mean(axis=1, keepdims=True)
std_X = X.std(axis=1, keepdims=True)
X = (X - mean_X) / std_X
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

print("Lenght of training data: {}".format(len(X_train)))


def accuracy(y_true, y_pred):
    prediction = np.argmax(y_pred, axis=1)
    incorrect = np.count_nonzero(y_true - prediction)
    return float(len(y_true) - incorrect) / len(y_true)


def to_categorical(y_true):
    cat = np.zeros((y_true.size, n_outputs), dtype=int)
    cat[np.arange(y_true.size), y_true] = 1
    return cat


y_train_cat = to_categorical(y_train)

session = Base.Session()

X = Base.Placeholder((batch_size, n_inputs))
y = Base.Placeholder((batch_size, n_outputs))
training = Base.Placeholder(1)
training.set_value(True)
layer1 = Layers.Dense(X, 10, activation=ops.ReLU)
op = ops.Power(layer1, 2)
op2 = ops.Power(layer1, 3)
op3 = ops.Concatenate(layer1, op, axis=1)
op4 = ops.Concatenate(op3, op2, axis=1)
output = Layers.Dense(op4, n_outputs, batch_norm=ops.BatchNormalization, training=training, activation=ops.Softmax)
loss = ops.CategoricalCrossentropy(y, output)
optimizer = optim.Adam()

for epoch in range(epochs):
    rnd_idx = np.random.permutation(len(X_train))
    X_train = X_train[rnd_idx]
    y_train_cat = y_train_cat[rnd_idx]
    y_train = y_train[rnd_idx]
    rnd_idx = np.random.permutation(len(X_test))
    X_test = X_test[rnd_idx]
    y_test = y_test[rnd_idx]
    acc_test = 0
    acc_train = 0
    avg_loss = 0
    training.set_value(True)
    for batch in range(len(X_train) // batch_size):
        x_batch = X_train[batch * batch_size: (batch + 1) * batch_size]
        y_batch = y_train_cat[batch * batch_size: (batch + 1) * batch_size]
        session.run([loss, optimizer], placeholder_values=[[X, x_batch], [y, y_batch]])
        avg_loss += loss.get_value()
        predicted = output.get_value()
        acc_train += accuracy(y_train[batch * batch_size: (batch + 1) * batch_size], predicted)

    training.set_value(False)
    session.forward([output], placeholder_values=[[X, X_test]])
    predicted = output.get_value()
    acc_test = accuracy(y_test, predicted)
    acc_train /= len(X_train) // batch_size
    avg_loss /= len(X_train) // batch_size

    print("Epoch {}".format(epoch))
    print("Average loss training: {}".format(avg_loss))
    print("Average acc training: {}".format(acc_train))
    print("Average acc test: {}".format(acc_test))