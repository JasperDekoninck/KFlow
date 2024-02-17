import kflow.base as Base
import kflow.layers as Layers
import kflow.operations as ops
import kflow.optimizers as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

np.random.seed(42)
data = load_breast_cancer()
batch_size = 16
epochs = 100
n_inputs = 30
n_outputs = 1
X = data.data
mean_X = X.mean(axis=1, keepdims=True)
std_X = X.std(axis=1, keepdims=True)
X = (X - mean_X) / std_X
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y)


print("Lenght of training data: {}".format(len(X_train)))


def accuracy(y_true, y_pred):
    prediction = np.round(y_pred).astype(np.int32)
    incorrect = np.count_nonzero(y_true - prediction)
    return float(np.prod(y_true.shape) - incorrect) / np.prod(y_true.shape)


y_train_cat = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

session = Base.Session()

X = Base.Placeholder((batch_size, n_inputs))
y = Base.Placeholder((batch_size, n_outputs))
training = Base.Placeholder(1)
training.set_value(True)
layer1 = Layers.Dense(X, 30, activation=ops.Swish, batch_norm=ops.BatchNormalization,
                      training=training)
layer2 = Layers.Dense(layer1, 20, activation=ops.Swish, batch_norm=ops.BatchNormalization,
                      training=training)
layer3 = Layers.Dense(layer2, 10, activation=ops.Swish, batch_norm=ops.BatchNormalization,
                      training=training)
output = Layers.Dense(layer3, n_outputs, activation=ops.Sigmoid)
loss = ops.BinaryCrossentropy(y, output)
optimizer = optim.Adam()

for epoch in range(epochs):
    rnd_idx = np.random.permutation(len(X_train))
    X_train = X_train[rnd_idx]
    y_train_cat = y_train_cat[rnd_idx]
    rnd_idx = np.random.permutation(len(X_test))
    X_test = X_test[rnd_idx]
    y_test = y_test[rnd_idx]
    acc_test = 0
    acc_train = 0
    avg_loss = 0
    for batch in range(len(X_train) // batch_size):
        x_batch = X_train[batch * batch_size: (batch + 1) * batch_size]
        y_batch = y_train_cat[batch * batch_size: (batch + 1) * batch_size]
        session.run([loss, optimizer], placeholder_values=[[X, x_batch], [y, y_batch], [training, True]])
        avg_loss += loss.get_value()
        predicted = output.get_value()
        acc_train += accuracy(y_train_cat[batch * batch_size: (batch + 1) * batch_size], predicted)

    session.forward([output], placeholder_values=[[X, X_test], [training, False]])
    predicted = output.get_value()
    acc_test = accuracy(y_test, predicted)

    acc_train /= len(X_train) // batch_size
    avg_loss /= len(X_train) // batch_size

    print("Epoch {}".format(epoch))
    print("Average loss training: {}".format(avg_loss))
    print("Average acc training: {}".format(acc_train))
    print("Average acc test: {}".format(acc_test))
