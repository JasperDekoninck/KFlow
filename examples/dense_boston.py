import kflow.base as Base
import kflow.layers as Layers
import kflow.operations as ops
import kflow.optimizers as optim
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(42)
data = load_boston()
batch_size = 16
epochs = 200
n_inputs = 13
n_outputs = 1
X = data.data
mean_X = X.mean(axis=1, keepdims=True)
std_X = X.std(axis=1, keepdims=True)
X = (X - mean_X) / std_X
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y)


print("Lenght of training data: {}".format(len(X_train)))


# NOTE TO SELF: reshaping is super important
y_train_cat = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


session = Base.Session()

X = Base.Placeholder((batch_size, n_inputs))
y = Base.Placeholder((batch_size, n_outputs))
training = Base.Placeholder(1)
training.set_value(True)
layer1 = Layers.Dense(X, 20, activation=ops.LeakyReLU, dropout_rate=0.6, batch_norm=ops.BatchNormalization,
                      training=training, weight_regularizer=ops.L1L2Regularizer, bias_regularizer=ops.L2Regularizer)
layer2 = Layers.Dense(X, 15, activation=ops.ELU)
output = Layers.Dense(layer2, n_outputs)
loss = ops.Mae(y, output)
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
    avg_loss_test = 0
    training.set_value(True)
    for batch in range(len(X_train) // batch_size):
        x_batch = X_train[batch * batch_size: (batch + 1) * batch_size]
        y_batch = y_train_cat[batch * batch_size: (batch + 1) * batch_size]
        session.run([loss, optimizer], placeholder_values=[[X, x_batch], [y, y_batch]])
        avg_loss += loss.get_value()
        predicted = output.get_value()

    training.set_value(False)
    session.forward([loss], placeholder_values=[[X, X_test], [y, y_test]])
    avg_loss_test = loss.get_value()
    avg_loss /= len(X_train) // batch_size

    print("Epoch {}".format(epoch))
    print("Average loss training: {}".format(avg_loss))
    print("Average loss test: {}".format(avg_loss_test))
