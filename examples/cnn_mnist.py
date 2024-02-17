import kflow.base as Base
import kflow.layers as Layers
import kflow.operations as Ops
import kflow.optimizers as Optim
from keras.utils import to_categorical
from keras.datasets import mnist
import time
import numpy as np

session = Base.Session()
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype("float32") / 255
X_train = X_train.reshape(-1, 28, 28, 1)
X_train_flat = X_train.reshape((len(X_train), 28 * 28))
X_test = X_test.astype("float32") / 255
X_test = X_test.reshape(-1, 28, 28, 1)
X_test_flat = X_test.reshape((len(X_test), 28 * 28))

y_train_cat = to_categorical(y_train)[:len(X_train)]
y_test_cat = to_categorical(y_test)[:len(X_test)]
batch_size = 50


def accuracy(y_true, y_pred):
    prediction = np.round(y_pred).astype(np.int32)
    incorrect = np.count_nonzero(y_true - prediction)
    return float(np.prod(y_true.shape) - incorrect) / np.prod(y_true.shape)


session.reset()

X = Base.Placeholder((batch_size, 28, 28, 1))
y = Base.Placeholder((batch_size, 10))

training = Base.Placeholder(1)
training.set_value(True)
layer1 = Layers.Conv2D(X, n_filters=16, padding=2, kernel_size=(3, 3), batch_norm=Ops.BatchNormalization, strides=(1, 1),
                       activation=Ops.ReLU, training=training, dropout_rate=0.3)
pooling1 = Layers.AveragePooling(layer1, (2, 2))
layer2 = Layers.Conv2D(pooling1, n_filters=8, kernel_size=(3, 3), batch_norm=Ops.BatchNormalization, strides=(1, 1),
                       activation=Ops.ReLU, training=training, dropout_rate=0.3)
pooling2 = Layers.AveragePooling(layer2, (2, 2))
flatten = Ops.Flatten(pooling2)
dense1 = Layers.Dense(flatten, 50, activation=Ops.ReLU, batch_norm=Ops.BatchNormalization, training=training)
output = Layers.Dense(dense1, 10, activation=Ops.Softmax)

loss = Ops.CategoricalCrossentropy(y, output)

optimizer = Optim.Adam()

epochs = 10

for epoch in range(epochs):
    rnd_idx = np.random.permutation(len(X_train))
    X_train_flat, X_train, y_train_cat, y_train = X_train_flat[rnd_idx], X_train[rnd_idx], y_train_cat[rnd_idx], \
                                                  y_train[rnd_idx]

    acc_train = 0
    avg_loss = 0
    t = time.time()
    training.set_value(True)
    for batch in range(0, len(X_train_flat) // batch_size):
        x_batch = X_train[batch * batch_size: (batch + 1) * batch_size]
        y_batch = y_train_cat[batch * batch_size: (batch + 1) * batch_size]
        los = session.run([loss, optimizer], placeholder_values=[[X, x_batch], [y, y_batch]])[0]
        avg_loss += los
        predicted = output.get_value()
        acc_train += accuracy(y_train_cat[batch * batch_size: (batch + 1) * batch_size], predicted)

    training.set_value(False)
    session.forward([output], placeholder_values=[[X, X_test], [training, False]])
    predicted = output.get_value()
    acc_test = accuracy(y_test_cat, predicted)
    acc_train /= len(X_train) // batch_size
    avg_loss /= len(X_train) // batch_size
    print("Epoch {}".format(epoch+1))
    print("Time: {}".format(round(time.time() - t, 2)))
    print("Average loss training: {}".format(round(float(avg_loss), 4)))
    print("Average acc training: {}".format(round(float(acc_train), 4)))
    print("Average acc test: {}".format(round(float(acc_test), 4)))
