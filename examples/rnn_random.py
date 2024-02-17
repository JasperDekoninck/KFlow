import kflow.base as Base
import kflow.layers as Layers
import kflow.operations as ops
import kflow.optimizers as optim
import time
import numpy as np

X_train = np.random.uniform(-1.0, 1.0, size=(1000, 5, 5))
y_train = X_train.sum(axis=2, keepdims=False).sum(axis=1, keepdims=True)
session = Base.Session()
batch_size = 32
X = Base.Placeholder((batch_size, 5, 5), name="X")
y = Base.Placeholder((batch_size, 1), name="y")


X_ = Layers.BasicRNNCell(X, 1, activation=ops.DoNothing, return_sequences=False)
loss = ops.Mse(X_, y, name="mse")
optimizer = optim.Adam(0.01, clipvalue=[-10, 10])

t = time.time()
epochs = 20

for epoch in range(epochs):
    avg_loss = 0
    for batch in range(0, len(X_train)//batch_size):
        x_batch = X_train[batch * batch_size: (batch + 1) * batch_size]
        y_batch = y_train[batch * batch_size: (batch + 1) * batch_size]
        los = session.run([loss, optimizer], placeholder_values=[[X, x_batch], [y, y_batch]])[0]
        avg_loss += los

    print(avg_loss / (len(X_train)//batch_size))

print(time.time() - t)
