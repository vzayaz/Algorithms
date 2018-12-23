"""
A naive implementaion of backpropagation
"""

import numpy as np

np.random.seed(1007)


class NetComponent:
    """
    An abstraction of a NN building block.
    It has an input-output shape and is aware of existence of batches
    """

    def __init__(self, n, m, batch):
        """
        Init a layer with n input m outputs
        :param n: input size
        :param m: output size
        :param batch: the size of the batch
        """
        self.n = n
        self.m = m
        self.batch = batch

    @property
    def shape(self):
        return self.n, self.m


class FullyConnected(NetComponent):
    """
    A fully connected layer
    """
    def __init__(self, n, m, batch, activation=None, sigma=0.1):
        """
        Initialisation
        :param n: input shape
        :param m: output shape
        :param batch: batch size
        :param activation: activation function name ('sigmoid' or None)
        :param sigma: control of random initialisation scale
        """

        super().__init__(n, m, batch)
        self.activation = activation
        if activation not in ['sigmoid', None]:
            raise ValueError('Only sigmoid activation implemented')

        # a batch of inputs
        self.W = np.random.uniform(-sigma, sigma, size=(m, n))
        self.bias = np.random.uniform(-sigma, sigma, size=m)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1+np.exp(-x))


    def forward(self, input: np.ndarray):
        """
        Propagate the data forward
        :param input: np.ndarray os shape (n, batch)
        :return:
        """
        self.x = input
        self.z = self.W @ input  # m x batch
        self.z.reshape((self.m, self.batch))

        self.z += self.bias.reshape((self.m, 1))

        if self.activation == 'sigmoid':
            self.y = FullyConnected.sigmoid(self.z)
        else:
            self.y = self.z
        return self.y


    def backward(self, error, lr):
        """
        Propagate the error backward
        :param error: error vector of shape (m,  batch)
        :return: propagated error of shape (n,  batch) and gradients
        """

        if self.activation == 'sigmoid':
            dydz = self.y * (1 - self.y) # of shape (m x batch)
        elif self.activation is None:
            dydz = np.ones(shape=(self.m, self.batch), dtype=np.float32)

        derrdz = dydz * error

        dzdx = self.W  # shape (m x n)

        dzdb = np.ones_like(self.z, dtype=np.float32) # (m x batch)

        derrdx = derrdz.transpose() @ dzdx  # (batch x m) times ( m x n ) = (batch x n)
        derrdx = derrdx.transpose()

        # d(z)/d(W) shape (n x m x batch)
        grad = np.broadcast_to(self.x.reshape(self.n, 1, self.batch), (self.n, self.m, self.batch))

        gradT = np.transpose(grad, axes=[1, 0, 2]) #  shape m n batch

        dydw = gradT * derrdz.reshape((self.m, 1, self.batch)) #  shape m n batch

        dydb = derrdz * dzdb

        # update:
        self.W -= lr * dydw.mean(axis=2)
        self.bias -= lr * dydb.mean(axis=1)

        return derrdx



class MSE(NetComponent):
    def __init__(self, n,  batch):
        super().__init__(n, 1, batch)

    def compute(self, input, y_true):
        """
        Compute MSE of batch
        :param input: n x batch
        :param y_true:
        :return: array of shape (1,  batch)
        """
        self.y = ((input-y_true)**2).mean(axis=0).reshape((1, self.batch))
        self.x = input
        self.y_true = y_true
        return self.y

    def derive(self):
        """
        Compute the derivative of the error relative to the inputs
        :param error: (1, 1, batch)
        :return: (n, 1, batch)
        """
        return 2 * (self.x - self.y_true)



class Network:
    """
    A very simple feed - forward NN without regularisation
    """
    def __init__(self, batch=1, loss='mse'):
        if loss != 'mse':
            raise ValueError('only mse supported')
        self.layers = []

        self.batch = batch
        self.loss = None

    def compile(self):
        """
        Call compile after adding all the FC layers
        :return:
        """
        self.loss = MSE(self.layers[-1].m, self.batch)

    def add_fc(self, n, m, activation=None):
        """
        Add a layer on top of the network
        :param n:
        :param m:
        :return:
        """
        if not self.layers:
            self.layers.append(FullyConnected(n, m, self.batch, activation))
        else:
            # TODO: check sizes
            self.layers.append(FullyConnected(n, m, self.batch, activation))

    def predict(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def train_batch(self, input, y_true, lr):
        """
        On a single batch
        :param input: (n, 1, batch)
        :param output: (m, 1, batch)
        :return:
        """
        for layer in self.layers:
            input = layer.forward(input)


        loss = self.loss.compute(input, y_true)

        derror = self.loss.derive()

        for layer in reversed(self.layers):
            derror = layer.backward(derror, lr)

        return loss



# Let's learn the logical and function
x_train = np.array([[1, 1], [0, 1], [1, 0], [0, 0]], dtype=np.float32).reshape((2, 4))
y_train = np.array([1, 0, 0, 0], dtype=np.float32).reshape(1, 4)

lr = 0.01

batch_size = 4
net = Network(batch_size)
net.add_fc(2, 20, activation='sigmoid')
net.add_fc(20, 1)

net.compile()

# now train
for i in range(1000):
    error = net.train_batch(x_train, y_train, lr)

    print(f"batch {i} mse {error.mean()}")



# first = np.arange(-1, 1, 0.01)
# second = 1 * np.ones_like(first, dtype=np.float32)
# n_in = len(first)
# test_in = np.stack([first, second], axis=1).transpose()
#
# arr_in = np.split(test_in, np.arange(4, n_in, batch_size), axis=1)
#
# # res = net.predict(arr_in[0])
# res = []
# for b in arr_in:
#     res.append(net.predict(b).reshape(-1))
#
# res = np.concatenate(res)


