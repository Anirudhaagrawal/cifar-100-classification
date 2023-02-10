import numpy as np


class Activation():

    def __init__(self, activation_type="sigmoid"):
        if activation_type not in ["sigmoid", "tanh", "ReLU",
                                   "output"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        self.activation_type = activation_type

        self.x = None

    def __call__(self, z):
        return self.forward(z)

    def forward(self, z):
        if self.activation_type == "sigmoid":
            return self.sigmoid(z)

        elif self.activation_type == "tanh":
            return self.tanh(z)

        elif self.activation_type == "ReLU":
            return self.ReLU(z)

        elif self.activation_type == "output":
            return self.output(z)

    def backward(self, z):
        if self.activation_type == "sigmoid":
            return self.grad_sigmoid(z)

        elif self.activation_type == "tanh":
            return self.grad_tanh(z)

        elif self.activation_type == "ReLU":
            return self.grad_ReLU(z)

        elif self.activation_type == "output":
            return self.grad_output(z)

    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))

    def tanh(self, x):
        return np.tanh(x)

    def ReLU(self, x):
        return np.maximum(0, x)

    def output(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def grad_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def grad_tanh(self, x):
        return 1 - np.square(self.tanh(x))

    def grad_ReLU(self, x):
        return np.where(x > 0, 1, 0)

    def grad_output(self, x):
        return 1


class Layer():
    def __init__(self, in_units, out_units, activation, weight_type):
        np.random.seed(42)
        self.w = None
        if weight_type == 'random':
            self.w = np.random.normal(0, 0.01, (in_units + 1, out_units))
        self.x = None
        self.a = None
        self.z = None
        self.activation = activation
        self.dw = 0

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):

        self.x = x
        self.x = np.append(self.x, np.ones((self.x.shape[0], 1)), axis=1)
        self.a = np.dot(self.x, self.w)
        self.z = self.activation(self.a)
        return self.z

    def backward(self, delta_cur, learning_rate, momentum_gamma, regularization, penalty=0.0, grad_reqd=True):
        delta_cur = delta_cur * self.activation.backward(self.a)
        if grad_reqd:
            if regularization == 'l2':
                self.dw = learning_rate * np.dot(self.x.T, delta_cur) + penalty * self.w + momentum_gamma * self.dw
            else:
                self.dw = learning_rate * np.dot(self.x.T, delta_cur) + momentum_gamma * self.dw
            self.w = self.w - self.dw
        return np.dot(delta_cur, self.w[:-1, :].T)


class Neuralnetwork():
    def __init__(self, config):
        self.layers = []
        self.num_layers = len(config['layer_specs']) - 1
        self.x = None
        self.y = None
        self.targets = None
        self.loss = None
        self.loss_grad = None
        self.config = config
        self.learning_rate = config['learning_rate']
        self.momentum_gamma = config['momentum_gamma']
        self.regularization = config['regularization']
        self.regularization_penalty = config['regularization_penalty']

        # Add layers specified by layer_specs.
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                self.layers.append(
                    Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation(config['activation']),
                          config["weight_type"]))
            elif i == self.num_layers - 1:
                self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation("output"),
                                         config["weight_type"]))

    def __call__(self, x, targets=None):
        return self.forward(x, targets)

    def accuracy(self, logits, targets):
        return np.sum(np.argmax(logits, axis=1) == np.argmax(targets, axis=1)) / targets.shape[0]

    def calculate_loss(self, logits, targets):
        return -np.sum(targets * np.log(logits)) / targets.shape[0]

    def forward(self, x, targets=None):
        self.x = x
        self.targets = targets
        for i in range(self.num_layers):
            self.x = self.layers[i].forward(self.x)
        self.y = self.x
        if targets is not None:
            return self.calculate_loss(self.y, targets), self.accuracy(self.y, targets)
        else:
            return self.calculate_loss(self.y, targets)

    def backward(self, grad_reqd=True):
        delta = self.y - self.targets
        for i in range(self.num_layers - 1, -1, -1):
            delta = self.layers[i].backward(delta, self.learning_rate, self.momentum_gamma, self.regularization,
                                            self.regularization_penalty, grad_reqd=grad_reqd, )
