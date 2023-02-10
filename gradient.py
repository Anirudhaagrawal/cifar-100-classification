import numpy as np
from neuralnet import Neuralnetwork

def check_grad(model, x_train, y_train):

    """
        args:
            model
            x_train: Small subset of the original train dataset
            y_train: Corresponding target labels of x_train

        Prints gradient difference of values calculated via numerical approximation and backprop implementation
    """

    epsilon = 1e-5
    model.forward(x_train, y_train)
    model.backward()

    for i in range(len(model.layers)):
        for j in range(len(model.layers[i].w)):
            for k in range(len(model.layers[i].w[j])):
                model.layers[i].w[j][k] += epsilon
                l1, a1 = model.forward(x_train, y_train)
                model.layers[i].w[j][k] -= 2*epsilon
                l2, a2 = model.forward(x_train, y_train)
                model.layers[i].w[j][k] += epsilon
                numerical_grad = (l1-l2)/(2*epsilon)
                print("Gradient difference of weights at layer", i, "node", j, "weight", k, "is", abs(numerical_grad - model.layers[i].dw[j][k]))




def checkGradient(x_train,y_train,config):

    subsetSize = 10
    sample_idx = np.random.randint(0,len(x_train),subsetSize)
    x_train_sample, y_train_sample = x_train[sample_idx], y_train[sample_idx]

    model = Neuralnetwork(config)
    check_grad(model, x_train_sample, y_train_sample)