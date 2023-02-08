import numpy as np
from neuralnet import Neuralnetwork

def check_grad(model, x_train, y_train):

    """
    TODO
        Checks if gradients computed numerically are within O(epsilon**2)

        args:
            model
            x_train: Small subset of the original train dataset
            y_train: Corresponding target labels of x_train

        Prints gradient difference of values calculated via numerical approximation and backprop implementation
    """
    epsilon = 1e-5
    weights = model.weights
    biases = model.biases
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            for k in range(len(weights[i][j])):
                weights[i][j][k] += epsilon
                loss1 = model.loss(x_train, y_train)
                weights[i][j][k] -= 2*epsilon
                loss2 = model.loss(x_train, y_train)
                weights[i][j][k] += epsilon
                grad = (loss1 - loss2) / (2 * epsilon)
                print("Weight gradient difference: ", grad - model.gradients["dW" + str(i+1)][j][k])

    for i in range(len(biases)):
        for j in range(len(biases[i])):
            biases[i][j] += epsilon
            loss1 = model.loss(x_train, y_train)
            biases[i][j] -= 2*epsilon
            loss2 = model.loss(x_train, y_train)
            biases[i][j] += epsilon
            grad = (loss1 - loss2) / (2 * epsilon)
            print("Bias gradient difference: ", grad - model.gradients["db" + str(i+1)][j])



def checkGradient(x_train,y_train,config):

    subsetSize = 10  #Feel free to change this
    sample_idx = np.random.randint(0,len(x_train),subsetSize)
    x_train_sample, y_train_sample = x_train[sample_idx], y_train[sample_idx]

    model = Neuralnetwork(config)
    check_grad(model, x_train_sample, y_train_sample)