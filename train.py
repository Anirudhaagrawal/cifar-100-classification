
import copy
from neuralnet import *
import util

def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    TODO: Train your model here.
    Learns the weights (parameters) for our model
    Implements mini-batch SGD to train the model.
    Implements Early Stopping.
    Uses config to set parameters for training like learning rate, momentum, etc.

    args:
        model - an object of the NeuralNetwork class
        x_train - the train set examples
        y_train - the test set targets/labels
        x_valid - the validation set examples
        y_valid - the validation set targets/labels

    returns:
        the trained model
    """

    # Read in the esssential configs
    learning_rate = config["learning_rate"]
    momentum = config["momentum"]
    batch_size = config["batch_size"]
    num_epochs = config["epochs"]
    early_stopping = config["early_stop_epoch"]
    gradReqd = config["gradReqd"]
    early_stop = config["early_stop"]

    # Initialize the lists to store the loss and accuracy for each epoch
    trainEpochLoss = []
    trainEpochAccuracy = []
    valEpochLoss = []
    valEpochAccuracy = []

    # Initialize the best model to None
    bestModel = None

    # Initialize the best validation accuracy to 0
    bestValAccuracy = 0

    # Initialize the number of epochs since the last improvement to 0
    epochsSinceLastImprovement = 0

    # Initialize the early stop to -1
    earlyStop = -1

    # Iterate over the number of epochs
    for epoch in range(num_epochs):

        # Initialize the number of correct predictions to 0
        numCorrect = 0

        # Initialize the total loss to 0
        totalLoss = 0

        # Iterate over the minibatches
        for x_batch, y_batch in util.generate_minibatches((x_train, y_train), batch_size):

            # Calculate the output of the model
            loss, accuracy = model.forward(x_batch, y_batch)

            # Calculate the gradients
            model.backward()

            # Calculate the number of correct predictions
            numCorrect += util.calculateCorrect(model.y, y_batch)

            # Add the loss to the total loss
            totalLoss += loss

        # Calculate the training accuracy
        trainAccuracy = numCorrect/len(x_train)

        # Calculate the training loss
        trainLoss = totalLoss/len(x_train)

        # Append the training loss and accuracy to the list
        trainEpochLoss.append(trainLoss)
        trainEpochAccuracy.append(trainAccuracy)

        # Calculate the validation accuracy and loss
        valAccuracy, valLoss = modelTest(model, x_valid, y_valid)

        # Append the validation loss and accuracy to the list
        valEpochLoss.append(valLoss)
        valEpochAccuracy.append(valAccuracy)

        # If the validation accuracy is greater than the best validation accuracy
        if valAccuracy > bestValAccuracy:

            # Update the best validation accuracy
            bestValAccuracy = valAccuracy

            # Update the best model
            bestModel = copy.deepcopy(model)

            # Reset the number of epochs since the last improvement
            epochsSinceLastImprovement = 0

            # Update the early stop
            earlyStop = epoch

        # Else
        else:

            # Increment the number of epochs since the last improvement
            epochsSinceLastImprovement += 1

        # If the number of epochs since the last improvement is greater than the early stopping
        if epochsSinceLastImprovement > early_stopping and early_stop:

            # Break
            break

        # Print the epoch, training loss, training accuracy, validation loss and validation accuracy
        print("Epoch: ", epoch, "Training Loss: ", trainLoss, "Training Accuracy: ", trainAccuracy, "Validation Loss: ", valLoss, "Validation Accuracy: ", valAccuracy)

    # Print the best validation accuracy
    print("Best Validation Accuracy: ", bestValAccuracy)

    # Print the early stop
    print("Early Stop: ", earlyStop)

    # Return the best model
    return bestModel

#This is the test method
def modelTest(model, X_test, y_test):
    """
    TODO
    Calculates and returns the accuracy & loss on the test set.

    args:
        model - the trained model, an object of the NeuralNetwork class
        X_test - the test set examples
        y_test - the test set targets/labels

    returns:
        test accuracy
        test loss
    """

    # Initialize the number of correct predictions to 0
    numCorrect = 0

    # Initialize the total loss to 0
    totalLoss = 0

    # Iterate over the minibatches
    for x_batch, y_batch in util.generate_minibatches((X_test, y_test), 1):

        # Calculate the output of the model
        loss, accuracy = model.forward(x_batch, y_batch)

        # Calculate the number of correct predictions
        numCorrect += util.calculateCorrect(model.y, y_batch)

        # Add the loss to the total loss
        totalLoss += loss

    # Calculate the test accuracy
    testAccuracy = numCorrect/len(X_test)

    # Calculate the test loss
    testLoss = totalLoss/len(X_test)

    # Return the test accuracy and loss
    return testAccuracy, testLoss


