import copy
import util


def train(model, x_train, y_train, x_valid, y_valid, config):
    """
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

    batch_size = config["batch_size"]
    num_epochs = config["epochs"]
    early_stopping = config["early_stop_epoch"]
    early_stop_on = config["early_stop"]

    train_epoch_loss = []
    train_epoch_accuracy = []
    val_epoch_loss = []
    val_epoch_accuracy = []
    best_model = None
    best_val_accuracy = 0
    epochs_since_last_improvement = 0
    early_stop = -1

    for epoch in range(num_epochs):

        num_correct = 0

        total_loss = 0

        for x_batch, y_batch in util.generate_minibatches((x_train, y_train), batch_size):
            loss, accuracy = model.forward(x_batch, y_batch)

            model.backward()

            num_correct += util.calculateCorrect(model.y, y_batch)

            total_loss += loss

        train_accuracy = num_correct / len(x_train)

        train_loss = total_loss / len(x_train)

        train_epoch_loss.append(train_loss)
        train_epoch_accuracy.append(train_accuracy)

        val_accuracy, valLoss = model_test(model, x_valid, y_valid)

        val_epoch_loss.append(valLoss)
        val_epoch_accuracy.append(val_accuracy)

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_model = copy.deepcopy(model)

            epochs_since_last_improvement = 0

            early_stop = epoch

        else:

            epochs_since_last_improvement += 1

        if epochs_since_last_improvement > early_stopping and early_stop_on:
            break

        print("Epoch: ", epoch, "Training Loss: ", train_loss, "Training Accuracy: ", train_accuracy,
              "Validation Loss: ", valLoss, "Validation Accuracy: ", val_accuracy)

    print("Best Validation Accuracy: ", best_val_accuracy)

    print("Early Stop: ", early_stop)

    util.plots(train_epoch_loss, train_epoch_accuracy, val_epoch_loss, val_epoch_accuracy, early_stop)

    return best_model


def model_test(model, X_test, y_test):
    """
    Calculates and returns the accuracy & loss on the test set.

    args:
        model - the trained model, an object of the NeuralNetwork class
        X_test - the test set examples
        y_test - the test set targets/labels

    returns:
        test accuracy
        test loss
    """

    num_correct = 0

    total_loss = 0

    for x_batch, y_batch in util.generate_minibatches((X_test, y_test), 1):
        loss, accuracy = model.forward(x_batch, y_batch)

        num_correct += util.calculateCorrect(model.y, y_batch)

        total_loss += loss

    test_accuracy = num_correct / len(X_test)

    test_loss = total_loss / len(X_test)

    return test_accuracy, test_loss
