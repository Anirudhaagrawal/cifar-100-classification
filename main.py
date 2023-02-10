################################################################################
# CSE 251B: Programming Assignment 2
# Winter 2023
# Code by Chaitanya Animesh
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import gradient
from constants import *
from train import *
from gradient import *
import argparse


def main(args):
    configFile = 'config_3c.yaml'  # Will contain the name of the config file to be loaded
    if (args.experiment == 'test_gradients'):
        configFile = 'config_3c.yaml'
    elif (args.experiment == 'test_learning_rate'):
        configFile = "config_3c.yaml"
    elif (args.experiment == 'test_regularization'):
        configFile = "config_3d.yaml"
    elif (args.experiment == 'test_activation_sigmoid'):
        configFile = "config_3e-i.yaml"
    elif (args.experiment == 'test_activation_relu'):
        configFile = "config_3e-ii.yaml"
    elif (args.experiment == 'test_hidden_units_half'):
        configFile = "config_3f-i.yaml"
    elif (args.experiment == 'test_hidden_units_double'):
        configFile = "config_3f-ii.yaml"
    elif (args.experiment == 'test_extra_layer'):
        configFile = "config_3f-iii.yaml"
    elif (args.experiment == 'test_100_classes'):
        configFile = "config_3g.yaml"


    x_train, y_train, x_valid, y_valid, x_test, y_test = util.load_data(
        path=datasetDir)


    config = util.load_config(configYamlPath + configFile)

    if (args.experiment == 'test_gradients'):
        gradient.checkGradient(x_train, y_train, config)
        return 1

    model = Neuralnetwork(config)

    model = train(model, x_train, y_train, x_valid, y_valid, config)

    test_acc, test_loss = model_test(model, x_test, y_test)

    print('Test Accuracy:', test_acc, ' Test Loss:', test_loss)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='test_activation',
                        help='Specify the experiment that you want to run')
    args = parser.parse_args()
    main(args)
