# Cifar 100 Classification
This is a simple example of how to use the [Cifar 100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset to train a model from scratch using only numpy.

## Requirements
One can install the requirements using the following command:
```
pip install -r requirements.txt
```

## Usage

To test gradients, run the following command:
```
python main.py --experiment test_gradients
```

To train a model from scratch, run the following command:
This will train a model without regularization with config in file config_3c.yaml

```
python main.py --experiment test_learning_rate
```

After training the model generates graphs for validation and training loss and accuracy these are in the folder named plots along with the graphs
for the current experiment. It also reports the test accuracy and test loss and the early stopping epoch.


To train a model with regularization, run the following command:
``` 
python main.py --experiment test_regularization
```

To train a model with sigmoid activation function, run the following command:
```
python main.py --experiment test_activation_sigmoid
```

To train a model with Relu activation function, run the following command:
```
python main.py --experiment test_activation_relu
```

To train a model with half the hidden units, run the following command:
```
python main.py --experiment test_hidden_units_half
```

To train a model with double the hidden units, run the following command:
```
python main.py --experiment test_hidden_units_double
```

To train a model with extra hidden layer, run the following command:
```
python main.py --experiment test_extra_layer
```

To train a model with 100 classes, run the following command:
```
python main.py --experiment test_100_classes
```


## Results
All the results are saved in the folder results with config file name as the folder of the directory. It contains the following files:
- config.yaml: the config file used for the experiment
- .csv with train_loss, val_loss, train_accuracy and the val_accuracy
- .png with the train_loss, val_loss, train_accuracy and the val_accuracy
- test_stats.txt with the test accuracy and the test loss and early stopping epoch