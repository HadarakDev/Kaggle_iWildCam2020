import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
import os
import inspect

from src.utils.tools import get_optimizer
from src.utils.models import model_fit



"""This function is create a sparse Linear model
- **parameters**, **types**, **return** and **return types**::
     :param size: number of input
     :param nb_ouptut: number of output ( classes )
     :param activation: activation function
     :param optimizer: optimizer used to train model
     :param loss: loss function to optimize
     :param lr: learning rate
"""


def linear_model(size_x, size_y, nb_output, activation, optimizer, loss, lr):
    optimizer_param = get_optimizer(optimizer, lr)
    model = tf.keras.Sequential()
    model.add(Flatten(batch_input_shape=(None, size_x, size_y, 3)))
    model.add(Dense(1, activation=activation))
    model.add(Dense(nb_output))
    model.compile(optimizer=optimizer_param, loss=loss, metrics=['categorical_accuracy'])
    return model





def linear(train_dataset, validation_dataset, nb_output, size_x, size_y, activation, optimizer, loss, epochs, lr, STEPS_PER_EPOCH_TRAIN, STEPS_PER_EPOCH_VALIDATION):
    name_folder = inspect.stack()[0][3]
    directory = "..\\models\\" + name_folder + "\\test2"
    if not os.path.exists(directory):
        os.mkdir(directory)

    model = linear_model(size_x, size_y, nb_output,
                         activation, optimizer, loss, lr)

    model = model_fit(model, train_dataset, validation_dataset, epochs, STEPS_PER_EPOCH_TRAIN, STEPS_PER_EPOCH_VALIDATION, directory)

    return model
