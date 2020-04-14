import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import numpy as np
import os

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


def linear_model(size, nb_output, activation, optimizer, loss, lr):
    optimizer_param = get_optimizer(optimizer, lr)
    model = tf.keras.Sequential()
    model.add(Dense(1, activation=activation, input_shape=(100, 100, 3)))
    model.add(Dense(nb_output, activation="softmax", input_dim=1))
    model.compile(optimizer=optimizer_param, loss=loss, metrics=['categorical_accuracy'])
    return model


def predict_linear(model, X):
    img = X.reshape(1, 3072)
    res = np.argmax((model.predict(img)))
    return res


def linear(train_data_generator, nb_output, image_size, activation, optimizer, loss, epochs, batch_size, lr, STEPS_PER_EPOCH):
    # directory = base_path + save_dir
    # if not os.path.exists(directory):
    #     os.mkdir(directory)
    # path = directory + "/model.h5"\
    model = linear_model(image_size, nb_output,
                         activation, optimizer, loss, lr)

    model = model_fit(model, train_data_generator, epochs, batch_size, STEPS_PER_EPOCH)

    return model
