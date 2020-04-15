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


def linear_model(size, nb_output, activation, optimizer, loss, lr, batch_size):
    # optimizer_param = get_optimizer(optimizer, lr)
    #
    # inputs = tf.keras.Input(shape=(30000,))
    # x = tf.keras.layers.Dense(1, activation=activation, name='linear')(inputs)
    # outputs = tf.keras.layers.Dense(10, activation="softmax", name='predictions')(x)
    # model = tf.keras.Model(inputs=inputs, outputs=outputs)
    #
    # model.compile(optimizer=optimizer_param,
    #               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    #               metrics=['categorical_accuracy'])
    # return model

    optimizer_param = get_optimizer(optimizer, lr)
    model = tf.keras.Sequential()
    model.add(Dense(1, activation=activation, batch_input_shape=[None, 100, 100, 3]))
    model.add(Dense(nb_output))
    model.compile(optimizer=optimizer_param, loss=loss, metrics=['categorical_accuracy'])
    return model


def predict_linear(model, X):
    img = X.reshape(1, 3072)
    res = np.argmax((model.predict(img)))
    return res


def linear(train_dataset, nb_output, image_size, activation, optimizer, loss, epochs, batch_size, lr, STEPS_PER_EPOCH):
    # directory = base_path + save_dir
    # if not os.path.exists(directory):
    #     os.mkdir(directory)
    # path = directory + "/model.h5"\
    model = linear_model(image_size, nb_output,
                         activation, optimizer, loss, lr, batch_size)

    model = model_fit(model, train_dataset, epochs, batch_size, STEPS_PER_EPOCH)

    return model
