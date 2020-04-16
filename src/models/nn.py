import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten
import numpy as np
import os
import inspect

from tensorflow_core.python.keras.regularizers import L1L2

from src.utils.tools import get_optimizer
from src.utils.models import model_fit


def nn_model(size_x, size_y, nb_output, activation, optimizer, loss, lr, array_layers, dropout, l1, l2):
    optimizer_param = get_optimizer(optimizer, lr)
    model = tf.keras.Sequential()
    model.add(Flatten(batch_input_shape=(None, size_x, size_y, 3)))
    model.add(Dense(array_layers[0], activation=activation))

    for i in range(1, len(array_layers)):
        if dropout != 0:
            model.add(Dropout(dropout))
        if l1 != 0 and l2 != 0:
            model.add(Dense(array_layers[i], activation=activation, kernel_regularizer=L1L2(l1=l1, l2=l2)))
        elif l1 != 0:
            model.add(Dense(array_layers[i], activation=activation, kernel_regularizer=L1L2(l1=l1)))
        elif l2 != 0:
            model.add(Dense(array_layers[i], activation=activation, kernel_regularizer=L1L2(l2=l2)))
        else:
            model.add(Dense(array_layers[i], activation=activation))

    model.add(Dense(nb_output, activation="softmax"))
    model.compile(optimizer=optimizer_param, loss=loss, metrics=['categorical_accuracy'])
    model.summary()
    return model


def predict_nn(model, X, nb_pixels):
    img = X.reshape(1, nb_pixels)
    res = np.argmax((model.predict(img)))
    return res


def nn(train_dataset, test_dataset, nb_output, size_x, size_y, activation, optimizer, loss, epochs, lr,
       STEPS_PER_EPOCH_TRAIN, STEPS_PER_EPOCH_VALIDATION, array_layers, dropout, l1, l2):
    name_folder = inspect.stack()[0][3]
    directory = "..\\models\\" + name_folder + "\\test"
    if not os.path.exists(directory):
        os.mkdir(directory)

    model = nn_model(size_x, size_y, nb_output, activation, optimizer, loss, lr, array_layers, dropout, l1, l2)

    model = model_fit(model, train_dataset, test_dataset, epochs, STEPS_PER_EPOCH_TRAIN, STEPS_PER_EPOCH_VALIDATION,
                      directory)

    return model
