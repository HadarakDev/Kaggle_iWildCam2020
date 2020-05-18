import inspect

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import os


from tensorflow_core.python.keras.regularizers import L1L2

from src.utils.tools import get_optimizer

from src.utils.models import model_fit


def cnn_model(size_x, size_y, nb_output, activation, optimizer, loss, lr, array_layers, pooling, kernel_shape, dropout, l1, l2):
    optimizer_param = get_optimizer(optimizer, lr)
    model = tf.keras.Sequential()
    # if dropout != 0:
    #     model.add(Dropout(dropout, input_shape=(size_x, size_y, 3)))
    model.add(tf.keras.layers.Conv2D(filters=array_layers[0], kernel_size=(kernel_shape, kernel_shape), padding='same', activation=activation, input_shape=(size_x, size_y, 3)))
    if pooling == "avg_pool":
        model.add(tf.keras.layers.AveragePooling2D((2, 2), padding='same'))
    else:
        model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))

    for i in range(1, len(array_layers)):
        if dropout != 0:
            model.add(Dropout(dropout))
        if l1 != 0 and l2 != 0:
            model.add(tf.keras.layers.Conv2D(array_layers[i], (kernel_shape, kernel_shape), padding='same',  activation=activation, kernel_regularizer=L1L2(l1=l1, l2=l2)))
        elif l1 != 0:
            model.add(tf.keras.layers.Conv2D(array_layers[i], (kernel_shape, kernel_shape), padding='same', activation=activation, kernel_regularizer=L1L2(l1=l1)))
        elif l2 != 0:
            model.add(tf.keras.layers.Conv2D(array_layers[i], (kernel_shape, kernel_shape), padding='same',
                                             activation=activation, kernel_regularizer=L1L2(l2=l2)))
        else:
            model.add(tf.keras.layers.Conv2D(array_layers[i], (kernel_shape, kernel_shape), padding='same',
                                             activation=activation))
        if pooling == "avg_pool":
            model.add(tf.keras.layers.AveragePooling2D((2, 2), padding='same'))
        else:
            model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
    model.add(tf.keras.layers.Flatten())
    model.add(Dense(nb_output, activation="softmax"))
    model.compile(optimizer=optimizer_param, loss=loss, metrics=['categorical_accuracy'])
    model.summary()
    return model



def predict_cnn(model, X):
    img = X.reshape(1, 3072)
    res = np.argmax((model.predict(img)))
    return res

def cnn(train_dataset, test_dataset, nb_output, size_x, size_y, activation, optimizer, loss, epochs, lr,
       STEPS_PER_EPOCH_TRAIN, STEPS_PER_EPOCH_VALIDATION, array_layers, dropout, l1, l2, model_name, kernel_shape, pooling):
    name_folder = inspect.stack()[0][3]
    directory = "..\\models\\" + name_folder + "\\test"
    if not os.path.exists(directory):
        os.mkdir(directory)

    array_layers = [int(x) for x in array_layers]

    model = cnn_model(size_x, size_y, nb_output, activation, optimizer, loss, lr, array_layers, pooling, kernel_shape, dropout, l1, l2)
    model = model_fit(model, train_dataset, test_dataset, epochs, STEPS_PER_EPOCH_TRAIN, STEPS_PER_EPOCH_VALIDATION,
                      directory, model_name)
    return model
