import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import os

from tensorflow_core.python.keras.layers import Flatten
from tensorflow_core.python.keras.regularizers import L1L2

from src.utils.tools import get_optimizer
from src.utils.models import model_fit


def nn_model(image_size, nb_output, activation, optimizer, loss, lr, array_layers, dropout, l1, l2):
    optimizer_param = get_optimizer(optimizer, lr)
    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=(image_size,)))
    model.add(Dense(array_layers[0], activation=activation, input_dim=image_size))

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
    model.compile(optimizer=optimizer_param, loss=loss, metrics=['sparse_categorical_accuracy'])
    model.summary()
    return model


def predict_nn(model, X, nb_pixels):
    img = X.reshape(1, nb_pixels)
    res = np.argmax((model.predict(img)))
    return res


def nn_sparse(X_all, Y, image_size, activation, optimizer, loss, epochs, batch_size, lr, save_dir, base_path,
              array_layers, dropout, l1, l2):

    nb_output = np.max(Y) + 1
    directory = base_path + save_dir
    if not os.path.exists(directory):
        os.mkdir(directory)
    path = directory + "/model.h5"
    model = nn_model(image_size, nb_output, activation, optimizer, loss, lr, array_layers, dropout, l1, l2)

    print("fitting")
    model = model_fit(model, X_all, Y,
                      epochs, batch_size,
                      path, save_dir, base_path)