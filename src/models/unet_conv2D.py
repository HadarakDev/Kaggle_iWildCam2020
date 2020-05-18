import os
import inspect

from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.metrics import *
from tensorflow.keras.utils import *
from tensorflow.keras.models import *
from tensorflow_core.python.keras.regularizers import L1L2

from src.utils.tools import get_optimizer
from src.utils.models import model_fit

def create_model(size_x, size_y, nb_output, activation, optimizer, loss, lr, array_layers, kernel_shape, dropout, l1, l2):
    depth = int(len(array_layers) / 2)
    if depth % 2 == 0:
        depth = depth - 1

    optimizer = get_optimizer(optimizer, lr)
    stack = []
    input_layer = Input((size_x, size_y, 3))
    # Edit padding depending of size of image. For cifar 10, do not zeropad
    padding_layer = ZeroPadding2D((2, 2))(input_layer)

    last_output = Conv2D(filters=array_layers[0],
                         kernel_size=(kernel_shape, kernel_shape),
                         activation=activation,
                         name=f"Conv2D",
                         padding='SAME',  kernel_regularizer=L1L2(l1=l1, l2=l2))(padding_layer)
    stack.append(last_output)
    last_output = MaxPooling2D((2, 2), padding='SAME', name=f"MaxPooling2D")(last_output)

    for i in range((depth * 2)):
        last_output = Dropout(dropout, name=f"Dropout_{i}")(last_output)
        last_output = Conv2D(filters=array_layers[i],
                             kernel_size=(kernel_shape, kernel_shape),
                             activation=activation,
                             name=f"Conv2D_{i}",
                             padding='SAME', kernel_regularizer=L1L2(l1=l1, l2=l2))(last_output)
        last_output = BatchNormalization(name=f"BatchNormalization_{i}")(last_output)
        last_output = Activation(activation=relu, name=f"Activation_{i}")(last_output)
        if i < depth:
            stack.append(last_output)
            last_output = MaxPooling2D((2, 2), padding='SAME', name=f"MaxPooling2D_{i}")(last_output)
        else:
            last_output = UpSampling2D((2, 2), name=f"UpSampling2D_{i}")(last_output)
            last_output = Add(name=f"Add_{i}")([last_output, stack.pop()])

    last_output = Conv2D(filters=array_layers[len(array_layers) - 1],
                         kernel_size=(kernel_shape, kernel_shape),
                         activation=activation,
                         name=f"Conv2D_last",
                         padding='SAME',  kernel_regularizer=L1L2(l1=l1, l2=l2))(last_output)
    last_output = UpSampling2D((2, 2), name=f"UpSampling2D_last")(last_output)
    last_output = Add(name=f"Add_last")([last_output, stack.pop()])

    last_output = Flatten(name="flatten")(last_output)
    # kernel_regularizer_param = get_kernel_regularizer()... kernel_regularizer=L1L2(l2=0.001 / depth)
    output_tensor = Dense(nb_output, activation=softmax, name=f"Dense_output")(last_output)
    model = Model(input_layer, output_tensor)

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=[categorical_accuracy])
    return model


def unet_conv2D(train_dataset, test_dataset, nb_output, size_x, size_y, activation, optimizer, loss, epochs, lr,
       STEPS_PER_EPOCH_TRAIN, STEPS_PER_EPOCH_VALIDATION, array_layers, dropout, l1, l2, model_name, kernel_shape):
    name_folder = inspect.stack()[0][3]
    array_layers = [int(x) for x in array_layers]

    directory = "..\\models\\" + name_folder + "\\test"
    if not os.path.exists(directory):
        os.mkdir(directory)

    model = create_model(size_x, size_y, nb_output, activation, optimizer, loss, lr, array_layers, kernel_shape,
                         dropout, l1, l2)
    model = model_fit(model, train_dataset, test_dataset, epochs, STEPS_PER_EPOCH_TRAIN, STEPS_PER_EPOCH_VALIDATION,
                      directory, model_name)
    return model
