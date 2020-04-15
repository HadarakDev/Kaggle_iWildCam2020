# from src.models.nn_sparse import nn_sparse
from src.models.linear_sparse import linear
from src.utils.tools import create_dirs, create_folder_from_categories, move_img_to_category_folder
from src.utils.data import load_dataset_pool, get_number_of_img_percent, generate_dataset
import time
import tensorflow as tf
import matplotlib as plt

import numpy as np


if __name__ == '__main__':
    create_dirs()

    path_train_folder = "E:\\dataset\\train_resized"
    # Linear
    batch = 100
    number_of_classes = 216
    size_image_flatten = 30000
    size_x = 100
    size_y = 100
    epochs = 10
    learning_rate = 0.2

    train_dataset, STEPS_PER_EPOCH = generate_dataset(path=path_train_folder, x=size_x, y=size_y,
                                                      batch_size=batch)
    model = linear(train_dataset, number_of_classes, size_x, size_y, "relu", "adam", "categorical_crossentropy", epochs,
                   batch, learning_rate, STEPS_PER_EPOCH)


# create_folder_from_categories("..\\categories.json")
# move_img_to_category_folder("..\\annotations\\iwildcam2020_train_annotations.json", "..\\categories.json")