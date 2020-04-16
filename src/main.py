# from src.models.nn_sparse import nn_sparse
from src.models.linear import linear
from src.models.nn import nn
from src.utils.tools import create_dirs, create_folder_from_categories, move_img_to_category_folder
from src.utils.data import load_dataset_pool, get_number_of_img_percent, generate_dataset, split_dataset
from src.utils.models import  predict
import time
import tensorflow as tf
import matplotlib as plt
import pandas as pd

import numpy as np


if __name__ == '__main__':
    create_dirs()

    #split_dataset("C:\\Users\\nico_\\Documents\\Kaggle_iWildCam2020_Main\\dataset\\train_resized", "C:\\Users\\nico_\\Documents\\Kaggle_iWildCam2020_Main\\dataset\\val", .2)

    path_train_folder = "E:\\dataset\\train_resized"
    path_val_folder ="E:\\dataset\\validation_resized"
    path_test_folder = "E:\\dataset\\test_resized"
    # # Linear
    batch = 10000
    number_of_classes = 267
    size_x = 100
    size_y = 100
    epochs = 5
    learning_rate = 0.001

    train_dataset, STEPS_PER_EPOCH_TRAIN, class_indices = generate_dataset(path=path_train_folder, x=size_x, y=size_y, batch_size=batch, shuffle_data=True)
    # validation_dataset, STEPS_PER_EPOCH_VALIDATION, _ = generate_dataset(path=path_val_folder, x=size_x, y=size_y, batch_size=batch)
    # linear_model = linear(train_dataset, validation_dataset, number_of_classes, size_x, size_y, "selu", "adam", "categorical_crossentropy", epochs,
    #                       learning_rate, STEPS_PER_EPOCH_TRAIN, STEPS_PER_EPOCH_VALIDATION)
    # nn_model = nn(train_dataset, validation_dataset, number_of_classes, size_x, size_y, "selu", "adamax", "categorical_crossentropy", epochs,
    #               learning_rate, STEPS_PER_EPOCH_TRAIN, STEPS_PER_EPOCH_VALIDATION, [255, 255, 255, 255, 255, 255, 255, 255],
    #               0.2, 0.1, 0.1)

    test_dataset, STEPS_PER_EPOCH_TEST, _ = generate_dataset(path=path_test_folder, x=size_x, y=size_y, batch_size=batch)

    model = tf.keras.models.load_model("E:\\Kaggle_iWildCam2020\\models\\nn\\test\\model.h5")
    result = predict(model, test_dataset, STEPS_PER_EPOCH_TEST, class_indices, path_test_folder)
    print(result)
    result.to_csv("E:\\submission.csv", index=False)


# create_folder_from_categories("..\\categories.json")
# move_img_to_category_folder("..\\annotations\\iwildcam2020_train_annotations.json", "..\\categories.json")