# from src.models.nn_sparse import nn_sparse
from src.models.linear import linear
from src.models.nn import nn
from src.utils.tools import create_dirs, create_folder_from_categories, move_img_to_category_folder, split_cat_data_generator
from src.utils.data import load_dataset_pool, get_number_of_img_percent, generate_dataset, split_dataset, generate_split_dataset
from src.utils.models import predict, model_fit, create_submit, use_mega_detector_on_submit
import time
import tensorflow as tf
import matplotlib as plt
import pandas as pd

import numpy as np


if __name__ == '__main__':
    create_dirs()

    #split_dataset("C:\\Users\\nico_\\Documents\\Kaggle_iWildCam2020_Main\\dataset\\train_resized_without_empty", "C:\\Users\\nico_\\Documents\\Kaggle_iWildCam2020_Main\\dataset\\val_resized_without_empty", .2)

    path_train_folder = "E:\\dataset\\train_resized"
    path_val_folder ="E:\\dataset\\validation_resized"
    path_test_folder = "E:\\dataset\\test_resized"

    path_train_folder_nico = "..\\..\\dataset\\train_resized_without_empty"
    path_val_folder_nico ="..\\..\\dataset\\val_resized_without_empty"
    path_test_folder_nico = "..\\..\\dataset\\test_resized"

    path_train_folder_generate = "..\\..\\dataset\\train_resized_data_gen"

    # # Linear
    batch = 10000
    number_of_classes = 266
    size_x = 100
    size_y = 100
    epochs = 50
    learning_rate = 0.0001

    train_dataset, STEPS_PER_EPOCH_TRAIN, class_indices = generate_split_dataset(path=path_train_folder_generate, x=size_x, y=size_y, batch_size=batch, shuffle_data=True)
    validation_dataset, STEPS_PER_EPOCH_VALIDATION, _ = generate_dataset(path=path_val_folder_nico, x=size_x, y=size_y, batch_size=batch)
    test_dataset, STEPS_PER_EPOCH_TEST, _ = generate_dataset(path=path_test_folder_nico, x=size_x, y=size_y, batch_size=batch)
    #linear_model = linear(train_dataset, validation_dataset, number_of_classes, size_x, size_y, "selu", "adam", "categorical_crossentropy", epochs,
    #                       learning_rate, STEPS_PER_EPOCH_TRAIN, STEPS_PER_EPOCH_VALIDATION)
    nn_model = nn(train_dataset, validation_dataset, number_of_classes, size_x, size_y, "elu", "adam", "categorical_crossentropy", epochs,
                   learning_rate, STEPS_PER_EPOCH_TRAIN, STEPS_PER_EPOCH_VALIDATION, [512, 512,512,512,512,512,512],
                   0, 0, 0, "nn_512_8")

    ## RETRAIN
    #model_fit(model, train_dataset, validation_dataset, 30, STEPS_PER_EPOCH_TRAIN, STEPS_PER_EPOCH_VALIDATION, "..\\models\\nn\\test", "nn_255_4_retrain")

    ## SUBMIT
    # result = predict(model, test_dataset, STEPS_PER_EPOCH_TEST, class_indices, path_test_folder_nico)
    # result.to_csv("C:\\Users\\nico_\\Documents\Kaggle_iWildCam2020_Main\\Kaggle_iWildCam2020\\submission_nn_255_44.csv", index=False)
    # create_submit("..\\models\\nn\\test\\nn_255_4_retrain.h5", test_dataset, STEPS_PER_EPOCH_TEST, class_indices, path_test_folder, "..\\submits\\submission_nn_255_44.csv")
    # use_mega_detector_on_submit('..\\annotations\\bbox_test_full.json', "..\\submits\\submission_nn_255_44.csv", "..\\submits\\submission_nn_255_44.csv")

    ## REVERT SPLIT
    # create_folder_from_categories("..\\categories.json")
    # move_img_to_category_folder("..\\annotations\\iwildcam2020_train_annotations.json", "..\\categories.json")

    # split_cat_data_generator("..\\annotations\\iwildcam2020_train_annotations.json", "..\\annotations\\categories.json")