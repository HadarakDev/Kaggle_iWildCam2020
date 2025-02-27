# from src.models.nn_sparse import nn_sparse
from src.models.linear import linear
from src.models.nn import nn
from src.utils.tools import create_dirs, create_folder_from_categories, move_img_to_category_folder, \
    split_cat_data_generator, calculate_confusion_matrix
from src.utils.data import load_dataset_pool, get_number_of_img_percent, generate_dataset, split_dataset, \
    generate_split_dataset, create_dataset_equal_classes
from src.models.cnn import cnn
from src.models.unet_conv2D import unet_conv2D
from src.utils.models import predict, model_fit, create_submit, use_mega_detector_on_submit
import time
import tensorflow as tf
import matplotlib as plt
import pandas as pd

import numpy as np


if __name__ == '__main__':
    create_dirs()

    ## CREATE CATEGORIES FOLDERS
    # create_folder_from_categories("..\\annotations\\categories.json", "D:\\Wildcam2020\\resized_100\\train_resized_100_folders\\")
    # move_img_to_category_folder("..\\annotations\\iwildcam2020_train_annotations.json", "..\\annotations\\categories.json", "D:\\Wildcam2020\\resized_100\\train_resized_100_folders\\")
    # tf.compat.v1.disable_eager_execution()
    #create_dataset_equal_classes(1000, "..\\..\\dataset\\resized_32\\train_resized_32_folders_aug_backup", "..\\..\\dataset\\resized_32\\train_resized_32_folders_aug_equal" )
    #split_dataset("..\\..\\dataset\\resized_32\\train_resized_32_folders_aug_equal", "..\\..\\dataset\\resized_32\\val_resized_32_folders_aug_equal", .2)
    #

    path_train_folder = "E:\\dataset\\train_resized"
    path_val_folder ="E:\\dataset\\validation_resized"
    path_test_folder = "E:\\dataset\\test_resized"

    path_train_folder_nico = "..\\..\\dataset\\resized_32\\aug_equal\\train_resized_32_folders_aug_equal_1000"
    path_val_folder_nico = "..\\..\\dataset\\resized_32\\aug_equal\\train_resized_32_folders_aug_equal_1000"
    path_test_folder_nico = "..\\..\\dataset\\resized_32\\test_resized_32"
    #
    # # Linear
    batch = 1000
    number_of_classes = 267  # 266 withotu empty
    size_x = 32
    size_y = 32
    epochs = 100
    learning_rate = 0.0001
    activation = "selu"
    layers = [256,256,256,256]
    optimizer = "adam"
    loss = "categorical_crossentropy"
    pooling = "avg_pool"
    kernel_shape = 2
    name_linear = "lin_" + str(batch) + "_" + str(epochs) + "_" + activation + "_" + optimizer + "_" + str(learning_rate) + "_aug_without_empty"

    name_nn = "nn_" + str(batch) + "_" + str(epochs) + "_" + activation + "_" + optimizer + "_" + str(layers) + "_" + str(learning_rate) + "_aug_without_empty"
    name_cnn = "cnn_" + str(batch) + "_" + str(epochs) + "_" + activation + "_" + optimizer + "_" + str(layers) + "_" + str(learning_rate) + "_aug_without_empty"
    name_unet_cnn = "unet_cnn_" + str(batch) + "_" + str(epochs) + "_" + activation + "_" + optimizer + "_" + str(layers) + "_" + str(learning_rate)


    #
    train_dataset, STEPS_PER_EPOCH_TRAIN, class_indices, _ = generate_dataset(path=path_train_folder_nico, x=size_x, y=size_y, batch_size=batch, shuffle_data=True)
    validation_dataset, STEPS_PER_EPOCH_VALIDATION, _, classes = generate_dataset(path=path_val_folder_nico, x=size_x, y=size_y, batch_size=batch)
    test_dataset, STEPS_PER_EPOCH_TEST, _, _ = generate_dataset(path=path_test_folder_nico, x=size_x, y=size_y, batch_size=batch)


    #calculate_confusion_matrix(validation_dataset, "..\\models\\nn\\test\\nn_4096_0.001_20_selu_adam_[256,256,256,256]_aug_without_empty.h5", STEPS_PER_EPOCH_VALIDATION, classes)

    # #
    # # ## LINEAR
    # linear_model, history = linear(train_dataset, validation_dataset, number_of_classes, size_x, size_y, activation, optimizer, loss, epochs,
    #                       learning_rate, STEPS_PER_EPOCH_TRAIN, STEPS_PER_EPOCH_VALIDATION, name_linear)
    # pd.DataFrame.from_dict(history.history).to_csv('../results/' + name_linear + ".csv", index=False)

    # # NN
    # nn_model, history = nn(train_dataset, validation_dataset, number_of_classes, size_x, size_y, activation, optimizer, "categorical_crossentropy", epochs,
    #                learning_rate, STEPS_PER_EPOCH_TRAIN, STEPS_PER_EPOCH_VALIDATION, layers,
    #                0, 0, 0, name)
    # pd.DataFrame.from_dict(history.history).to_csv('../results/' + name_nn + ".csv", index=False)
    # CNN
    # cnn, history = cnn(train_dataset, validation_dataset, number_of_classes, size_x, size_y, activation, optimizer, loss, epochs,
    #            learning_rate, STEPS_PER_EPOCH_TRAIN, STEPS_PER_EPOCH_VALIDATION, layers, 0, 0, 0, name_cnn, kernel_shape, pooling)

    # Unet CNN
    unet_cnn, history = unet_conv2D(train_dataset, validation_dataset, number_of_classes, size_x, size_y, activation, optimizer,
                           loss, epochs, learning_rate, STEPS_PER_EPOCH_TRAIN, STEPS_PER_EPOCH_VALIDATION, layers, 0, 0,
                           0, name_unet_cnn, kernel_shape)
    pd.DataFrame.from_dict(history.history).to_csv('../results/' + name_unet_cnn + ".csv", index=False)

    ## RETRAIN
    # model = tf.keras.models.load_model( "C:\\Users\\nico_\\Documents\\Kaggle_iWildCam2020_Main\\Kaggle_iWildCam2020\\models\\nn\\test\\" + name + ".h5")
    # model, history = model_fit(model, train_dataset, validation_dataset, 10, STEPS_PER_EPOCH_TRAIN, STEPS_PER_EPOCH_VALIDATION, "..\\models\\nn\\test", name + "_retrain")
    # pd.DataFrame.from_dict(history.history).to_csv('../results/' + name + "_retrain.csv", index=False)
    # pd.DataFrame.from_dict(history.history).to_csv(
    #     '../results/nn_4096_0.0001_10_selu_adam_[128,128,128,128]_aug_without_empty_retrain.csv', index=False)

    ## SUBMIT
    #model = tf.keras.models.load_model(
      #   "C:\\Users\\nico_\\Documents\\Kaggle_iWildCam2020_Main\\Kaggle_iWildCam2020\\models\\nn\\test\\nn_4096_0.0001_10_selu_adam_[64,64,64,64].h5")
    # result = predict(model, test_dataset, STEPS_PER_EPOCH_TEST, class_indices, path_test_folder_nico)
    # result.to_csv("C:\\Users\\nico_\\Documents\Kaggle_iWildCam2020_Main\\Kaggle_iWildCam2020\\submits\\submissionn_nn_4096_0.0001_10_selu_adam_[64,64,64,64].csv", index=False)
    #create_submit("..\\models\\cnn\\test\\cnn_1000_100_selu_adam_[256, 256, 256, 256]_0.0001_aug.h5", test_dataset, STEPS_PER_EPOCH_TEST, class_indices, path_test_folder_nico, "..\\submits\\submission_cnn_256_aug.csv")
    # use_mega_detector_on_submit('..\\annotations\\bbox_test_full.json',"..\\submits\\submission_cnn_256.csv", "..\\submits\\submission_cnn_256_empty.csv")



    # split_cat_data_generator("..\\annotations\\iwildcam2020_train_annotations.json", "..\\annotations\\categories.json")