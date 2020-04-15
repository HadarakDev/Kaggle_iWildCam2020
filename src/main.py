# from src.models.nn_sparse import nn_sparse
from src.models.linear_sparse import linear
from src.utils.tools import create_dirs, create_folder_from_categories, move_img_to_category_folder
from src.utils.data import load_dataset_pool, get_number_of_img_percent, generate_dataset
import time
import tensorflow as tf
import matplotlib as plt

import numpy as np

if __name__ == '__main__':
    #X_all = load_dataset_pool("..\\..\\dataset\\train_resized\\", 10)
    #print(X_all)
    # create_folder_from_categories("..\\categories.json")
    # move_img_to_category_folder("..\\annotations\\iwildcam2020_train_annotations.json", "..\\categories.json")
    batch = 100
    number_of_classes = 2
    size_image_flatten = 30000
    train_dataset, STEPS_PER_EPOCH = generate_dataset(path="E:\\dataset\\test\\", x=100, y=100, batch_size=batch)


    model = linear(train_dataset, number_of_classes, size_image_flatten, "relu", "adam", "categorical_crossentropy", 2, batch, 0.02, STEPS_PER_EPOCH)


    # for X, Y in train_dataset:
    #     start_time = time.time()
    #     print(X)
    #     print(Y)

# Y = np.zeros(len(X_all))
#
# create_dirs()
# print(len(X_all))
# print(len(Y))
# nn_sparse(X_all, Y, 3, "selu", "adam", "categorical_crossentropy", 500, 1000, 0.0001, "./", "./", [512, 512, 512, 512], 0.2, 0, 0)






