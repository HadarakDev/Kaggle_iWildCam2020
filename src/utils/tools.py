import os
import json
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD
from shutil import copyfile
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def calculate_confusion_matrix(validation_generator, model_path, STEPS_PER_EPOCH, class_indices):
    model = tf.keras.models.load_model(model_path)
    Y_pred = model.predict_generator(validation_generator, verbose=1, steps=STEPS_PER_EPOCH)
    y_pred = np.argmax(Y_pred, axis=1)
    print(y_pred)
    cm = confusion_matrix(class_indices, y_pred)
    # fig = plt.figure()
    # plt.matshow(cm)
    # plt.title('Problem 1: Confusion Matrix Digit Recognition')
    # plt.colorbar()
    # plt.ylabel('True Label')
    # plt.xlabel('Predicated Label')
    # plt.savefig('confusion_matrix.jpg')

def get_optimizer(optimizer_param, lr_param):
    if optimizer_param == "adadelta":
        optimizer_param = Adadelta(lr=lr_param)
    if optimizer_param == "adagrad":
        optimizer_param = Adagrad(lr=lr_param)
    if optimizer_param == "adam":
        optimizer_param = Adam(lr=lr_param)
    if optimizer_param == "adamax":
        optimizer_param = Adamax(lr=lr_param)
    if optimizer_param == "ftrl":
        optimizer_param = Ftrl(lr=lr_param)
    if optimizer_param == "nadam":
        optimizer_param = Nadam(lr=lr_param)
    if optimizer_param == "rmsprop":
        optimizer_param = RMSprop(lr=lr_param)
    if optimizer_param == "sgd":
        optimizer_param = SGD(lr=lr_param)
    return optimizer_param


def create_dirs():
    list_dir = ["cnn", "linear", "nn", "cnn", "unet_conv2D"]
    if not os.path.exists("../models/"):
        os.mkdir("../models/")
    for dir in list_dir:
        directory = "../models/" + dir
        if not os.path.exists(directory):
            print(directory)
            os.mkdir(directory)


def create_folder_from_categories(path_json_categories, dest_path_folder):
    with open(path_json_categories) as json_file:
        data = json.load(json_file)
        for dir in data["categories"]:
            directory = dest_path_folder + str(dir["name"])
            print(directory)
            if not os.path.exists(directory):
                os.mkdir(directory)


def move_img_to_category_folder(path_json_img, path_categories, dest_path_folder):
    with open(path_json_img) as json_file1, open(path_categories) as json_file2:
        data = json.load(json_file1)
        data2 = json.load(json_file2)
        for f in data["annotations"]:
            try:
                img_name = f["image_id"] + ".jpg"
                category = f["category_id"]
                dir = next(item for item in data2["categories"] if item["id"] == category)["name"]
                os.rename(dest_path_folder + img_name, dest_path_folder + "{}/{}".format(dir, img_name))
            except IOError:
                print('cannot open', dir)


def split_cat_data_generator(path_json_img, path_categories):
    with open(path_json_img) as json_file1, open(path_categories) as json_file2:
        data = json.load(json_file1)
        data2 = json.load(json_file2)
        dirs = [item["name"] for item in data2["categories"]]
        dirs = set(dirs)
        for empty_dir in dirs:
            for empty_dir2 in dirs:
                Path("../../dataset/train_resized_data_gen/{}/{}".format(empty_dir, empty_dir2)).mkdir(parents=True, exist_ok=True)
        for f in data["annotations"]:
            try:
                img_name = f["image_id"] + ".jpg"
                category = f["category_id"]
                dir = next(item for item in data2["categories"] if item["id"] == category)["name"]
                copyfile("../../dataset/train_resized/{}/{}".format(dir, img_name),  "../../dataset/train_resized_data_gen/{}/{}/{}".format(dir, dir, img_name))
            except IOError:
                print('cannot open', dir)

