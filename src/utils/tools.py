import os
import json
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD


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
    list_dir = ["cnn_sparse", "linear_sparse" , "nn_sparse", "cnn_sparse"]
    for dir in list_dir:
        directory = "../models/" + dir
        if not os.path.exists(directory):
            print(directory)
            os.mkdir(directory)


def create_folder_from_categories(path):
    with open(path) as json_file:
        data = json.load(json_file)
        for dir in data["categories"]:
            directory = "../../dataset/train_resized/" + str(dir["name"])
            if not os.path.exists(directory):
                print(directory)
                os.mkdir(directory)


def move_img_to_category_folder(path_json_img, path_categories):
    with open(path_json_img) as json_file1, open(path_categories) as json_file2:
        data = json.load(json_file1)
        data2 = json.load(json_file2)
        for f in data["annotations"]:
            try:
                img_name = f["image_id"] + ".jpg"
                category = f["category_id"]
                dir = next(item for item in data2["categories"] if item["id"] == category)["name"]
                os.rename("../../dataset/train_resized/" + img_name, "../../dataset/train_resized/{}/{}".format(dir, img_name))
                print(dir)
            except IOError:
                print('cannot open', dir)
