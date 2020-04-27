import math
import os
import glob
import random
import shutil

import tensorflow as tf
import numpy as np
from PIL import Image
from multiprocessing import Pool, Array
from pathlib import Path
import matplotlib.pyplot as plt
import shutil

def move_multiple_images(images, src_dir, dest_dir):
    for img in images:
        shutil.move(src_dir + "\\" + img, dest_dir + "\\" + img)

def split_dataset(dataset_path, validation_path, percentage):
    if not os.path.isdir(validation_path):
        os.mkdir(validation_path)
    dirs = os.listdir(dataset_path)
    for dir in dirs:
        if not os.path.isdir(validation_path + "\\" + dir):
            os.mkdir(validation_path + "\\" + dir)
        images = os.listdir(dataset_path + "\\" + dir)
        nb_images = len(images)
        print(nb_images)
        img_to_move = nb_images * percentage
        print(round(img_to_move))

        selected_img = random.sample(images, round(img_to_move))
        move_multiple_images(selected_img, dataset_path + "\\" + dir, validation_path + "\\" + dir)



def get_number_of_img_percent(percent, nb_img):
    return int(percent / 100 * nb_img)


def load_dataset(path, size):
    jpg_files_list = get_all_files_path(path)
    dataset = np.ndarray(shape=(len(jpg_files_list), size), dtype=float)
    i = 0
    for file in jpg_files_list:
        try:
            im = Image.open(path + file)
            pixels = list(im.getdata())
            img = np.asarray(pixels, dtype="float32").flatten()
            dataset[i] = img / 255
            i = i + 1
            if i % 1000 == 0:
                print("Loading image " + str(i))
        except IOError:
            print('cannot open', file)
    return dataset


def open_img_thread(jpg_file):
    try:
        im = Image.open(jpg_file)
        pixels = list(im.getdata())
        img = np.asarray(pixels, dtype="float32").flatten()
        dataset = img / 255
        return dataset
    except IOError:
        print('cannot open', jpg_file)
        return None


def load_dataset_pool(path, percent):
    jpg_files_list = get_all_files_path(path)
    jpg_files_list = [path + f for f in jpg_files_list]
    # get indexes images to load
    nb_images_to_extract = get_number_of_img_percent(percent, len(jpg_files_list))

    p = Pool()
    all_params = []
    params = []
    print(len(jpg_files_list))
    for i in range(len(jpg_files_list)):
        all_params.append(jpg_files_list[i])

    print(nb_images_to_extract)
    while nb_images_to_extract > 0:
        params.append(all_params.pop(random.randrange(len(all_params))))
        nb_images_to_extract = nb_images_to_extract - 1

    print(nb_images_to_extract)
    print(params)
    print(len(params))
    result = p.map(open_img_thread, params)
    p.close()
    p.join()
    return result


def get_all_files_path(path):
    jpg_files = []

    for file_path in glob.glob(path + "*.jpg"):
        file_name = file_path.split("\\")[-1]
        jpg_files.append(file_name)
    return jpg_files


def resize_save_images(jpg_files_list, size_x, size_y, path_folder, new_path_folder):
    for file in jpg_files_list:
        try:
            im = Image.open(path_folder + file)
            out = im.resize((size_x, size_y))
            out.save(new_path_folder + file)
        except IOError:
            print('cannot open', file)

# path = "..\\..\\..\\dataset\\test\\"
# new_path = "..\\..\\..\\dataset\\test_resized\\"
# jpg_files = get_all_files_path(path)
# resize_save_images(jpg_files, 100, 100, path, new_path)


def generate_dataset(path, x, y, batch_size, shuffle_data=False):
    data_dir = Path(path)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)
    CLASS_NAMES = []
    for item in data_dir.glob('*'):
        # dir = os.listdir(item)
        # if len(dir) != 0:
            CLASS_NAMES.append(item.name)

    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    STEPS_PER_EPOCH = np.ceil(image_count / batch_size)

    data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                         batch_size=batch_size,
                                                         shuffle=shuffle_data,
                                                         target_size=(x, y),
                                                         classes=list(CLASS_NAMES))

    dataset = tf.data.Dataset.from_generator(
        lambda: data_gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, x, y, 3],
                       [None, len(CLASS_NAMES)]))

    return dataset, STEPS_PER_EPOCH, data_gen.class_indices


def generate_dataset(path, x, y, batch_size, shuffle_data=False):
    data_dir = Path(path)
    image_count = len(list(data_dir.glob('*/*.jpg')))

    CLASS_NAMES = []
    for item in data_dir.glob('*'):
        # dir = os.listdir(item)
        # if len(dir) != 0:
            CLASS_NAMES.append(item.name)

    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    STEPS_PER_EPOCH = np.ceil(image_count / batch_size)

    data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                         batch_size=batch_size,
                                                         shuffle=shuffle_data,
                                                         target_size=(x, y),
                                                         classes=list(CLASS_NAMES))

    dataset = tf.data.Dataset.from_generator(
        lambda: data_gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, x, y, 3],
                       [None, len(CLASS_NAMES)]))

    return dataset, STEPS_PER_EPOCH, data_gen.class_indices


def move_validation_to_train_folder(path_validation):
    for root, directories, filenames in os.walk(path_validation):
        for filename in filenames:
            source = os.path.join(root, filename)
            dest = os.path.join(root, filename).replace("validation", "train")
            shutil.move(source, dest)

def generate_split_dataset(path, x, y, batch_size, shuffle_data=False):
    data_dir = Path(path)
    image_count = len(list(data_dir.glob('*/*/*.jpg')))
    print(image_count)
    STEPS_PER_EPOCH = np.ceil(image_count / batch_size)
    dir_list = os.listdir(path)
    datasets = []
    for dir in dir_list:
        print(dir)
        datasets.append(generate_dataset(path + "\\" + dir, x, y, batch_size, shuffle_data))

    final_dataset = datasets[0][0].concatenate(datasets[1][0])
    for i in range(2, len(datasets)):
        final_dataset = final_dataset.concatenate(datasets[i][0])

    return final_dataset, STEPS_PER_EPOCH, datasets[0][2]
    # concat all