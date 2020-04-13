import glob
import numpy as np
from PIL import Image
from multiprocessing import Pool, Array


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
        #print('cannot open', jpg_file)
        return None


def load_dataset_pool(path):
    jpg_files_list = get_all_files_path(path)
    jpg_files_list = [path + f for f in jpg_files_list]
    p = Pool()
    params = []
    print(len(jpg_files_list))
    for i in range(len(jpg_files_list)):
        params.append(jpg_files_list[i])

    print("params done")
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


# path = "..\\..\\..\\dataset\\train\\"
# new_path = "..\\..\\..\\dataset\\train_resized\\"
# jpg_files = get_all_files_path(path)
# resize_save_images(jpg_files, 100, 100, path, new_path)
