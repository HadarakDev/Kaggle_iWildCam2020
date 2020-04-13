import glob
import numpy as np
from PIL import Image

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
            print(np.shape(dataset))
            i = i + 1
            if i == 100:
                break
        except IOError:
            print('cannot open', file)
    return dataset

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
