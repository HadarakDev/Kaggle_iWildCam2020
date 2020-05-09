import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from skimage import transform
from skimage import io
import os
from pathlib import Path
import time

folder_path = "E:\\dataset\\train_data_aug\\" # CHANGE ME

def get_image_path(folder_path):
    images = []
    for folder in os.listdir(folder_path):
        sub_folder_path = folder_path + folder
        for file in os.listdir(sub_folder_path):
            images.append(os.path.join(sub_folder_path, file))
    return images


def random_rotation(image_array: ndarray):
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)


def random_noise(image_array: ndarray):
    return sk.util.random_noise(image_array)


def horizontal_flip(image_array: ndarray):
    return image_array[:, ::-1]


def augment_folder(images, img_to_create):
    img_done = 0

    while img_done < img_to_create:
        image_path = random.choice(images)
        image_to_transform = io.imread(image_path)
        transformed_img = random.choice(list_transformations)(image_to_transform)


        i = 0
        random.shuffle(range_retransform)
        while range_retransform[i] != 0:
            transformed_img = random.choice(list_transformations)(transformed_img)
            i = i + 1

        img_name = "_data_aug{}.".join(image_path.split("\\")[-1].split(".")).format(img_done)

        path = "\\".join(image_path.split("\\")[:-1])
        io.imsave(path + "\\" + img_name, transformed_img)
        img_done = img_done + 1

list_transformations = [random_rotation,
                        random_noise,
                        horizontal_flip
                        ]

range_retransform = list(range(4))



data_dir = Path(folder_path)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)


dir_list = os.listdir(folder_path)
datasets = []
image_counts = []
images = []
for dir in dir_list:

    data_dir = Path(folder_path + "\\" + dir)
    image_count = len(list(data_dir.glob('*.jpg')))
    l_files = list(data_dir.glob('*.jpg'))
    l_files = [str(f) for f in l_files]
    images.append(l_files)
    image_counts.append(image_count)

print(images)

print(max(image_counts), min(image_counts))
max_int = max(image_counts)
ratios = [abs((x / max_int) - 1.0) for x in image_counts]
print(ratios)

avg = sum(image_counts) / len(image_counts)
print(image_counts)
img_to_creates = [avg - x if x > 0 else 0 for x in image_counts]
print(img_to_creates)

t1 = time.time()
for i in range(len(images)):
    if (img_to_creates[i] > 0):
        print(img_to_creates[i])
        print(images[i])
        augment_folder(images[i], img_to_creates[i])

print(time.time() - t1)
# Pour RM les fichiers en subshell :
# find . -wholename "*/*_data_aug*.jpg" -exec rm {} \;



