import os
import json
from src.utils.tools import create_folder_from_categories
import matplotlib.pyplot as plt

def crop_dataset(dataset_src_path, dataset_dest_path):
    if not os.path.exists(dataset_dest_path):
        os.mkdir(dataset_dest_path)
    create_folder_from_categories("../annotations/categories.json", dataset_dest_path)

    with open('../annotations/iwildcam2020_megadetector_results.json') as json_data:
        data_dict = json.load(json_data)

def crop(Bbox, image):
    im = plt.imread(image)
    y = im.shape[0]
    x = im.shape[1]
    x1 = round(Bbox[0] * x)
    y1 = round(Bbox[1] * y)
    width = round(Bbox[2] * x)
    height = round(Bbox[3] * y)
    #print(x1, y1, width, height)
    x2, y2 = x1 + width, y1 + height
    new_image = im[y1:y2, x1:x2]
    plt.imshow(new_image)
    plt.savefig('./test_crop/'+image)