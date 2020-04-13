# from src.models.nn_sparse import nn_sparse
# from src.utils.tools import create_dirs
from src.utils.data import load_dataset_pool

import numpy as np

if __name__ == '__main__':
    X_all = load_dataset_pool("..\\..\\dataset\\train_resized\\")
    print(X_all)

# Y = np.zeros(len(X_all))
#
# create_dirs()
# print(len(X_all))
# print(len(Y))
# nn_sparse(X_all, Y, 3, "selu", "adam", "categorical_crossentropy", 500, 1000, 0.0001, "./", "./", [512, 512, 512, 512], 0.2, 0, 0)






