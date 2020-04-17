import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import json

def get_callbacks(log_dir):
    # Tensorboard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # Avoid overfit from accuracy
    # earlystop_callback = EarlyStopping(monitor='categorical_accuracy', min_delta=0.0001, patience=10)
    # earlystop_val_callback = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0.0001, patience=10)
    return [tensorboard_callback]


def get_accuracy(model, X, Y):
    print("NEW MODEL PREDICTION")
    res = model.predict(X)
    return accuracy_score(Y, res)


def model_fit(model, train_dataset, test_dataset, epochs, STEPS_PER_EPOCH_TRAIN, STEPS_PER_EPOCH_VALIDATION, directory):
    call_backs = get_callbacks(directory)
    model.fit_generator(generator=train_dataset, validation_data=test_dataset, steps_per_epoch=STEPS_PER_EPOCH_TRAIN,
                        validation_steps=STEPS_PER_EPOCH_VALIDATION, epochs=epochs, verbose=1, use_multiprocessing=True, workers=4) # callbacks=call_backs)
    model.save(directory + "\\model.h5")
    return model

def model_fit_no_val(model, train_dataset,  epochs, STEPS_PER_EPOCH_TRAIN, STEPS_PER_EPOCH_VALIDATION, directory):
    call_backs = get_callbacks(directory)
    model.fit_generator(generator=train_dataset,  steps_per_epoch=STEPS_PER_EPOCH_TRAIN,
                        validation_steps=STEPS_PER_EPOCH_VALIDATION, epochs=epochs, verbose=1, use_multiprocessing=True, workers=4) # callbacks=call_backs)
    model.save(directory + "\\model.h5")
    return model


def predict(model, test_dataset, STEPS_PER_EPOCH_TEST, class_indices, path_test_folder):
    pred = model.predict_generator(test_dataset, verbose=1, steps=STEPS_PER_EPOCH_TEST)
    predicted_class_indices = np.argmax(pred, axis=1)
    labels = dict((v, k) for k, v in class_indices.items())
    predictions = [labels[k] for k in predicted_class_indices]
    filenames = []
    for filename in test_dataset.list_files(path_test_folder + "\\*.jpg"):
        filenames.append(str(filename.numpy()).split("\\")[-1].split(".")[0])
    results = pd.DataFrame({"Id": filenames,
                            "id_left": predictions})

    with open("..\\categories.json") as json_file:
        data = json.load(json_file)
        my_categories = pd.DataFrame(data["categories"])

    results = pd.merge(results, my_categories, left_on='id_left', right_on='name')

    del results['id_left']
    del results['name']
    del results['count']

    mapping = {results.columns[0]: 'Id', results.columns[1]: 'Category'}
    results = results.rename(columns=mapping)

    return results
