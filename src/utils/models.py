import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import json

def create_submit(model_path, test_dataset, STEPS_PER_EPOCH_TEST, class_indices, path_test_folder, submit_path):
    model = tf.keras.models.load_model(model_path)
    result = predict(model, test_dataset, STEPS_PER_EPOCH_TEST, class_indices, path_test_folder)
    result.to_csv(submit_path, index=False)

def use_mega_detector_on_submit(path_mega_detector, path_submit_src, path_submit_dest):
    with open(path_mega_detector, encoding='utf-8') as json_file:
        bbox_test_full_json = json.load(json_file)
        df_submit = pd.read_csv(path_submit_src)

        for i in range(len(df_submit)):
            if i % 1000 == 0:
                print(i)
            id = df_submit.loc[i]["Id"]
            for n in range(len(bbox_test_full_json["images"])):
                if id in bbox_test_full_json["images"][n]["file"]:
                    if bbox_test_full_json["images"][n]["max_detection_conf"] < 0.5:
                        df_submit.loc[i] = [df_submit.loc[i]["Id"], 0]
                    bbox_test_full_json["images"].pop(n)
                    break
    df_submit.to_csv(path_submit_dest, index=False)

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


def model_fit(model, train_dataset, test_dataset, epochs, STEPS_PER_EPOCH_TRAIN, STEPS_PER_EPOCH_VALIDATION, directory, model_name):
    call_backs = get_callbacks(directory)
    model.fit_generator(generator=train_dataset, validation_data=test_dataset, steps_per_epoch=STEPS_PER_EPOCH_TRAIN,
                        validation_steps=STEPS_PER_EPOCH_VALIDATION, epochs=epochs, verbose=1, use_multiprocessing=True, workers=10) # callbacks=call_backs)
    model.save(directory + "\\" + model_name + ".h5")
    return model

def model_fit_no_val(model, train_dataset,  epochs, STEPS_PER_EPOCH_TRAIN, STEPS_PER_EPOCH_VALIDATION, directory, model_name):
    call_backs = get_callbacks(directory)
    model.fit_generator(generator=train_dataset,  steps_per_epoch=STEPS_PER_EPOCH_TRAIN,
                        validation_steps=STEPS_PER_EPOCH_VALIDATION, epochs=epochs, verbose=1, use_multiprocessing=True, workers=4) # callbacks=call_backs)
    model.save(directory + "\\" + model_name + ".h5")
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
