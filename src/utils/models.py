import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

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
                        validation_steps=STEPS_PER_EPOCH_VALIDATION, epochs=epochs, verbose=1, callbacks=call_backs)
    model.save(directory + "\\model.h5")
    return model
