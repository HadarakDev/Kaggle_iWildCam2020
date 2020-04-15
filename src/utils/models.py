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


def model_fit(model, train_dataset, test_dataset, epochs, batch_size, STEPS_PER_EPOCH, directory):
    call_backs = get_callbacks(directory)
    model.fit_generator(generator=train_dataset, validation_data=test_dataset,  steps_per_epoch=STEPS_PER_EPOCH, epochs=epochs, verbose=1, callbacks=call_backs)
    # model.fit(X_param, Y_param, batch_size=batch_size, verbose=1, epochs=epochs) # , callbacks=call_backs)
    model.save(directory + "\\model.h5")
    return model
