import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

def get_callbacks(log_dir):
    #Tensorboard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    #Avoid overfit from accuracy 
    earlystop_callback = EarlyStopping(monitor='sparse_categorical_accuracy', min_delta=0.0001, patience=10)
    earlystop_val_callback = EarlyStopping(monitor='val_sparse_categorical_accuracy', min_delta=0.0001, patience=10)
    return [tensorboard_callback, earlystop_callback, earlystop_val_callback]


def get_accuracy(model, X, Y):
    print("NEW MODEL PREDICTION")
    res = model.predict(X)
    return accuracy_score(Y, res)


def model_fit(model, train_dataset, epochs, batch_size, STEPS_PER_EPOCH):
    # log_dir = basePath + save_dir
    # call_backs = get_callbacks(log_dir)
    # print(log_dir)
    # print(call_backs)
    model.fit_generator(generator=train_dataset, steps_per_epoch=STEPS_PER_EPOCH, epochs=epochs, verbose=1)
    # model.fit(train_dataset.take(100), epochs=3)
    # model.fit(X_param, Y_param, batch_size=batch_size, verbose=1, epochs=epochs) # , callbacks=call_backs)
    # model.save(save_path)
    return model
