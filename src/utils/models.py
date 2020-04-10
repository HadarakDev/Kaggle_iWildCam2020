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

# base path
def model_fit(model, X_param, Y_param, epochs, batch_size, save_path, save_dir, basePath):
    log_dir = basePath + save_dir
    call_backs = get_callbacks(log_dir)
    model.fit(X_param, Y_param, batch_size=batch_size, verbose=1, epochs=epochs, callbacks=call_backs, validation_split=0.2)
    model.save(save_path)
    return model