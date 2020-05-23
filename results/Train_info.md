## 1  
    - batch = 8192
    - number_of_classes = 266 # 266 withotu empty
    - size_x = 32
    - size_y = 32
    - epochs = 30
    - activation = "selu"
    - layers = [64,64,64,64]
    - optimizer = "adam"
    - build_lrfn(0.001, 0.002, 0.0001, 10)
   
   Name =   nn_8192_30_selu_adam_[64, 64, 64, 64]_aug_without_empty
-  loss: 2.7290 - categorical_accuracy: 0.3075 - val_loss: 2.7502 - val_categorical_accuracy: 0.3086

## 2  
    - batch = 8192
    - number_of_classes = 266 # 266 withotu empty
    - size_x = 32
    - size_y = 32
    - epochs = 30
    - activation = "selu"
    - layers = [256,256,256,256]
    - optimizer = "adam"
    - build_lrfn(0.001, 0.002, 0.0001, 10)

Name = nn_8192_30_selu_adam_[256, 256, 256, 256]_aug_without_empty
- loss: 2.3020 - categorical_accuracy: 0.3914 - val_loss: 2.3198 - val_categorical_accuracy: 0.3903

## 3 More ramp and bigger structure
    - batch = 8192
    - number_of_classes = 266 # 266 withotu empty
    - size_x = 32
    - size_y = 32
    - epochs = 30
    - activation = "selu"
    - layers = [512,512,512,512]
    - optimizer = "adam"
    - build_lrfn(0.001, 0.005, 0.0001, 15)
Name = nn_8192_30_selu_adam_[512, 512, 512, 512]_aug_without_empty
- loss: 3.6302 - categorical_accuracy: 0.3490 - val_loss: 3.6085 - val_categorical_accuracy: 0.3481

## 4 no scheduler and more epochs
    batch = 8192
    number_of_classes = 266 # 266 withotu empty
    size_x = 32
    size_y = 32
    epochs = 100
    learning_rate = 0.0001
    activation = "selu"
    layers = [256, 256, 256, 256]
    optimizer = "adam"
 
Name = nn_8192_100_selu_adam_[256, 256, 256, 256]100epochs_aug_without_empty
- loss: 2.6809 - categorical_accuracy: 0.3211 - val_loss: 2.7007 - val_categorical_accuracy: 0.3221 
##### Stop at epochs 47 ( early stopping 0.01)

## 5 no scheduler and more epochs
    batch = 8192
    number_of_classes = 266 # 266 withotu empty
    size_x = 32
    size_y = 32
    epochs = 100
    learning_rate = 0.0001
    activation = "selu"
    layers = [512,512,512,512, 512,512,512,512]
    optimizer = "adam"
Name = nn_8192_100_selu_adam_[512, 512, 512, 512, 512, 512, 512, 512]_0.0001_100ep_aug_without_empty
-loss: 2.0487 - categorical_accuracy: 0.4416 - val_loss: 1.9950 - val_categorical_accuracy: 0.4578

## 6 linear 
    batch = 8192
    number_of_classes = 266 # 266 withotu empty
    size_x = 32
    size_y = 32
    epochs = 100
    learning_rate = 0.0001
    activation = "selu"
    layers = []
    optimizer = "adam"
    loss = "categorical_crossentropy"
    pooling = "avg_pool"
    
    - Name  = lin_8192_100_selu_adam_0.0001_100ep_aug_without_empty
    loss: 7.9390 - categorical_accuracy: 0.0030 - val_loss: 8.8940 - val_categorical_accuracy: 0.0030
    
    
 ## 7 CNN
     batch = 100
    number_of_classes = 266 # 266 withotu empty
    size_x = 32
    size_y = 32
    epochs = 100
    learning_rate = 0.0001
    activation = "selu"
    layers = [256,256,256]
    optimizer = "adam"
    loss = "categorical_crossentropy"
    pooling = "avg_pool"
    pooling 2
Name = cnn_1000_100_selu_adam_[256, 256, 256, 256]_0.0001_aug_without_empty
    
## 7 CNN avwc empty
    batch = 1000
    number_of_classes = 267  # 266 withotu empty 
    size_x = 32
    size_y = 32
    epochs = 100
    learning_rate = 0.0001
    activation = "selu"
    layers = [256,256,256,256]
    optimizer = "adam"
    loss = "categorical_crossentropy"
    pooling = "avg_pool"
Name = cnn_1000_100_selu_adam_[256, 256, 256, 256]_0.0001_aug
loss: 1.3741 - categorical_accuracy: 0.6187 - val_loss: 1.4019 - val_categorical_accuracy: 0.6151

## 8 CNN aug equal 1000
    batch = 1000
    number_of_classes = 267  # 266 withotu empty 
    size_x = 32
    size_y = 32
    epochs = 100
    learning_rate = 0.0001
    activation = "selu"
    layers = [256,256,256,256]
    optimizer = "adam"
    loss = "categorical_crossentropy"
    pooling = "avg_pool"
    Name = 
    60%

## 9 CNN aug equal 5000 ( with empty)
    batch = 1000
    number_of_classes = 267  # 266 withotu empty
    size_x = 32
    size_y = 32
    epochs = 100
    learning_rate = 0.0001
    activation = "selu"
    layers = [256,256,256,256]
    optimizer = "adam"
    loss = "categorical_crossentropy"
    pooling = "avg_pool"
  - cnn_1000_100_selu_adam_[256, 256, 256, 256]_0.0001_aug_equal_5000
 - loss: 1.6365 - categorical_accuracy: 0.5453 - val_loss: 1.6361 - val_categorical_accuracy: 0.5454
