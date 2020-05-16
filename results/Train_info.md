1)  
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

2)  
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

3)  More ramp and bigger structure
    - batch = 8192
    - number_of_classes = 266 # 266 withotu empty
    - size_x = 32
    - size_y = 32
    - epochs = 30
    - activation = "selu"
    - layers = [512,512,512,512]
    - optimizer = "adam"
    - build_lrfn(0.001, 0.005, 0.0001, 15)