# Idée

Pas de raw => Trop violent
100x100 et/ou 50x50 => Comparer

Pour le crop avec l'outil pretrain:

    - Crop strict
    - Crop avec plus de données, aggrandir la bbox? Peut etre rajout d'info sur le fond, verdure, temps ect

Recup des images en plus des années precedente pour avoir une meilleur répartition des classes.

Faire de la data augmentation sur les classes les moins fréquentes. => Rajouter une images avec un ID unique + rajouter dans le Json de référence.

Stocker les valeurs image ID et Y dans un csv, comme pour la submission pour le training. => Ca nous permets de split/shuffle ou autre nos images dans la memoire

Bagging avec des models différents

# Workflow

- Train avec 2020 images ( full dataset)
- Validation avec custom validation ( images 2017 uniquement classe commune avec 2020)
- Matrix de confusion ( identifié les classes mal prédites )
- rajouter des datas avec 2018, 2019 ( transfer learning)

# Kaggle submission

- cnn with real data ( 100x100 ) => predict only 0 => score 0.395
- cnn with only 5 top classes => predict 0 and 372 => score 0.389
- cnn with only 5 top classes and reduced 0 to 17k images => predict 0, 372 and 374 => score 0.164
- detect empty with animal detector ( treshold 10) => score 0.293
- detect empty with animal detector ( treshold 30) => score 0.319
- detect empty with animal detector ( treshold 50) => 0.328

- NN [512,512, 512, 512,512] / Model detector 0.3  (lr 0.001) => 0.380
- NN [512,512, 512, 512,512] / Model detector 0.5  (lr 0.001) => 0.389 
- Config : NN [64, 64, 64, 64] 100 epochs without empty class- val accuracy : 0.6 - model detector treshold 0.3 => 0.340
- Config : NN [64, 64, 64, 64] 100 epochs without empty class - val accuracy : 0.6 - model detector treshold 0.5 => 0.349 
- Config : NN [256, 256, 256, 256] 100 epochs without empty class- val accuracy : 0.77 model detector treshold 0.5 => 0.349

# Types of Models
### Linear

Config : 
 size 32x32
- nn_4096_0.0001_30_selu_adam_[256,256,256,256] => ~35 bloque a 10 epochs
- nn_4096_0.0001_30_selu_adam_[512-512-512-512] => loss: 2.7976 - categorical_accuracy: 0.3484 - val_loss: 2.8297 - val_categorical_accuracy: 0.3467 ( 10 epochs bloque) 

35 % ( probleme predit que classe 0 car le dataset est principalement 0 ( peu importe la structure du nn))


- data avec augmentation (equal , 1000)
nn_4096_0.001_30_selu_adam_[64,64,64,64] => loss: 2.3290 - categorical_accuracy: 0.4161 - val_loss: 2.3503 - val_categorical_accuracy: 0.4198

