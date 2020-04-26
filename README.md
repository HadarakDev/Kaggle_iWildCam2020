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

# To do

- Clasisication 5 classes uniforme => avec val uniforme
- tester 


# Kaggle submission

- cnn with real data ( 100x100 ) => predict only 0 => score 0.395
- cnn with only 5 top classes => predict 0 and 372 => score 0.389
- cnn with only 5 top classes and reduced 0 to 17k images => predict 0, 372 and 374 => score 0.164