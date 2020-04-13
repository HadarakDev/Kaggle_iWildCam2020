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