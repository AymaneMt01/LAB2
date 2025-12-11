
# Rapport de Laboratoire : Classification MNIST avec PyTorch

Ce rapport détaille la mise en œuvre de deux architectures de réseaux de neurones profonds pour la classification du dataset MNIST (chiffres manuscrits) en utilisant la bibliothèque PyTorch.

## 1. Architecture CNN (CNNModel)

### Préparation des Données
Une approche manuelle a été privilégiée pour le chargement des données afin de maîtriser le pipeline :
* **Extraction :** Création des fonctions `load_mnist_images` et `load_mnist_labels` pour lire les fichiers binaires.
* **Prétraitement :** Normalisation des pixels (0 à 1) et redimensionnement.
* **PyTorch :** Conversion en `Tensors` (ajout de la dimension du canal gris), création de `Datasets` personnalisés et utilisation de `DataLoaders` pour la gestion des batchs.

### Architecture du Modèle
Le modèle **CNNModel** hérite de `nn.Module` et suit une structure séquentielle classique :
1.  **Couches Convolutionnelles :**
    * *Conv1* : 32 filtres, noyau 3x3, stride 1, padding 1.
    * *Conv2* : 64 filtres.
    * *Conv3* : 128 filtres.
2.  **Pooling :** Application de `MaxPool2d` pour réduire la dimensionnalité de moitié après chaque convolution.
3.  **Classification :** Aplatissement des données (Flatten) suivi de couches entièrement connectées (`Fully Connected`) pour la sortie finale.



### Configuration de l'Entraînement
* **Matériel :** Exécution sur GPU.
* **Hyper-paramètres :**
    * Optimiseur : Adam (Learning rate : 0.001).
    * Fonction de perte : `CrossEntropyLoss` (Multi-classes).
    * Durée : 5 époques.

---

## 2. Approche Faster R-CNN (RCNNClassifier)

### Implémentation
Bien que le chargement des données suive la même logique rigoureuse (lecture binaire, redimensionnement 28x28, normalisation /255), l'architecture du second modèle, désigné comme **RCNNClassifier**, a été affinée :

* **Extraction de caractéristiques (Feature Extraction) :**
    * Série de convolutions (32 $\rightarrow$ 64 $\rightarrow$ 128 filtres) avec préservation de taille via padding, suivies systématiquement de Max Pooling (noyau 2x2).
    * Réduction progressive de la taille des images : $28 \times 28 \rightarrow 14 \times 14 \rightarrow 7 \times 7$.
* **Classification (Tête du réseau) :**
    * Transformation du tenseur $128 \times 7 \times 7$ en vecteur plat.
    * Couche Dense 1 (`fc1`) : 512 neurones.
    * Couche Dense 2 (`fc2`) : 10 neurones (correspondant aux classes 0-9).

### Cycle d'Entraînement
Le pipeline `forward` et la boucle d'entraînement `train_model` gèrent le transfert des batchs sur GPU, le calcul des gradients (`backward`), l'optimisation des poids et le suivi en temps réel de la précision et de la perte.

---

## 3. Analyse Comparative des Performances

Les deux modèles ont été évalués selon quatre métriques clés. Voici les résultats obtenus après 5 époques :

| Métrique | Modèle 1 (CNNModel) | Modèle 2 (RCNNClassifier) | Analyse |
| :--- | :--- | :--- | :--- |
| **Précision (Accuracy)** | 98.90% | **99.06%** | Le RCNNClassifier offre une légère amélioration de la précision (+0.16%). |
| **F1 Score** | 0.9890 | **0.9906** | La cohérence entre précision et rappel est meilleure sur le second modèle. |
| **Perte (Loss)** | 0.0290 | **0.0289** | La convergence est quasi identique, avec un très léger avantage pour le modèle 2. |
| **Temps d'Entraînement** | **587.17 s** (~10 min) | 2238.45 s (~37 min) | Le modèle 2 est presque **4 fois plus lent**. |

### Conclusion
Le modèle **RCNNClassifier** (Modèle 2) est le plus performant en termes de qualité de prédiction, atteignant une précision remarquable de 99.06%. Cependant, ce gain de performance a un coût computationnel très élevé, nécessitant beaucoup plus de temps d'entraînement que le CNN standard. Pour une application en temps réel ou avec des ressources limitées, le **CNNModel** (Modèle 1) reste un choix plus efficient, offrant un excellent compromis rapidité/précision.
