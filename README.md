Projet CUDA : Multiplication de Matrices Optimisée (Tiled Matrix Multiplication)
Ce projet implémente et analyse la multiplication de matrices sur GPU en utilisant NVIDIA CUDA. L'objectif principal est de démontrer l'optimisation des performances grâce à l'utilisation de la Mémoire Partagée (Shared Memory) et de la technique du Tiling (Tuilage).

📂 Structure du Projet
Le projet contient les fichiers sources suivants :

matrix_mul.cu : Implémentation de base avec Tiling (Matrice 4x4, Tuiles 2x2).

comparison.cu : Benchmark comparant la vitesse de la Mémoire Globale vs Mémoire Partagée sur de grandes matrices (N=1024).

exo3.cu : Exercice sur des matrices 8x8 avec des Tuiles 4x4 (Occupation optimale).

exo4.cu : Exercice sur des matrices 8x8 avec des Tuiles 2x2 (Analyse de l'impact des petites tuiles).

🚀 Prérequis et Compilation
Environnement
NVIDIA GPU (Testé sur Tesla T4 via Google Colab).

CUDA Toolkit installé (nvcc).

Compilation
Pour compiler les fichiers, utilisez le compilateur nvcc. Note : Le flag -arch=sm_75 est recommandé pour les GPU récents (comme le T4) pour éviter les erreurs de compatibilité PTX.

Bash

# Compiler le code de base
nvcc -arch=sm_75 matrix_mul.cu -o matrix_mul

# Compiler le benchmark de performance
nvcc -arch=sm_75 comparison.cu -o comparison

# Compiler les exercices
nvcc -arch=sm_75 exo3.cu -o exo3
nvcc -arch=sm_75 exo4.cu -o exo4
Exécution
Bash

./matrix_mul
./comparison
./exo3
./exo4
📊 Concepts Clés & Analyse
1. Mémoire Globale vs Mémoire Partagée
L'exercice de comparaison (comparison.cu) démontre une différence significative de performance :

Mémoire Globale (Global Memory) : Lente (DRAM). Chaque thread va chercher ses données dans la mémoire principale du GPU pour chaque calcul. Latence élevée (~400-800 cycles).

Mémoire Partagée (Shared Memory) : Très rapide (On-chip). Les threads collaborent pour charger une "tuile" de données une seule fois, puis la réutilisent plusieurs fois directement depuis la puce. Latence très faible (~1-2 cycles).

Résultat : L'implémentation "Shared Memory" est nettement plus rapide car elle réduit drastiquement la bande passante mémoire nécessaire.

2. Impact de la taille des Tuiles (Tile Size)
Les exercices 3 et 4 comparent des tuiles de tailles différentes sur une même matrice :

Tuiles 4x4 (Exo 3) : Bonne balance. Le bloc contient 16 threads.

Tuiles 2x2 (Exo 4) : Moins efficace.

Raison : Un warp GPU contient 32 threads. Avec des blocs de 4 threads (2x2), le GPU sous-utilise ses capacités (mauvaise "Occupancy") et perd du temps en synchronisation (__syncthreads) plus fréquente.

📝 Auteur
Projet réalisé dans le cadre d'un laboratoire d'introduction au calcul parallèle sur GPU (CUDA).
