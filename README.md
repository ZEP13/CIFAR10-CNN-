# Image Classification Web App

## Objectif

Ce projet propose une application web permettant de classer des images dans 10 catégories issues du dataset CIFAR-10 : `plane`, `car`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`. L'utilisateur peut téléverser une image via l'interface web, et le modèle de deep learning pré-entraîné prédit la classe correspondante.

## Architecture du projet

- **Backend** : FastAPI (Python)
- **Frontend** : HTML, CSS, JavaScript
- **Modèle** : Réseau de neurones convolutif (PyTorch)
- **Données** : CIFAR-10

```
IMAGE_CLASSIFICATION/
│
├── app/
│   ├── main.py           # API FastAPI (routes, prédiction)
│   └── model2.py         # Définition et chargement du modèle PyTorch
│
├── templates/
│   └── index.html        # Page web principale
│
├── static/
│   └── style.css         # Feuille de style CSS
│
├── train_net.pth         # Poids du modèle entraîné
├── IMG_class_cnn.ipynb   # Notebook d'entraînement et d'expérimentation
├── README.md             # Ce fichier
└── requirements.txt      # Dépendances Python
```

## Installation

1. **Cloner le dépôt**

   ```bash
   git clone https://github.com/ZEP13/CIFAR10-CNN-
   cd IMAGE_CLASSIFICATION
   ```

2. **Créer un environnement virtuel (optionnel mais recommandé)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # ou venv\Scripts\activate sous Windows
   ```

3. **Installer les dépendances**

   ```bash
   pip install -r requirements.txt
   ```

   Exemple de contenu pour `requirements.txt` :

   ```
   fastapi
   uvicorn
   torch
   torchvision
   pillow
   jinja2
   tqdm
   ```

4. **Vérifier la présence du modèle**
   - Le fichier `train_net.pth` doit être présent à la racine du projet. Si besoin, réentraîner le modèle via le notebook `IMG_class_cnn.ipynb`.

## Lancement de l'application

1. **Démarrer le serveur FastAPI**

   ```bash
   uvicorn app.main:app --reload
   ```

2. **Accéder à l'interface web**
   - Ouvrir un navigateur à l'adresse : [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Utilisation

- Glissez-déposez ou sélectionnez une image dans l'interface.
- Cliquez sur "Predict".
- La classe prédite et la confiance s'affichent.

## Entraînement du modèle

- Utilisez le notebook `IMG_class_cnn.ipynb` pour entraîner ou réentraîner le modèle sur CIFAR-10.
- Les poids sont sauvegardés dans `train_net.pth`.

## Auteurs

- Projet réalisé dans le cadre du cours de Machine Learning & Deep Learning.

## Licence

- Ce projet est distribué sous licence MIT ou autre selon vos besoins.
