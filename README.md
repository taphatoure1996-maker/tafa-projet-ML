# Projet ML - Impact des Conditions Climatiques sur la Sinistralité Automobile

## 📋 Description du Projet

Ce projet analyse l'**impact des conditions climatiques sur la sinistralité automobile** en utilisant des méthodes avancées de machine learning et de réduction de dimension. Il combine des données d'assurance automobile avec des données météorologiques pour prédire la fréquence et la gravité des sinistres.

## 🎯 Objectifs

1. **Construction d'une base jointe** intégrant données d'assurance et climatiques
2. **Analyse descriptive** approfondie des facteurs de sinistralité
3. **Réduction de dimension** (ACP/PLS) sur 163 variables climatiques
4. **Modélisation prédictive** de la fréquence et gravité des sinistres
5. **Interprétation des résultats** pour des recommandations opérationnelles

## 📊 Données Utilisées

- **pg17trainpol.csv** : ~100 000 polices d'assurance avec caractéristiques des assurés, véhicules et contrats
- **pg17trainclaim.csv** : Base des sinistres avec montants et fréquences
- **DataClimatiques.csv** : 28 162 observations, 163 variables météorologiques mensuelles
- **fremuni17.csv** : Données complémentaires des communes françaises

## 🏗️ Structure du Projet

```
tafa-projet-ML/
│
├── data/                           # Fichiers de données CSV
│   ├── pg17trainpol.csv
│   ├── pg17trainclaim.csv
│   ├── DataClimatiques.csv
│   └── fremuni17.csv
│
├── src/                            # Modules Python
│   ├── __init__.py
│   ├── data_preprocessing.py       # Chargement et nettoyage
│   ├── feature_engineering.py      # Construction des features
│   ├── dimension_reduction.py      # ACP et PLS
│   ├── models.py                   # Modèles ML
│   └── evaluation.py               # Métriques et évaluation
│
├── notebooks/                      # Notebooks Jupyter
│   └── projet_sinistralite_climat.ipynb
│
├── results/                        # Résultats et modèles sauvegardés
│
├── requirements.txt                # Dépendances Python
└── README.md                       # Ce fichier
```

## 🚀 Installation

### 1. Cloner le repository

```bash
git clone https://github.com/taphatoure1996-maker/tafa-projet-ML.git
cd tafa-projet-ML
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

## 📖 Utilisation

### Option 1 : Jupyter Notebook (Recommandé)

```bash
jupyter notebook notebooks/projet_sinistralite_climat.ipynb
```

Le notebook principal contient toutes les analyses avec :
- Visualisations interactives
- Commentaires détaillés
- Résultats des modèles

### Option 2 : Scripts Python

```python
from src import data_preprocessing as dp
from src import feature_engineering as fe
from src import models as md

# Charger les données
donnees = dp.charger_toutes_donnees('.')

# Construire la base finale
df_finale = fe.construire_base_finale('.')

# Modéliser
# ... voir le notebook pour plus de détails
```

## 🔬 Méthodologie

### 1. Prétraitement des Données

- Nettoyage des valeurs manquantes
- Encodage des variables catégorielles
- Création de variables dérivées (âge catégoriel, ratios, etc.)
- Jointure des bases (polices + sinistres + climat)

### 2. Analyse Descriptive

- Distribution de la fréquence et gravité des sinistres
- Analyse par facteurs : couverture, bonus-malus, âge, usage
- Corrélations entre variables
- Statistiques climatiques par région

### 3. Réduction de Dimension

- **ACP (Analyse en Composantes Principales)** : Réduction de 163 variables climatiques
- **PLS (Partial Least Squares)** : Extraction de facteurs prédictifs
- Interprétation : facteurs "pluie", "température", "vent", etc.

### 4. Modélisation Prédictive

#### Fréquence des Sinistres (Classification)
- Régression Logistique
- Régression Pénalisée (Lasso, Ridge, ElasticNet)
- Random Forest Classifier
- XGBoost Classifier

#### Gravité des Sinistres (Régression)
- Régression Linéaire
- Régression Pénalisée (Ridge, Lasso)
- Random Forest Regressor
- XGBoost Regressor

### 5. Évaluation et Sélection

- Validation croisée (5-fold)
- Métriques : AUC, Accuracy, RMSE, MAE, R²
- Comparaison des performances
- Sélection du meilleur modèle

### 6. Interprétabilité

- Feature Importance
- SHAP Values
- Coefficients des modèles linéaires
- Recommandations opérationnelles

## 📈 Résultats Principaux

Les résultats détaillés sont disponibles dans le notebook. Voici les principaux enseignements :

### Facteurs de Risque Identifiés

**Facteurs Assurantiels :**
- Bonus-malus du conducteur
- Âge et expérience du conducteur
- Âge et puissance du véhicule
- Usage du véhicule (professionnel vs privé)

**Facteurs Climatiques :**
- Certaines composantes climatiques montrent une corrélation avec la sinistralité
- L'impact varie selon les régions et les saisons

### Performances des Modèles

Les modèles ensemble (Random Forest, XGBoost) obtiennent généralement les meilleures performances pour les deux tâches (fréquence et gravité).

## 🛠️ Technologies Utilisées

- **Python 3.8+**
- **Pandas & NumPy** : Manipulation de données
- **Scikit-learn** : Machine Learning
- **XGBoost & LightGBM** : Modèles avancés
- **Statsmodels** : Modèles statistiques
- **Matplotlib & Seaborn** : Visualisation
- **SHAP** : Interprétabilité
- **Jupyter** : Notebooks interactifs

## 👥 Applications pour Assureurs

1. **Tarification** : Ajustement des primes basé sur les facteurs de risque
2. **Souscription** : Amélioration de la sélection des risques
3. **Prévention** : Campagnes ciblées sur profils à risque
4. **Provisionnement** : Estimation plus précise des réserves

## 📝 Licence

Ce projet est fourni à des fins éducatives et de recherche.

## 👤 Auteur

Tafa Touré - [GitHub](https://github.com/taphatoure1996-maker)

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

## 📧 Contact

Pour toute question ou suggestion, veuillez ouvrir une issue sur GitHub.