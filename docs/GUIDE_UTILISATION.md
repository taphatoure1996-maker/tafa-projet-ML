# Guide d'Utilisation du Projet

## 🚀 Démarrage Rapide

### Installation

```bash
# 1. Cloner le repository
git clone https://github.com/taphatoure1996-maker/tafa-projet-ML.git
cd tafa-projet-ML

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

## 📝 Trois Façons d'Utiliser le Projet

### 1. Exécution Rapide - Script de Test

Pour vérifier que tout fonctionne correctement :

```bash
python test_projet.py
```

Ce script teste :
- Le chargement des données
- Le prétraitement
- La construction des features
- Un modèle simple de classification

**Durée : ~30 secondes**

### 2. Exemple Simple - Script d'Analyse

Pour une analyse rapide et complète :

```bash
python exemple_utilisation.py
```

Ce script réalise :
- Analyse descriptive de la sinistralité
- Entraînement de plusieurs modèles
- Comparaison des performances
- Identification des features importantes

**Durée : ~1-2 minutes**

### 3. Analyse Complète - Notebook Jupyter

Pour l'analyse complète avec visualisations :

```bash
jupyter notebook notebooks/projet_sinistralite_climat.ipynb
```

Le notebook contient :
- Toutes les analyses descriptives
- Réduction de dimension (ACP/PLS)
- 8+ modèles de machine learning
- Visualisations interactives
- Interprétabilité (SHAP values)

**Durée : 15-30 minutes**

## 📊 Structure des Données

### Fichiers Requis

Les fichiers CSV suivants doivent être dans le répertoire racine :

- `pg17trainpol.csv` - Polices d'assurance (~100k lignes)
- `pg17trainclaim.csv` - Sinistres (~14k lignes)
- `DataClimatiques.csv` - Données météo (~28k lignes)
- `fremuni17.csv` - Communes françaises (~50k lignes)

### Format des Données

**Important :** Les fichiers CSV utilisent le séparateur `;` (format européen).

## 🔧 Utilisation Programmatique

### Exemple Minimal

```python
from src import data_preprocessing as dp
from src import feature_engineering as fe
from src import models as md

# Charger les données
donnees = dp.charger_toutes_donnees('.')

# Préparer
df_polices = dp.nettoyer_donnees_polices(donnees['polices'])
df_sinistres = dp.nettoyer_donnees_sinistres(donnees['sinistres'])
df_sinistres_agg = dp.agregation_sinistres_par_police(df_sinistres)

# Joindre
df_base = fe.joindre_polices_sinistres(df_polices, df_sinistres_agg)
df_finale = fe.creer_variables_derivees(df_base)

# Modéliser
features = ['pol_bonus', 'drv_age1', 'vh_age', 'vh_din', 'vh_value']
X, y, _ = fe.preparer_donnees_modelisation(df_finale, features, 'a_sinistre')

X_train, X_test, y_train, y_test = md.diviser_donnees(X, y)
model = md.entrainer_random_forest_classifier(X_train, y_train)
```

## 📈 Résultats Attendus

### Taux de Sinistralité

- Taux global : ~11%
- Variation selon l'usage : 8-25%
- Variation selon l'âge du véhicule : 3-15%

### Performances des Modèles

**Fréquence (Classification) :**
- Logistic Regression : AUC ~0.61
- Random Forest : AUC ~0.62
- XGBoost : AUC ~0.63

**Gravité (Régression) :**
- Random Forest : R² ~0.15-0.25
- XGBoost : R² ~0.20-0.30

### Features Importantes

1. Âge du véhicule
2. Valeur du véhicule
3. Bonus-malus
4. Âge du conducteur
5. Puissance du véhicule

## 🛠️ Modules Disponibles

### `src.data_preprocessing`

Fonctions pour charger et nettoyer les données :
- `charger_toutes_donnees()` - Charge tous les fichiers
- `nettoyer_donnees_polices()` - Nettoie les polices
- `nettoyer_donnees_sinistres()` - Nettoie les sinistres
- `preparer_variables_climatiques()` - Prépare les variables météo

### `src.feature_engineering`

Fonctions pour construire les features :
- `joindre_polices_sinistres()` - Joint les bases
- `creer_variables_derivees()` - Crée des variables supplémentaires
- `selectionner_features_modelisation()` - Sélectionne les features
- `preparer_donnees_modelisation()` - Prépare X et y

### `src.dimension_reduction`

Fonctions pour la réduction de dimension :
- `analyse_acp()` - Analyse en Composantes Principales
- `analyser_pls()` - Partial Least Squares
- `visualiser_variance_expliquee()` - Graphiques de variance
- `interpreter_composantes_climat()` - Interprétation

### `src.models`

Fonctions pour la modélisation :
- `entrainer_logistic_regression()` - Régression logistique
- `entrainer_random_forest_classifier()` - Random Forest
- `entrainer_xgboost_classifier()` - XGBoost
- Et versions régression pour la gravité

### `src.evaluation`

Fonctions pour l'évaluation :
- `evaluer_classification()` - Métriques de classification
- `evaluer_regression()` - Métriques de régression
- `comparer_modeles()` - Compare plusieurs modèles
- `calculer_shap_values()` - SHAP pour interprétabilité

## 🐛 Dépannage

### Erreur : "ModuleNotFoundError"

```bash
pip install -r requirements.txt
```

### Erreur : "FileNotFoundError" pour les CSV

Vérifiez que les fichiers CSV sont dans le bon répertoire :
```bash
ls -la *.csv
```

### Performances lentes

- Réduisez le nombre de features climatiques
- Utilisez un sous-échantillon des données pour les tests
- Réduisez `n_estimators` pour Random Forest/XGBoost

### Problèmes avec SHAP

```bash
pip install shap
```

Si toujours des problèmes, commentez les sections SHAP dans le notebook.

## 📚 Documentation des Variables

Consultez le fichier `Description des variables de la base de données.docx` pour :
- Description de toutes les colonnes
- Format et unités
- Valeurs possibles

## 🤝 Support

Pour toute question :
1. Consultez le README.md
2. Examinez les exemples dans `exemple_utilisation.py`
3. Ouvrez une issue sur GitHub

## 📄 Licence

Ce projet est fourni à des fins éducatives et de recherche.
