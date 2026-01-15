# Documentation Technique

## 📋 Architecture du Projet

### Vue d'Ensemble

Le projet est organisé en modules Python indépendants qui peuvent être utilisés séparément ou ensemble. Chaque module a une responsabilité spécifique suivant le principe de séparation des préoccupations.

```
src/
├── data_preprocessing.py      # Chargement et nettoyage des données
├── feature_engineering.py     # Construction et transformation des features
├── dimension_reduction.py     # ACP, PLS et visualisations
├── models.py                  # Modèles de ML (classification et régression)
└── evaluation.py              # Métriques, évaluation et interprétabilité
```

## 🔄 Pipeline de Traitement

### 1. Chargement des Données

```python
donnees = dp.charger_toutes_donnees('.')
# Retourne: {'polices': df, 'sinistres': df, 'climat': df, 'communes': df}
```

**Particularités :**
- Séparateur `;` (format européen)
- Encodage automatique détecté
- Gestion des colonnes avec espaces

### 2. Prétraitement

#### Polices d'Assurance

```python
df_polices = dp.nettoyer_donnees_polices(donnees['polices'])
```

**Opérations effectuées :**
- Conversion des colonnes numériques
- Gestion des valeurs manquantes pour conducteur secondaire
- Création d'identifiants uniques `policy_year_id`
- Traitement des valeurs aberrantes

#### Sinistres

```python
df_sinistres = dp.nettoyer_donnees_sinistres(donnees['sinistres'])
```

**Opérations effectuées :**
- Nettoyage du format `claim_amount` ("amount= 1236" → 1236)
- Suppression des sinistres sans montant
- Création de l'identifiant `policy_year_id`
- Conversion en types numériques

#### Agrégation des Sinistres

```python
df_sinistres_agg = dp.agregation_sinistres_par_police(df_sinistres)
```

**Résultat :**
- Une ligne par police-année
- `nb_sinistres_total` : nombre total de sinistres
- `montant_total` : montant cumulé
- `a_sinistre` : indicateur binaire (0/1)

### 3. Jointure des Bases

#### Polices + Sinistres

```python
df_base = fe.joindre_polices_sinistres(df_polices, df_sinistres_agg)
```

**Type de jointure :** LEFT JOIN (conserve toutes les polices)

**Résultat :**
- Polices avec sinistres : `a_sinistre=1`, montants renseignés
- Polices sans sinistre : `a_sinistre=0`, montants à 0

#### Ajout des Données Climatiques

```python
df_climat_agg = fe.agregation_climat_par_dept_annee(df_climat, variables_climat)
df_finale = fe.joindre_avec_climat(df_base, df_climat_agg)
```

**Logique de jointure :**
1. Extraction du département depuis `pol_insee_code` (2 premiers chiffres)
2. Mapping `id_year` vers année numérique (Year 0 = 2017)
3. Agrégation climatique par département-année (moyenne annuelle)
4. Jointure sur `(code_dept, annee)`

### 4. Construction des Features

#### Variables Dérivées

```python
df_finale = fe.creer_variables_derivees(df_finale)
```

**Variables créées :**

| Catégorie | Variables |
|-----------|-----------|
| Véhicule | `vh_age_cat`, `vh_value_cat`, `vh_puissance_cat` |
| Conducteur | `drv_age1_cat`, `drv_experience` |
| Contrat | `pol_bonus_cat`, `a_conducteur_secondaire` |
| Ratios | `ratio_puissance_poids` |

**Catégorisation :**
- Utilise `pd.cut()` pour les intervalles fixes
- Utilise `pd.qcut()` pour les quantiles

## 🎯 Variables Cibles

### Fréquence

**Variable :** `a_sinistre` (binaire 0/1)

**Distribution typique :**
- 0 (pas de sinistre) : ~89%
- 1 (au moins un sinistre) : ~11%

**Problématique :** Déséquilibre de classes
**Solutions appliquées :**
- Stratification lors du split train/test
- Métriques adaptées (AUC-ROC plutôt qu'accuracy)

### Gravité

**Variable :** `montant_total` (continue, > 0)

**Distribution :**
- Asymétrique (distribution lognormale)
- Présence de valeurs extrêmes
- Moyenne : ~1100€, Médiane : ~600€

**Transformation possible :** `log(montant_total + 1)`

## 🔬 Réduction de Dimension

### ACP (Analyse en Composantes Principales)

```python
resultats_acp = dr.analyse_acp(X_climat_scaled, n_components=20)
```

**Processus :**
1. Standardisation (mean=0, std=1)
2. Calcul des composantes principales
3. Tri par variance expliquée
4. Analyse des loadings

**Interprétation des composantes :**
- PC1 : Souvent liée à la température moyenne
- PC2 : Souvent liée aux précipitations
- PC3+ : Facteurs plus spécifiques (vent, neige, etc.)

### PLS (Partial Least Squares)

```python
resultats_pls = dr.analyser_pls(X_climat_scaled, y, n_components=10)
```

**Avantage sur l'ACP :**
- Maximise la covariance avec la variable cible
- Composantes directement prédictives
- Meilleure pour la modélisation

## 🤖 Modèles Implémentés

### Classification (Fréquence)

| Modèle | Fonction | Hyperparamètres Clés |
|--------|----------|---------------------|
| Logistic Regression | `entrainer_logistic_regression()` | `max_iter=1000` |
| Lasso (L1) | `entrainer_logistic_penalisee()` | `penalty='l1', C=0.1` |
| Ridge (L2) | `entrainer_logistic_penalisee()` | `penalty='l2', C=1.0` |
| ElasticNet | `entrainer_logistic_penalisee()` | `penalty='elasticnet', l1_ratio=0.5` |
| Random Forest | `entrainer_random_forest_classifier()` | `n_estimators=100, max_depth=10` |
| XGBoost | `entrainer_xgboost_classifier()` | `n_estimators=100, learning_rate=0.1` |

### Régression (Gravité)

| Modèle | Fonction | Hyperparamètres Clés |
|--------|----------|---------------------|
| Linear Regression | `entrainer_regression_lineaire()` | `alpha=0.01` |
| Ridge | `entrainer_regression_penalisee()` | `method='ridge', alpha=1.0` |
| Lasso | `entrainer_regression_penalisee()` | `method='lasso', alpha=1.0` |
| Random Forest | `entrainer_random_forest_regressor()` | `n_estimators=100, max_depth=10` |
| XGBoost | `entrainer_xgboost_regressor()` | `n_estimators=100, learning_rate=0.1` |

## 📊 Métriques d'Évaluation

### Classification

| Métrique | Description | Interprétation |
|----------|-------------|----------------|
| Accuracy | Proportion de bonnes prédictions | Biaisé si classes déséquilibrées |
| Precision | VP / (VP + FP) | Fiabilité des prédictions positives |
| Recall | VP / (VP + FN) | Capacité à détecter les sinistres |
| F1-Score | Moyenne harmonique Precision/Recall | Équilibre entre les deux |
| **AUC-ROC** | Aire sous la courbe ROC | **Métrique principale** (non biaisée) |

### Régression

| Métrique | Description | Unité | Préférence |
|----------|-------------|-------|------------|
| MSE | Mean Squared Error | €² | Pénalise fortement les erreurs |
| **RMSE** | Root MSE | € | **Métrique principale** |
| MAE | Mean Absolute Error | € | Robuste aux outliers |
| R² | Coefficient de détermination | 0-1 | Variance expliquée |
| MAPE | Mean Absolute % Error | % | Erreur relative |

## 🔍 Interprétabilité

### Feature Importance

**Méthodes implémentées :**

1. **Coefficients linéaires** (Logistic/Linear Regression)
   - Valeurs directes des coefficients
   - Signe indique la direction de l'effet

2. **Importances Gini** (Random Forest, XGBoost)
   - Basées sur les réductions d'impureté
   - Normalisées pour sommer à 1

3. **SHAP Values** (tous modèles)
   - Valeurs de Shapley
   - Contribution de chaque feature par prédiction
   - Visualisation avec summary plots

## ⚙️ Optimisation des Hyperparamètres

### GridSearchCV

```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.1, 0.3]
}

grid_search = md.optimiser_hyperparametres(
    model, param_grid, X_train, y_train, cv=5
)
```

### Validation Croisée

```python
resultats_cv = md.validation_croisee(model, X, y, cv=5, scoring='roc_auc')
```

**Stratégies :**
- 5-fold pour rapidité
- 10-fold pour stabilité
- Stratification pour classification

## 🚀 Performance et Optimisation

### Temps d'Exécution Typiques

| Opération | Durée |
|-----------|-------|
| Chargement données | 5-10s |
| Prétraitement | 10-15s |
| Jointures | 5-10s |
| ACP (50 variables) | 2-5s |
| Random Forest (100 arbres) | 30-60s |
| XGBoost (100 arbres) | 15-30s |

### Conseils d'Optimisation

1. **Données volumineuses :**
   - Échantillonner pour les tests
   - Utiliser `n_jobs=-1` (parallélisation)

2. **Modèles lents :**
   - Réduire `n_estimators`
   - Réduire `max_depth`
   - Utiliser `early_stopping_rounds` (XGBoost)

3. **Mémoire limitée :**
   - Charger les données par chunks
   - Utiliser types de données optimisés (`category`, `float32`)

## 📝 Bonnes Pratiques

### Code

- Documentation en français (docstrings)
- Type hints pour les paramètres
- Gestion des erreurs avec messages explicites
- Logging des étapes principales

### Modélisation

- Toujours séparer train/test **avant** toute transformation
- Standardiser les features numériques
- Gérer les valeurs manquantes explicitement
- Valider sur plusieurs métriques

### Reproductibilité

- `random_state=42` partout
- Sauvegarder les modèles (pickle/joblib)
- Documenter les versions des packages
- Sauvegarder les résultats intermédiaires

## 🔗 Dépendances Clés

| Package | Version | Usage |
|---------|---------|-------|
| pandas | ≥1.5.0 | Manipulation de données |
| numpy | ≥1.23.0 | Calculs numériques |
| scikit-learn | ≥1.2.0 | ML classique |
| xgboost | ≥1.7.0 | Gradient boosting |
| matplotlib | ≥3.6.0 | Visualisation |
| seaborn | ≥0.12.0 | Visualisation statistique |
| shap | ≥0.41.0 | Interprétabilité |
| statsmodels | ≥0.13.0 | Modèles statistiques |

## 📚 Références

### Machine Learning

- Hastie, T., et al. (2009). *The Elements of Statistical Learning*
- James, G., et al. (2013). *An Introduction to Statistical Learning*

### Assurance

- Denuit, M., et al. (2007). *Actuarial Modelling of Claim Counts*
- Ohlsson, E., & Johansson, B. (2010). *Non-Life Insurance Pricing with GLM*

### Python & ML

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
