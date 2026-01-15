"""
Module de modélisation pour le projet de sinistralité automobile et climat.

Ce module contient les fonctions pour :
- Modèles de fréquence (classification)
- Modèles de gravité (régression)
- Entraînement et prédiction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def diviser_donnees(X, y, test_size=0.2, random_state=42, stratify=None):
    """
    Divise les données en ensembles d'entraînement et de test.
    
    Paramètres
    ----------
    X : array-like
        Matrice des features
    y : array-like
        Variable cible
    test_size : float
        Proportion de l'ensemble de test
    random_state : int
        Seed pour la reproductibilité
    stratify : array-like, optional
        Variable pour stratification
        
    Retourne
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    print(f"Division des données :")
    print(f"  - Entraînement : {len(X_train)} observations")
    print(f"  - Test : {len(X_test)} observations")
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# MODÈLES DE FRÉQUENCE (Classification)
# ============================================================================

def entrainer_logistic_regression(X_train, y_train, max_iter=1000):
    """
    Entraîne une régression logistique classique.
    
    Paramètres
    ----------
    X_train : array-like
        Features d'entraînement
    y_train : array-like
        Cible d'entraînement
    max_iter : int
        Nombre maximal d'itérations
        
    Retourne
    --------
    LogisticRegression
        Modèle entraîné
    """
    print("\n" + "=" * 60)
    print("RÉGRESSION LOGISTIQUE")
    print("=" * 60)
    
    model = LogisticRegression(max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Modèle entraîné : {model.__class__.__name__}")
    
    return model


def entrainer_logistic_penalisee(X_train, y_train, penalty='l2', C=1.0, max_iter=1000):
    """
    Entraîne une régression logistique pénalisée (Ridge, Lasso, ElasticNet).
    
    Paramètres
    ----------
    X_train : array-like
        Features d'entraînement
    y_train : array-like
        Cible d'entraînement
    penalty : str
        Type de pénalisation ('l1', 'l2', 'elasticnet')
    C : float
        Inverse de la force de régularisation
    max_iter : int
        Nombre maximal d'itérations
        
    Retourne
    --------
    LogisticRegression
        Modèle entraîné
    """
    print("\n" + "=" * 60)
    print(f"RÉGRESSION LOGISTIQUE PÉNALISÉE ({penalty.upper()})")
    print("=" * 60)
    
    if penalty == 'elasticnet':
        model = LogisticRegression(
            penalty=penalty, C=C, solver='saga', l1_ratio=0.5,
            max_iter=max_iter, random_state=42
        )
    else:
        solver = 'saga' if penalty == 'l1' else 'lbfgs'
        model = LogisticRegression(
            penalty=penalty, C=C, solver=solver,
            max_iter=max_iter, random_state=42
        )
    
    model.fit(X_train, y_train)
    
    print(f"Modèle entraîné : {penalty.upper()}, C={C}")
    
    return model


def entrainer_random_forest_classifier(X_train, y_train, n_estimators=100, max_depth=None):
    """
    Entraîne un Random Forest Classifier.
    
    Paramètres
    ----------
    X_train : array-like
        Features d'entraînement
    y_train : array-like
        Cible d'entraînement
    n_estimators : int
        Nombre d'arbres
    max_depth : int, optional
        Profondeur maximale des arbres
        
    Retourne
    --------
    RandomForestClassifier
        Modèle entraîné
    """
    print("\n" + "=" * 60)
    print("RANDOM FOREST CLASSIFIER")
    print("=" * 60)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    print(f"Modèle entraîné : {n_estimators} arbres, profondeur={max_depth}")
    
    return model


def entrainer_xgboost_classifier(X_train, y_train, n_estimators=100, max_depth=3, learning_rate=0.1):
    """
    Entraîne un XGBoost Classifier.
    
    Paramètres
    ----------
    X_train : array-like
        Features d'entraînement
    y_train : array-like
        Cible d'entraînement
    n_estimators : int
        Nombre d'arbres
    max_depth : int
        Profondeur maximale des arbres
    learning_rate : float
        Taux d'apprentissage
        
    Retourne
    --------
    XGBClassifier
        Modèle entraîné
    """
    print("\n" + "=" * 60)
    print("XGBOOST CLASSIFIER")
    print("=" * 60)
    
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    print(f"Modèle entraîné : {n_estimators} arbres, LR={learning_rate}")
    
    return model


# ============================================================================
# MODÈLES DE GRAVITÉ (Régression)
# ============================================================================

def entrainer_regression_lineaire(X_train, y_train):
    """
    Entraîne une régression linéaire classique.
    
    Paramètres
    ----------
    X_train : array-like
        Features d'entraînement
    y_train : array-like
        Cible d'entraînement
        
    Retourne
    --------
    Ridge
        Modèle entraîné (Ridge avec alpha=0 ~ régression linéaire)
    """
    print("\n" + "=" * 60)
    print("RÉGRESSION LINÉAIRE")
    print("=" * 60)
    
    model = Ridge(alpha=0.01)  # Alpha très faible ~ régression linéaire
    model.fit(X_train, y_train)
    
    print(f"Modèle entraîné : Régression linéaire")
    
    return model


def entrainer_regression_penalisee(X_train, y_train, method='ridge', alpha=1.0):
    """
    Entraîne une régression pénalisée (Ridge, Lasso, ElasticNet).
    
    Paramètres
    ----------
    X_train : array-like
        Features d'entraînement
    y_train : array-like
        Cible d'entraînement
    method : str
        Type de pénalisation ('ridge', 'lasso', 'elasticnet')
    alpha : float
        Force de régularisation
        
    Retourne
    --------
    model
        Modèle entraîné
    """
    print("\n" + "=" * 60)
    print(f"RÉGRESSION PÉNALISÉE ({method.upper()})")
    print("=" * 60)
    
    if method.lower() == 'ridge':
        model = Ridge(alpha=alpha)
    elif method.lower() == 'lasso':
        model = Lasso(alpha=alpha, max_iter=10000)
    elif method.lower() == 'elasticnet':
        model = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)
    else:
        raise ValueError(f"Méthode inconnue : {method}")
    
    model.fit(X_train, y_train)
    
    print(f"Modèle entraîné : {method.upper()}, alpha={alpha}")
    
    return model


def entrainer_random_forest_regressor(X_train, y_train, n_estimators=100, max_depth=None):
    """
    Entraîne un Random Forest Regressor.
    
    Paramètres
    ----------
    X_train : array-like
        Features d'entraînement
    y_train : array-like
        Cible d'entraînement
    n_estimators : int
        Nombre d'arbres
    max_depth : int, optional
        Profondeur maximale des arbres
        
    Retourne
    --------
    RandomForestRegressor
        Modèle entraîné
    """
    print("\n" + "=" * 60)
    print("RANDOM FOREST REGRESSOR")
    print("=" * 60)
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    print(f"Modèle entraîné : {n_estimators} arbres, profondeur={max_depth}")
    
    return model


def entrainer_xgboost_regressor(X_train, y_train, n_estimators=100, max_depth=3, learning_rate=0.1):
    """
    Entraîne un XGBoost Regressor.
    
    Paramètres
    ----------
    X_train : array-like
        Features d'entraînement
    y_train : array-like
        Cible d'entraînement
    n_estimators : int
        Nombre d'arbres
    max_depth : int
        Profondeur maximale des arbres
    learning_rate : float
        Taux d'apprentissage
        
    Retourne
    --------
    XGBRegressor
        Modèle entraîné
    """
    print("\n" + "=" * 60)
    print("XGBOOST REGRESSOR")
    print("=" * 60)
    
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    print(f"Modèle entraîné : {n_estimators} arbres, LR={learning_rate}")
    
    return model


# ============================================================================
# SÉLECTION DE MODÈLES
# ============================================================================

def validation_croisee(model, X, y, cv=5, scoring='accuracy'):
    """
    Effectue une validation croisée.
    
    Paramètres
    ----------
    model : estimator
        Modèle sklearn
    X : array-like
        Features
    y : array-like
        Cible
    cv : int
        Nombre de folds
    scoring : str
        Métrique de scoring
        
    Retourne
    --------
    dict
        Résultats de la validation croisée
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    resultats = {
        'scores': scores,
        'mean': scores.mean(),
        'std': scores.std()
    }
    
    print(f"Validation croisée ({cv} folds) - {scoring}:")
    print(f"  - Score moyen : {resultats['mean']:.4f} (+/- {resultats['std']:.4f})")
    
    return resultats


def optimiser_hyperparametres(model, param_grid, X_train, y_train, cv=5, scoring='accuracy'):
    """
    Optimise les hyperparamètres avec GridSearchCV.
    
    Paramètres
    ----------
    model : estimator
        Modèle sklearn de base
    param_grid : dict
        Grille de paramètres à tester
    X_train : array-like
        Features d'entraînement
    y_train : array-like
        Cible d'entraînement
    cv : int
        Nombre de folds
    scoring : str
        Métrique de scoring
        
    Retourne
    --------
    GridSearchCV
        Objet GridSearchCV avec le meilleur modèle
    """
    print(f"\nOptimisation des hyperparamètres avec GridSearchCV...")
    print(f"  - Nombre de combinaisons : {len(param_grid)}")
    
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring,
        n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"\nMeilleurs paramètres : {grid_search.best_params_}")
    print(f"Meilleur score : {grid_search.best_score_:.4f}")
    
    return grid_search


def extraire_feature_importance(model, feature_names):
    """
    Extrait l'importance des features d'un modèle.
    
    Paramètres
    ----------
    model : estimator
        Modèle entraîné
    feature_names : list
        Noms des features
        
    Retourne
    --------
    pd.DataFrame
        DataFrame avec l'importance des features
    """
    if hasattr(model, 'feature_importances_'):
        # Modèles tree-based
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Modèles linéaires
        importances = np.abs(model.coef_)
        if len(importances.shape) > 1:
            importances = importances[0]
    else:
        print("Le modèle n'a pas d'attribut 'feature_importances_' ou 'coef_'")
        return None
    
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return df_importance
