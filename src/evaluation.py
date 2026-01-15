"""
Module d'évaluation pour le projet de sinistralité automobile et climat.

Ce module contient les fonctions pour :
- Évaluation des modèles de classification
- Évaluation des modèles de régression
- Comparaison de modèles
- Interprétabilité (SHAP values)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# MÉTRIQUES DE CLASSIFICATION
# ============================================================================

def evaluer_classification(y_true, y_pred, y_pred_proba=None):
    """
    Évalue un modèle de classification.
    
    Paramètres
    ----------
    y_true : array-like
        Vraies étiquettes
    y_pred : array-like
        Prédictions
    y_pred_proba : array-like, optional
        Probabilités prédites
        
    Retourne
    --------
    dict
        Dictionnaire des métriques
    """
    metriques = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_pred_proba is not None:
        metriques['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metriques


def afficher_metriques_classification(metriques, nom_modele="Modèle"):
    """
    Affiche les métriques de classification de manière formatée.
    
    Paramètres
    ----------
    metriques : dict
        Dictionnaire des métriques
    nom_modele : str
        Nom du modèle
    """
    print(f"\n{'='*60}")
    print(f"MÉTRIQUES - {nom_modele}")
    print(f"{'='*60}")
    print(f"Accuracy  : {metriques['accuracy']:.4f}")
    print(f"Precision : {metriques['precision']:.4f}")
    print(f"Recall    : {metriques['recall']:.4f}")
    print(f"F1-Score  : {metriques['f1_score']:.4f}")
    if 'roc_auc' in metriques:
        print(f"ROC-AUC   : {metriques['roc_auc']:.4f}")


def matrice_confusion(y_true, y_pred, labels=None, save_path=None):
    """
    Affiche la matrice de confusion.
    
    Paramètres
    ----------
    y_true : array-like
        Vraies étiquettes
    y_pred : array-like
        Prédictions
    labels : list, optional
        Noms des classes
    save_path : str, optional
        Chemin pour sauvegarder la figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels if labels else ['Classe 0', 'Classe 1'],
                yticklabels=labels if labels else ['Classe 0', 'Classe 1'])
    plt.ylabel('Vraie étiquette')
    plt.xlabel('Prédiction')
    plt.title('Matrice de Confusion')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée : {save_path}")
    
    return cm


def courbe_roc(y_true, y_pred_proba, save_path=None):
    """
    Trace la courbe ROC.
    
    Paramètres
    ----------
    y_true : array-like
        Vraies étiquettes
    y_pred_proba : array-like
        Probabilités prédites
    save_path : str, optional
        Chemin pour sauvegarder la figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Courbe ROC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée : {save_path}")
    
    return fpr, tpr, auc


# ============================================================================
# MÉTRIQUES DE RÉGRESSION
# ============================================================================

def evaluer_regression(y_true, y_pred):
    """
    Évalue un modèle de régression.
    
    Paramètres
    ----------
    y_true : array-like
        Vraies valeurs
    y_pred : array-like
        Prédictions
        
    Retourne
    --------
    dict
        Dictionnaire des métriques
    """
    metriques = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
    # MAPE (Mean Absolute Percentage Error)
    mask = y_true != 0
    if mask.sum() > 0:
        metriques['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return metriques


def afficher_metriques_regression(metriques, nom_modele="Modèle"):
    """
    Affiche les métriques de régression de manière formatée.
    
    Paramètres
    ----------
    metriques : dict
        Dictionnaire des métriques
    nom_modele : str
        Nom du modèle
    """
    print(f"\n{'='*60}")
    print(f"MÉTRIQUES - {nom_modele}")
    print(f"{'='*60}")
    print(f"MSE  : {metriques['mse']:.2f}")
    print(f"RMSE : {metriques['rmse']:.2f}")
    print(f"MAE  : {metriques['mae']:.2f}")
    print(f"R²   : {metriques['r2']:.4f}")
    if 'mape' in metriques:
        print(f"MAPE : {metriques['mape']:.2f}%")


def graphique_predictions_vs_reelles(y_true, y_pred, save_path=None):
    """
    Graphique des prédictions vs valeurs réelles.
    
    Paramètres
    ----------
    y_true : array-like
        Vraies valeurs
    y_pred : array-like
        Prédictions
    save_path : str, optional
        Chemin pour sauvegarder la figure
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    
    # Ligne de référence y=x
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Prédiction parfaite')
    
    plt.xlabel('Valeurs Réelles')
    plt.ylabel('Prédictions')
    plt.title('Prédictions vs Valeurs Réelles')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée : {save_path}")


def graphique_residus(y_true, y_pred, save_path=None):
    """
    Graphique des résidus.
    
    Paramètres
    ----------
    y_true : array-like
        Vraies valeurs
    y_pred : array-like
        Prédictions
    save_path : str, optional
        Chemin pour sauvegarder la figure
    """
    residus = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Résidus vs prédictions
    ax1.scatter(y_pred, residus, alpha=0.5, s=10)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Prédictions')
    ax1.set_ylabel('Résidus')
    ax1.set_title('Résidus vs Prédictions')
    ax1.grid(True, alpha=0.3)
    
    # Distribution des résidus
    ax2.hist(residus, bins=50, edgecolor='black')
    ax2.set_xlabel('Résidus')
    ax2.set_ylabel('Fréquence')
    ax2.set_title('Distribution des Résidus')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée : {save_path}")


# ============================================================================
# COMPARAISON DE MODÈLES
# ============================================================================

def comparer_modeles(resultats_modeles, metrique='accuracy'):
    """
    Compare les performances de plusieurs modèles.
    
    Paramètres
    ----------
    resultats_modeles : dict
        Dictionnaire {nom_modele: metriques}
    metrique : str
        Métrique à comparer
        
    Retourne
    --------
    pd.DataFrame
        DataFrame de comparaison
    """
    # Extraire la métrique de tous les modèles
    comparaison = []
    for nom, metriques in resultats_modeles.items():
        if metrique in metriques:
            comparaison.append({
                'Modèle': nom,
                metrique: metriques[metrique]
            })
    
    df_comp = pd.DataFrame(comparaison).sort_values(metrique, ascending=False)
    
    print(f"\n{'='*60}")
    print(f"COMPARAISON DES MODÈLES - {metrique}")
    print(f"{'='*60}")
    print(df_comp.to_string(index=False))
    
    return df_comp


def visualiser_comparaison_modeles(resultats_modeles, metriques_a_comparer=None, save_path=None):
    """
    Visualise la comparaison de plusieurs modèles.
    
    Paramètres
    ----------
    resultats_modeles : dict
        Dictionnaire {nom_modele: metriques}
    metriques_a_comparer : list, optional
        Liste des métriques à comparer
    save_path : str, optional
        Chemin pour sauvegarder la figure
    """
    if metriques_a_comparer is None:
        # Détecter les métriques disponibles
        first_metrics = list(resultats_modeles.values())[0]
        metriques_a_comparer = list(first_metrics.keys())
    
    n_metriques = len(metriques_a_comparer)
    fig, axes = plt.subplots(1, n_metriques, figsize=(5 * n_metriques, 5))
    
    if n_metriques == 1:
        axes = [axes]
    
    for i, metrique in enumerate(metriques_a_comparer):
        noms_modeles = []
        valeurs = []
        
        for nom, metriques in resultats_modeles.items():
            if metrique in metriques:
                noms_modeles.append(nom)
                valeurs.append(metriques[metrique])
        
        axes[i].barh(noms_modeles, valeurs)
        axes[i].set_xlabel(metrique)
        axes[i].set_title(f'Comparaison - {metrique}')
        axes[i].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée : {save_path}")


# ============================================================================
# INTERPRÉTABILITÉ
# ============================================================================

def visualiser_feature_importance(importance_df, top_n=20, save_path=None):
    """
    Visualise l'importance des features.
    
    Paramètres
    ----------
    importance_df : pd.DataFrame
        DataFrame avec colonnes 'feature' et 'importance'
    top_n : int
        Nombre de features à afficher
    save_path : str, optional
        Chemin pour sauvegarder la figure
    """
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Features les Plus Importantes')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée : {save_path}")


def calculer_shap_values(model, X, feature_names=None, max_display=20):
    """
    Calcule et visualise les SHAP values pour l'interprétabilité.
    
    Paramètres
    ----------
    model : estimator
        Modèle entraîné
    X : array-like
        Features
    feature_names : list, optional
        Noms des features
    max_display : int
        Nombre de features à afficher
        
    Retourne
    --------
    shap_values
        SHAP values calculées
    """
    try:
        import shap
        
        print("\nCalcul des SHAP values...")
        
        # Créer l'explainer approprié selon le type de modèle
        if hasattr(model, 'predict_proba'):
            # Modèle de classification
            explainer = shap.TreeExplainer(model) if hasattr(model, 'estimators_') else shap.Explainer(model, X[:100])
        else:
            # Modèle de régression
            explainer = shap.TreeExplainer(model) if hasattr(model, 'estimators_') else shap.Explainer(model, X[:100])
        
        shap_values = explainer.shap_values(X[:1000])  # Limiter à 1000 observations pour la performance
        
        # Visualisation summary plot
        plt.figure(figsize=(10, 8))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X[:1000], feature_names=feature_names, max_display=max_display, show=False)
        else:
            shap.summary_plot(shap_values, X[:1000], feature_names=feature_names, max_display=max_display, show=False)
        plt.tight_layout()
        
        print("SHAP values calculées et visualisées")
        
        return shap_values
        
    except ImportError:
        print("SHAP non disponible. Installer avec: pip install shap")
        return None
    except Exception as e:
        print(f"Erreur lors du calcul des SHAP values : {e}")
        return None
