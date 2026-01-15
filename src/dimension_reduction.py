"""
Module de réduction de dimension pour le projet de sinistralité automobile et climat.

Ce module contient les fonctions pour :
- Analyse en Composantes Principales (ACP/PCA)
- Partial Least Squares (PLS)
- Visualisation des résultats
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
import seaborn as sns


def standardiser_variables(X, feature_names=None):
    """
    Standardise les variables (moyenne=0, écart-type=1).
    
    Paramètres
    ----------
    X : array-like
        Matrice des variables à standardiser
    feature_names : list, optional
        Noms des features
        
    Retourne
    --------
    tuple
        (X_scaled, scaler, feature_names)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if feature_names is None and hasattr(X, 'columns'):
        feature_names = X.columns.tolist()
    
    print(f"Standardisation effectuée sur {X_scaled.shape[1]} variables")
    
    return X_scaled, scaler, feature_names


def analyse_acp(X, n_components=None, feature_names=None):
    """
    Effectue une Analyse en Composantes Principales (ACP/PCA).
    
    Paramètres
    ----------
    X : array-like
        Matrice des variables (doit être standardisée)
    n_components : int, optional
        Nombre de composantes à calculer
    feature_names : list, optional
        Noms des variables
        
    Retourne
    --------
    dict
        Dictionnaire contenant les résultats de l'ACP
    """
    print("=" * 60)
    print("ANALYSE EN COMPOSANTES PRINCIPALES (ACP)")
    print("=" * 60)
    
    # Créer le modèle PCA
    if n_components is None:
        n_components = min(X.shape[0], X.shape[1], 20)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Calculer les statistiques
    variance_expliquee = pca.explained_variance_ratio_
    variance_cumulee = np.cumsum(variance_expliquee)
    
    print(f"\nNombre de composantes : {n_components}")
    print(f"Variance expliquée par les 5 premières composantes : {variance_cumulee[4]*100:.2f}%")
    print(f"Variance expliquée par les 10 premières composantes : {variance_cumulee[9]*100:.2f}%")
    
    # Créer un DataFrame des loadings
    if feature_names is not None:
        loadings_df = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=feature_names
        )
    else:
        loadings_df = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
    
    # Identifier les variables les plus importantes pour chaque composante
    print("\nVariables principales par composante :")
    for i in range(min(5, n_components)):
        top_vars = loadings_df[f'PC{i+1}'].abs().nlargest(5)
        print(f"\nPC{i+1} (variance: {variance_expliquee[i]*100:.2f}%) :")
        for var, val in top_vars.items():
            print(f"  - {var}: {loadings_df.loc[var, f'PC{i+1}']:.3f}")
    
    resultats = {
        'pca': pca,
        'X_pca': X_pca,
        'variance_expliquee': variance_expliquee,
        'variance_cumulee': variance_cumulee,
        'loadings': loadings_df,
        'n_components': n_components
    }
    
    return resultats


def analyser_pls(X, y, n_components=10, feature_names=None):
    """
    Effectue une régression Partial Least Squares (PLS).
    
    Paramètres
    ----------
    X : array-like
        Matrice des variables prédictives (doit être standardisée)
    y : array-like
        Variable cible
    n_components : int
        Nombre de composantes PLS
    feature_names : list, optional
        Noms des variables
        
    Retourne
    --------
    dict
        Dictionnaire contenant les résultats PLS
    """
    print("=" * 60)
    print("PARTIAL LEAST SQUARES (PLS)")
    print("=" * 60)
    
    # Créer le modèle PLS
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X, y)
    
    # Transformer les données
    X_pls = pls.transform(X)
    
    # Récupérer les loadings (coefficients des variables)
    if feature_names is not None:
        loadings_df = pd.DataFrame(
            pls.x_weights_,
            columns=[f'PLS{i+1}' for i in range(n_components)],
            index=feature_names
        )
    else:
        loadings_df = pd.DataFrame(
            pls.x_weights_,
            columns=[f'PLS{i+1}' for i in range(n_components)]
        )
    
    # Calculer le R² pour chaque nombre de composantes
    from sklearn.metrics import r2_score
    r2_scores = []
    for n in range(1, n_components + 1):
        pls_temp = PLSRegression(n_components=n, scale=False)
        pls_temp.fit(X, y)
        y_pred = pls_temp.predict(X)
        r2 = r2_score(y, y_pred)
        r2_scores.append(r2)
    
    print(f"\nNombre de composantes : {n_components}")
    print(f"R² avec {n_components} composantes : {r2_scores[-1]:.4f}")
    
    # Identifier les variables les plus importantes pour chaque composante
    print("\nVariables principales par composante PLS :")
    for i in range(min(5, n_components)):
        top_vars = loadings_df[f'PLS{i+1}'].abs().nlargest(5)
        print(f"\nPLS{i+1} :")
        for var, val in top_vars.items():
            print(f"  - {var}: {loadings_df.loc[var, f'PLS{i+1}']:.3f}")
    
    resultats = {
        'pls': pls,
        'X_pls': X_pls,
        'loadings': loadings_df,
        'n_components': n_components,
        'r2_scores': r2_scores
    }
    
    return resultats


def interpreter_composantes_climat(loadings_df, n_components=5):
    """
    Interprète les composantes principales en termes de facteurs climatiques.
    
    Paramètres
    ----------
    loadings_df : pd.DataFrame
        DataFrame des loadings
    n_components : int
        Nombre de composantes à interpréter
        
    Retourne
    --------
    dict
        Dictionnaire avec interprétations
    """
    interpretations = {}
    
    for i in range(n_components):
        col_name = f'PC{i+1}' if f'PC{i+1}' in loadings_df.columns else f'PLS{i+1}'
        
        # Récupérer les variables avec les plus forts loadings
        top_positive = loadings_df[col_name].nlargest(5)
        top_negative = loadings_df[col_name].nsmallest(5)
        
        # Essayer d'interpréter
        interpretation = f"Composante {i+1}:\n"
        interpretation += "  Variables positives:\n"
        for var, val in top_positive.items():
            interpretation += f"    - {var}: {val:.3f}\n"
        interpretation += "  Variables négatives:\n"
        for var, val in top_negative.items():
            interpretation += f"    - {var}: {val:.3f}\n"
        
        # Détection de patterns
        vars_str = ' '.join([str(v) for v in list(top_positive.index) + list(top_negative.index)])
        
        if 'RR' in vars_str or 'NBJRR' in vars_str:
            interpretation += "  → Semble lié à la PLUIE\n"
        if 'TX' in vars_str or 'TN' in vars_str:
            interpretation += "  → Semble lié à la TEMPÉRATURE\n"
        if 'FF' in vars_str or 'FX' in vars_str:
            interpretation += "  → Semble lié au VENT\n"
        if 'HNEIGE' in vars_str or 'NEIGE' in vars_str:
            interpretation += "  → Semble lié à la NEIGE\n"
        
        interpretations[col_name] = interpretation
    
    return interpretations


def visualiser_variance_expliquee(resultats_acp, save_path=None):
    """
    Visualise la variance expliquée par les composantes principales.
    
    Paramètres
    ----------
    resultats_acp : dict
        Résultats de l'ACP
    save_path : str, optional
        Chemin pour sauvegarder la figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    n_comp = len(resultats_acp['variance_expliquee'])
    indices = np.arange(1, n_comp + 1)
    
    # Variance expliquée par composante
    ax1.bar(indices, resultats_acp['variance_expliquee'] * 100)
    ax1.set_xlabel('Composante Principale')
    ax1.set_ylabel('Variance Expliquée (%)')
    ax1.set_title('Variance Expliquée par Composante')
    ax1.set_xticks(indices)
    
    # Variance cumulée
    ax2.plot(indices, resultats_acp['variance_cumulee'] * 100, marker='o')
    ax2.axhline(y=80, color='r', linestyle='--', label='80%')
    ax2.axhline(y=90, color='g', linestyle='--', label='90%')
    ax2.set_xlabel('Nombre de Composantes')
    ax2.set_ylabel('Variance Cumulée (%)')
    ax2.set_title('Variance Cumulée Expliquée')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(indices)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée : {save_path}")
    
    return fig


def visualiser_loadings(loadings_df, n_components=5, n_vars=10, save_path=None):
    """
    Visualise les loadings des composantes principales.
    
    Paramètres
    ----------
    loadings_df : pd.DataFrame
        DataFrame des loadings
    n_components : int
        Nombre de composantes à visualiser
    n_vars : int
        Nombre de variables à afficher par composante
    save_path : str, optional
        Chemin pour sauvegarder la figure
    """
    n_rows = (n_components + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_components > 1 else [axes]
    
    for i in range(n_components):
        col_name = loadings_df.columns[i]
        
        # Sélectionner les n_vars variables avec les plus forts loadings (en valeur absolue)
        top_loadings = loadings_df[col_name].abs().nlargest(n_vars)
        values = loadings_df.loc[top_loadings.index, col_name]
        
        # Créer le barplot
        colors = ['red' if v < 0 else 'blue' for v in values]
        axes[i].barh(range(len(values)), values, color=colors)
        axes[i].set_yticks(range(len(values)))
        axes[i].set_yticklabels(values.index, fontsize=8)
        axes[i].set_xlabel('Loading')
        axes[i].set_title(f'{col_name} - Top {n_vars} variables')
        axes[i].axvline(x=0, color='black', linewidth=0.8)
        axes[i].grid(True, alpha=0.3, axis='x')
    
    # Masquer les axes non utilisés
    for i in range(n_components, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée : {save_path}")
    
    return fig


def visualiser_biplot(resultats_acp, feature_names=None, save_path=None):
    """
    Crée un biplot des deux premières composantes principales.
    
    Paramètres
    ----------
    resultats_acp : dict
        Résultats de l'ACP
    feature_names : list, optional
        Noms des variables (pour les flèches)
    save_path : str, optional
        Chemin pour sauvegarder la figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scores (observations)
    X_pca = resultats_acp['X_pca']
    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3, s=10)
    
    # Loadings (variables) - flèches
    loadings = resultats_acp['loadings'].iloc[:, :2].values
    scale_factor = max(np.abs(X_pca[:, :2]).max(axis=0)) / max(np.abs(loadings).max(axis=0))
    
    # N'afficher que les 15 variables avec les plus forts loadings
    loading_norms = np.sqrt((loadings ** 2).sum(axis=1))
    top_indices = np.argsort(loading_norms)[-15:]
    
    for i in top_indices:
        ax.arrow(0, 0, loadings[i, 0] * scale_factor, loadings[i, 1] * scale_factor,
                head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.6)
        if feature_names is not None:
            ax.text(loadings[i, 0] * scale_factor * 1.1, 
                   loadings[i, 1] * scale_factor * 1.1,
                   feature_names[i], fontsize=8, ha='center')
    
    ax.set_xlabel(f'PC1 ({resultats_acp["variance_expliquee"][0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({resultats_acp["variance_expliquee"][1]*100:.1f}%)')
    ax.set_title('Biplot ACP - PC1 vs PC2')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée : {save_path}")
    
    return fig
