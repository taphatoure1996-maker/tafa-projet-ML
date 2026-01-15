"""
Module de construction des features pour le projet de sinistralité automobile et climat.

Ce module contient les fonctions pour :
- Joindre les différentes bases de données
- Créer des variables dérivées
- Construire la base finale pour la modélisation
"""

import pandas as pd
import numpy as np
from datetime import datetime


def joindre_polices_sinistres(df_polices, df_sinistres_agg):
    """
    Joint les polices avec les sinistres agrégés.
    
    Paramètres
    ----------
    df_polices : pd.DataFrame
        DataFrame des polices nettoyées
    df_sinistres_agg : pd.DataFrame
        DataFrame des sinistres agrégés par police-année
        
    Retourne
    --------
    pd.DataFrame
        DataFrame joint au niveau police-année
    """
    print("Jointure polices-sinistres...")
    
    # Créer l'identifiant policy_year_id dans les polices si nécessaire
    if 'policy_year_id' not in df_polices.columns:
        df_polices['policy_year_id'] = df_polices['id_policy'] + '_' + df_polices['id_year'].astype(str)
    
    # Jointure LEFT pour garder toutes les polices (même sans sinistre)
    df_joint = df_polices.merge(
        df_sinistres_agg,
        on='policy_year_id',
        how='left'
    )
    
    # Compléter les polices sans sinistre
    df_joint['a_sinistre'] = df_joint['a_sinistre'].fillna(0)
    df_joint['nb_sinistres_total'] = df_joint['nb_sinistres_total'].fillna(0)
    df_joint['montant_total'] = df_joint['montant_total'].fillna(0)
    
    print(f"  - {len(df_joint)} polices-années dans la base jointe")
    print(f"  - {df_joint['a_sinistre'].sum()} polices avec sinistre(s)")
    print(f"  - Taux de sinistralité : {df_joint['a_sinistre'].mean()*100:.2f}%")
    
    return df_joint


def preparer_jointure_climat(df_climat):
    """
    Prépare les données climatiques pour la jointure avec les polices.
    
    Paramètres
    ----------
    df_climat : pd.DataFrame
        DataFrame des données climatiques
        
    Retourne
    --------
    pd.DataFrame
        DataFrame climatique préparé pour jointure
    """
    df = df_climat.copy()
    
    # Créer une clé de jointure basée sur le département
    # Le NUM_POSTE peut être relié au département
    if 'DEP' in df.columns:
        df['code_dept'] = df['DEP'].astype(str).str.zfill(2)
    
    # Extraire année et mois si ce n'est pas déjà fait
    if 'annee' not in df.columns and 'AAAAMM' in df.columns:
        df['annee'] = df['AAAAMM'] // 100
        df['mois'] = df['AAAAMM'] % 100
    
    return df


def mapper_insee_vers_dept(df_polices):
    """
    Extrait le code département du code INSEE.
    
    Paramètres
    ----------
    df_polices : pd.DataFrame
        DataFrame des polices avec pol_insee_code
        
    Retourne
    --------
    pd.DataFrame
        DataFrame avec colonne code_dept ajoutée
    """
    df = df_polices.copy()
    
    if 'pol_insee_code' in df.columns:
        # Extraire les 2 premiers chiffres du code INSEE
        df['code_dept'] = df['pol_insee_code'].astype(str).str[:2]
    
    return df


def agregation_climat_par_dept_annee(df_climat, variables_climat):
    """
    Agrège les données climatiques par département et année.
    
    Paramètres
    ----------
    df_climat : pd.DataFrame
        DataFrame des données climatiques
    variables_climat : list
        Liste des variables climatiques à agréger
        
    Retourne
    --------
    pd.DataFrame
        DataFrame climatique agrégé
    """
    print("Agrégation des données climatiques par département-année...")
    
    # Préparer les clés de groupement
    if 'code_dept' not in df_climat.columns:
        df_climat = preparer_jointure_climat(df_climat)
    
    # Sélectionner les variables à agréger (celles qui sont numériques)
    vars_a_agreger = [v for v in variables_climat if v in df_climat.columns and pd.api.types.is_numeric_dtype(df_climat[v])]
    
    # Agréger par département et année (moyenne annuelle)
    agg_dict = {v: 'mean' for v in vars_a_agreger}
    
    df_climat_agg = df_climat.groupby(['code_dept', 'annee']).agg(agg_dict).reset_index()
    
    print(f"  - {len(df_climat_agg)} lignes département-année")
    print(f"  - {len(vars_a_agreger)} variables climatiques agrégées")
    
    return df_climat_agg


def joindre_avec_climat(df_polices_sinistres, df_climat_agg):
    """
    Joint la base polices-sinistres avec les données climatiques.
    
    Paramètres
    ----------
    df_polices_sinistres : pd.DataFrame
        DataFrame joint polices-sinistres
    df_climat_agg : pd.DataFrame
        DataFrame climatique agrégé
        
    Retourne
    --------
    pd.DataFrame
        DataFrame final avec données climatiques
    """
    print("Jointure avec les données climatiques...")
    
    df = df_polices_sinistres.copy()
    
    # Extraire le département du code INSEE si nécessaire
    if 'code_dept' not in df.columns:
        df = mapper_insee_vers_dept(df)
    
    # Mapper id_year vers année numérique
    # Supposons que Year 0 = 2017, Year 1 = 2018, etc.
    year_mapping = {f'Year {i}': 2017 + i for i in range(10)}
    df['annee'] = df['id_year'].map(year_mapping)
    
    # Jointure avec les données climatiques
    df_final = df.merge(
        df_climat_agg,
        on=['code_dept', 'annee'],
        how='left'
    )
    
    print(f"  - {len(df_final)} polices dans la base finale")
    nb_sans_climat = df_final[[col for col in df_climat_agg.columns if col not in ['code_dept', 'annee']]].isnull().all(axis=1).sum()
    print(f"  - {nb_sans_climat} polices sans données climatiques")
    
    return df_final


def creer_variables_derivees(df):
    """
    Crée des variables dérivées utiles pour la modélisation.
    
    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame de base
        
    Retourne
    --------
    pd.DataFrame
        DataFrame avec variables dérivées
    """
    df_new = df.copy()
    
    # Variables véhicule
    if 'vh_age' in df_new.columns:
        df_new['vh_age_cat'] = pd.cut(df_new['vh_age'], bins=[0, 2, 5, 10, 20, 100], 
                                       labels=['Neuf', 'Recent', 'Moyen', 'Ancien', 'Tres_ancien'])
    
    if 'vh_value' in df_new.columns:
        df_new['vh_value_cat'] = pd.qcut(df_new['vh_value'], q=4, 
                                          labels=['Bas', 'Moyen', 'Eleve', 'Tres_eleve'], duplicates='drop')
    
    if 'vh_din' in df_new.columns:
        df_new['vh_puissance_cat'] = pd.cut(df_new['vh_din'], bins=[0, 80, 120, 150, 300],
                                            labels=['Faible', 'Moyenne', 'Forte', 'Tres_forte'])
    
    # Variables conducteur
    if 'drv_age1' in df_new.columns:
        df_new['drv_age1_cat'] = pd.cut(df_new['drv_age1'], bins=[0, 25, 35, 50, 65, 100],
                                        labels=['Jeune', 'Adulte_jeune', 'Adulte', 'Senior', 'Tres_senior'])
    
    if 'drv_age_lic1' in df_new.columns:
        df_new['drv_experience'] = pd.cut(df_new['drv_age_lic1'], bins=[0, 2, 5, 10, 50],
                                          labels=['Novice', 'Peu_exp', 'Experimente', 'Tres_exp'])
    
    # Variables contrat
    if 'pol_bonus' in df_new.columns:
        df_new['pol_bonus_cat'] = pd.cut(df_new['pol_bonus'], bins=[0, 0.5, 0.7, 0.9, 1.0, 3.5],
                                         labels=['Excellent', 'Bon', 'Moyen', 'Malus_faible', 'Malus_fort'])
    
    # Indicateur conducteur secondaire
    if 'drv_drv2' in df_new.columns:
        df_new['a_conducteur_secondaire'] = (df_new['drv_drv2'] == 'Yes').astype(int)
    
    # Ratio puissance/poids (si disponible)
    if 'vh_din' in df_new.columns and 'vh_weight' in df_new.columns:
        df_new['ratio_puissance_poids'] = df_new['vh_din'] / (df_new['vh_weight'] + 1)
    
    print(f"Variables dérivées créées")
    
    return df_new


def selectionner_features_modelisation(df, inclure_climat=True):
    """
    Sélectionne les features pertinentes pour la modélisation.
    
    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame complet
    inclure_climat : bool
        Si True, inclut les variables climatiques
        
    Retourne
    --------
    list
        Liste des noms de colonnes à utiliser comme features
    """
    # Features assurance de base
    features_base = [
        'pol_bonus', 'pol_duration', 'pol_sit_duration',
        'drv_age1', 'drv_age_lic1',
        'vh_age', 'vh_cyl', 'vh_din', 'vh_value', 'vh_weight'
    ]
    
    # Features catégorielles encodées
    features_cat = [
        'pol_coverage', 'pol_pay_freq', 'pol_usage',
        'drv_sex1', 'vh_fuel', 'vh_type'
    ]
    
    # Variables dérivées
    features_derivees = [
        'a_conducteur_secondaire', 'ratio_puissance_poids'
    ]
    
    features = features_base + features_cat + features_derivees
    
    # Ajouter les variables climatiques si demandé
    if inclure_climat:
        # Identifier les colonnes climatiques dans le DataFrame
        cols_climat = [col for col in df.columns 
                      if any(prefix in col for prefix in ['RR', 'TX', 'TN', 'TM', 'UN', 'FF', 'INST', 'GLOT'])
                      and not col.startswith('NB')]
        features.extend(cols_climat[:50])  # Limiter à 50 variables climat pour l'instant
    
    # Filtrer pour ne garder que les colonnes existantes dans le DataFrame
    features_disponibles = [f for f in features if f in df.columns]
    
    print(f"  - {len(features_disponibles)} features sélectionnées pour la modélisation")
    
    return features_disponibles


def preparer_donnees_modelisation(df, features, target='a_sinistre'):
    """
    Prépare les données pour la modélisation (X, y).
    
    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame complet
    features : list
        Liste des features à utiliser
    target : str
        Nom de la variable cible
        
    Retourne
    --------
    tuple
        (X, y, indices) où X est la matrice des features, y la cible, indices les index
    """
    # Supprimer les lignes avec valeurs manquantes sur les features ou la target
    df_clean = df[features + [target]].dropna()
    
    X = df_clean[features]
    y = df_clean[target]
    indices = df_clean.index
    
    print(f"Données de modélisation préparées :")
    print(f"  - {len(X)} observations")
    print(f"  - {len(features)} features")
    print(f"  - Distribution target : {y.value_counts().to_dict()}")
    
    return X, y, indices


def construire_base_finale(dossier_data='.'):
    """
    Construit la base finale en joignant toutes les sources.
    
    Paramètres
    ----------
    dossier_data : str
        Chemin vers le dossier contenant les données
        
    Retourne
    --------
    pd.DataFrame
        Base de données finale prête pour l'analyse
    """
    from .data_preprocessing import (
        charger_toutes_donnees,
        nettoyer_donnees_polices,
        nettoyer_donnees_sinistres,
        agregation_sinistres_par_police,
        preparer_variables_climatiques
    )
    
    print("=" * 60)
    print("CONSTRUCTION DE LA BASE FINALE")
    print("=" * 60)
    
    # 1. Charger les données
    donnees = charger_toutes_donnees(dossier_data)
    
    # 2. Nettoyer les polices
    df_polices = nettoyer_donnees_polices(donnees['polices'])
    
    # 3. Nettoyer et agréger les sinistres
    df_sinistres = nettoyer_donnees_sinistres(donnees['sinistres'])
    df_sinistres_agg = agregation_sinistres_par_police(df_sinistres)
    
    # 4. Joindre polices et sinistres
    df_base = joindre_polices_sinistres(df_polices, df_sinistres_agg)
    
    # 5. Préparer les données climatiques
    df_climat, variables_climat = preparer_variables_climatiques(donnees['climat'])
    df_climat_agg = agregation_climat_par_dept_annee(df_climat, variables_climat)
    
    # 6. Joindre avec les données climatiques
    df_finale = joindre_avec_climat(df_base, df_climat_agg)
    
    # 7. Créer les variables dérivées
    df_finale = creer_variables_derivees(df_finale)
    
    print("=" * 60)
    print(f"BASE FINALE CONSTRUITE : {len(df_finale)} observations")
    print("=" * 60)
    
    return df_finale
