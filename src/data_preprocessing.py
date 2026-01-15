"""
Module de prétraitement des données pour le projet de sinistralité automobile et climat.

Ce module contient les fonctions pour charger et nettoyer les différentes bases de données :
- Base des polices d'assurance (pg17trainpol.csv)
- Base des sinistres (pg17trainclaim.csv)
- Base des données climatiques (DataClimatiques.csv)
- Base des communes françaises (fremuni17.csv)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def charger_polices(chemin_fichier, sep=';'):
    """
    Charge la base des polices d'assurance.
    
    Paramètres
    ----------
    chemin_fichier : str
        Chemin vers le fichier pg17trainpol.csv
    sep : str, default=';'
        Séparateur des colonnes
        
    Retourne
    --------
    pd.DataFrame
        DataFrame contenant les polices d'assurance
    """
    print("Chargement de la base des polices...")
    df_polices = pd.read_csv(chemin_fichier, sep=sep)
    print(f"  - {len(df_polices)} polices chargées")
    print(f"  - {len(df_polices.columns)} colonnes")
    
    return df_polices


def charger_sinistres(chemin_fichier, sep=';'):
    """
    Charge la base des sinistres.
    
    Paramètres
    ----------
    chemin_fichier : str
        Chemin vers le fichier pg17trainclaim.csv
    sep : str, default=';'
        Séparateur des colonnes
        
    Retourne
    --------
    pd.DataFrame
        DataFrame contenant les sinistres
    """
    print("Chargement de la base des sinistres...")
    df_sinistres = pd.read_csv(chemin_fichier, sep=sep)
    print(f"  - {len(df_sinistres)} sinistres chargés")
    
    # Nettoyer la colonne claim_amount (format "amount= 1236")
    if 'claim_amount' in df_sinistres.columns:
        df_sinistres['claim_amount'] = df_sinistres['claim_amount'].astype(str).str.replace('amount=', '').str.strip()
        df_sinistres['claim_amount'] = pd.to_numeric(df_sinistres['claim_amount'], errors='coerce')
    
    return df_sinistres


def charger_climat(chemin_fichier, sep=';'):
    """
    Charge la base des données climatiques.
    
    Paramètres
    ----------
    chemin_fichier : str
        Chemin vers le fichier DataClimatiques.csv
    sep : str, default=';'
        Séparateur des colonnes
        
    Retourne
    --------
    pd.DataFrame
        DataFrame contenant les données climatiques
    """
    print("Chargement de la base climatique...")
    df_climat = pd.read_csv(chemin_fichier, sep=sep)
    print(f"  - {len(df_climat)} observations climatiques chargées")
    print(f"  - {len(df_climat.columns)} variables climatiques")
    
    return df_climat


def charger_communes(chemin_fichier, sep=';'):
    """
    Charge la base des communes françaises.
    
    Paramètres
    ----------
    chemin_fichier : str
        Chemin vers le fichier fremuni17.csv
    sep : str, default=';'
        Séparateur des colonnes
        
    Retourne
    --------
    pd.DataFrame
        DataFrame contenant les données des communes
    """
    print("Chargement de la base des communes...")
    df_communes = pd.read_csv(chemin_fichier, sep=sep)
    print(f"  - {len(df_communes)} communes chargées")
    
    return df_communes


def nettoyer_donnees_polices(df_polices):
    """
    Nettoie et prépare les données des polices d'assurance.
    
    Paramètres
    ----------
    df_polices : pd.DataFrame
        DataFrame des polices brutes
        
    Retourne
    --------
    pd.DataFrame
        DataFrame nettoyé
    """
    df = df_polices.copy()
    
    # Convertir les colonnes numériques
    colonnes_numeriques = [
        'pol_bonus', 'pol_duration', 'pol_sit_duration',
        'drv_age1', 'drv_age2', 'drv_age_lic1', 'drv_age_lic2',
        'vh_age', 'vh_cyl', 'vh_din', 'vh_value', 'vh_weight'
    ]
    
    for col in colonnes_numeriques:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Traiter les valeurs manquantes pour les conducteurs secondaires
    if 'drv_age2' in df.columns:
        df['drv_age2'].fillna(0, inplace=True)
    if 'drv_age_lic2' in df.columns:
        df['drv_age_lic2'].fillna(0, inplace=True)
    
    # Créer un identifiant unique police
    if 'id_policy' in df.columns and 'id_year' in df.columns:
        df['policy_year_id'] = df['id_policy'] + '_' + df['id_year'].astype(str)
    
    print(f"Nettoyage terminé : {len(df)} polices, {df.isnull().sum().sum()} valeurs manquantes restantes")
    
    return df


def nettoyer_donnees_sinistres(df_sinistres):
    """
    Nettoie et prépare les données des sinistres.
    
    Paramètres
    ----------
    df_sinistres : pd.DataFrame
        DataFrame des sinistres bruts
        
    Retourne
    --------
    pd.DataFrame
        DataFrame nettoyé
    """
    df = df_sinistres.copy()
    
    # Créer un identifiant unique pour joindre avec les polices
    if 'id_client' in df.columns and 'id_vehicle' in df.columns and 'id_year' in df.columns:
        df['id_policy'] = df['id_client'] + '-' + df['id_vehicle']
        df['policy_year_id'] = df['id_policy'] + '_' + df['id_year'].astype(str)
    
    # Convertir claim_nb en numérique
    if 'claim_nb' in df.columns:
        df['claim_nb'] = pd.to_numeric(df['claim_nb'], errors='coerce')
    
    # Supprimer les lignes avec montant manquant
    if 'claim_amount' in df.columns:
        df = df[df['claim_amount'].notna()]
        df = df[df['claim_amount'] > 0]
    
    print(f"Nettoyage terminé : {len(df)} sinistres valides")
    
    return df


def agregation_sinistres_par_police(df_sinistres):
    """
    Agrège les sinistres au niveau police-année.
    
    Paramètres
    ----------
    df_sinistres : pd.DataFrame
        DataFrame des sinistres nettoyés
        
    Retourne
    --------
    pd.DataFrame
        DataFrame agrégé avec nombre et montant total des sinistres par police
    """
    if 'policy_year_id' not in df_sinistres.columns:
        raise ValueError("La colonne 'policy_year_id' doit exister dans df_sinistres")
    
    # Agréger par police-année
    agg_dict = {
        'claim_nb': 'sum',
        'claim_amount': 'sum',
        'id_claim': 'count'
    }
    
    df_agg = df_sinistres.groupby('policy_year_id').agg(agg_dict).reset_index()
    df_agg.columns = ['policy_year_id', 'nb_sinistres_total', 'montant_total', 'nb_lignes_sinistres']
    
    # Créer l'indicateur binaire de sinistralité
    df_agg['a_sinistre'] = 1
    
    print(f"Agrégation terminée : {len(df_agg)} polices avec sinistre(s)")
    
    return df_agg


def encoder_variables_categorielles(df, colonnes_cat=None):
    """
    Encode les variables catégorielles.
    
    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame à encoder
    colonnes_cat : list, optional
        Liste des colonnes catégorielles à encoder
        
    Retourne
    --------
    pd.DataFrame
        DataFrame avec variables encodées
    dict
        Dictionnaire des mappings utilisés
    """
    df_encoded = df.copy()
    mappings = {}
    
    if colonnes_cat is None:
        # Détecter automatiquement les colonnes catégorielles
        colonnes_cat = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in colonnes_cat:
        if col in df_encoded.columns:
            # Utiliser le label encoding pour les variables catégorielles
            df_encoded[col] = df_encoded[col].astype('category')
            mappings[col] = dict(enumerate(df_encoded[col].cat.categories))
            df_encoded[col] = df_encoded[col].cat.codes
    
    print(f"Encodage terminé : {len(colonnes_cat)} colonnes encodées")
    
    return df_encoded, mappings


def preparer_variables_climatiques(df_climat):
    """
    Prépare les variables climatiques pour l'analyse.
    
    Paramètres
    ----------
    df_climat : pd.DataFrame
        DataFrame des données climatiques brutes
        
    Retourne
    --------
    pd.DataFrame
        DataFrame climatique nettoyé
    list
        Liste des variables numériques climatiques
    """
    df = df_climat.copy()
    
    # Identifier les colonnes métadonnées vs variables climatiques
    cols_meta = ['NUM_POSTE', 'NOM_USUEL', 'LAT', 'LON', 'ALTI', 'AAAAMM', 'DEP']
    
    # Liste des variables climatiques numériques (exclure les colonnes de qualité Q*)
    cols_climat = [col for col in df.columns 
                   if col not in cols_meta 
                   and not col.startswith('Q')
                   and not col.startswith('NB')]
    
    # Convertir en numérique
    for col in cols_climat:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Extraire année et mois de AAAAMM
    if 'AAAAMM' in df.columns:
        df['annee'] = df['AAAAMM'] // 100
        df['mois'] = df['AAAAMM'] % 100
    
    print(f"Préparation climatique : {len(cols_climat)} variables identifiées")
    
    return df, cols_climat


def charger_toutes_donnees(dossier_data='.'):
    """
    Charge toutes les bases de données du projet.
    
    Paramètres
    ----------
    dossier_data : str
        Chemin vers le dossier contenant les fichiers CSV
        
    Retourne
    --------
    dict
        Dictionnaire contenant tous les DataFrames chargés
    """
    dossier = Path(dossier_data)
    
    donnees = {
        'polices': charger_polices(dossier / 'pg17trainpol.csv'),
        'sinistres': charger_sinistres(dossier / 'pg17trainclaim.csv'),
        'climat': charger_climat(dossier / 'DataClimatiques.csv'),
        'communes': charger_communes(dossier / 'fremuni17.csv')
    }
    
    return donnees
