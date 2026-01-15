"""
Exemple d'utilisation rapide du projet de sinistralité automobile.

Ce script montre comment utiliser les modules pour une analyse simple.
"""

import sys
sys.path.append('.')

from src import data_preprocessing as dp
from src import feature_engineering as fe
from src import models as md
from src import evaluation as ev
import pandas as pd


def exemple_analyse_simple():
    """
    Exemple d'analyse simple de la sinistralité automobile.
    """
    print("\n" + "="*70)
    print("EXEMPLE D'ANALYSE DE SINISTRALITÉ AUTOMOBILE")
    print("="*70)
    
    # 1. Chargement des données
    print("\n1. Chargement des données...")
    donnees = dp.charger_toutes_donnees('.')
    
    # 2. Nettoyage et préparation
    print("\n2. Nettoyage et préparation...")
    df_polices = dp.nettoyer_donnees_polices(donnees['polices'])
    df_sinistres = dp.nettoyer_donnees_sinistres(donnees['sinistres'])
    df_sinistres_agg = dp.agregation_sinistres_par_police(df_sinistres)
    
    # 3. Construction de la base
    print("\n3. Construction de la base finale...")
    df_base = fe.joindre_polices_sinistres(df_polices, df_sinistres_agg)
    df_finale = fe.creer_variables_derivees(df_base)
    
    # 4. Analyse descriptive rapide
    print("\n4. Analyse descriptive")
    print("-" * 70)
    print(f"Nombre total de polices: {len(df_finale):,}")
    print(f"Taux de sinistralité: {df_finale['a_sinistre'].mean()*100:.2f}%")
    print(f"Montant moyen des sinistres: {df_finale[df_finale['montant_total']>0]['montant_total'].mean():.2f}€")
    
    # Analyse par usage
    print("\nSinistralité par usage du véhicule:")
    sinistralite_usage = df_finale.groupby('pol_usage')['a_sinistre'].agg(['mean', 'count'])
    sinistralite_usage.columns = ['Taux_sinistralité', 'Nb_polices']
    sinistralite_usage['Taux_sinistralité'] = (sinistralite_usage['Taux_sinistralité'] * 100).round(2)
    print(sinistralite_usage)
    
    # Analyse par âge véhicule
    if 'vh_age_cat' in df_finale.columns:
        print("\nSinistralité par âge du véhicule:")
        sinistralite_age = df_finale.groupby('vh_age_cat')['a_sinistre'].agg(['mean', 'count'])
        sinistralite_age.columns = ['Taux_sinistralité', 'Nb_polices']
        sinistralite_age['Taux_sinistralité'] = (sinistralite_age['Taux_sinistralité'] * 100).round(2)
        print(sinistralite_age)
    
    # 5. Modélisation simple
    print("\n5. Modélisation prédictive de la fréquence")
    print("-" * 70)
    
    # Sélectionner les features
    features = ['pol_bonus', 'pol_duration', 'drv_age1', 'drv_age_lic1',
                'vh_age', 'vh_din', 'vh_value', 'vh_weight']
    features = [f for f in features if f in df_finale.columns]
    
    print(f"Features utilisées: {', '.join(features)}")
    
    # Préparer les données
    X, y, indices = fe.preparer_donnees_modelisation(df_finale, features, target='a_sinistre')
    
    # Diviser les données
    X_train, X_test, y_train, y_test = md.diviser_donnees(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Entraîner plusieurs modèles
    print("\nEntraînement de plusieurs modèles...")
    
    # Logistic Regression
    model_lr = md.entrainer_logistic_regression(X_train, y_train, max_iter=500)
    y_pred_lr = model_lr.predict(X_test)
    y_pred_proba_lr = model_lr.predict_proba(X_test)[:, 1]
    metriques_lr = ev.evaluer_classification(y_test, y_pred_lr, y_pred_proba_lr)
    
    # Random Forest
    model_rf = md.entrainer_random_forest_classifier(X_train, y_train, n_estimators=50, max_depth=10)
    y_pred_rf = model_rf.predict(X_test)
    y_pred_proba_rf = model_rf.predict_proba(X_test)[:, 1]
    metriques_rf = ev.evaluer_classification(y_test, y_pred_rf, y_pred_proba_rf)
    
    # 6. Comparaison des modèles
    print("\n6. Comparaison des modèles")
    print("-" * 70)
    
    resultats = {
        'Logistic Regression': metriques_lr,
        'Random Forest': metriques_rf
    }
    
    df_comp = ev.comparer_modeles(resultats, metrique='roc_auc')
    
    # 7. Feature importance
    print("\n7. Features les plus importantes (Random Forest)")
    print("-" * 70)
    importance = md.extraire_feature_importance(model_rf, features)
    print(importance.head(10).to_string(index=False))
    
    # 8. Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print(f"✓ Analyse terminée avec succès!")
    print(f"✓ Meilleur modèle: {df_comp.iloc[0]['Modèle']} (AUC = {df_comp.iloc[0]['roc_auc']:.4f})")
    print(f"✓ Les features les plus importantes sont liées au bonus-malus,")
    print(f"  à l'âge du conducteur et aux caractéristiques du véhicule.")
    print(f"\nPour une analyse plus approfondie, consultez le notebook:")
    print(f"  notebooks/projet_sinistralite_climat.ipynb")
    print("="*70 + "\n")


if __name__ == "__main__":
    exemple_analyse_simple()
