"""
Script de test rapide du projet de sinistralité automobile et climat.

Ce script teste les fonctionnalités principales sans exécuter l'analyse complète.
"""

import sys
sys.path.append('.')

from src import data_preprocessing as dp
from src import feature_engineering as fe
from src import models as md
from src import evaluation as ev


def test_chargement_donnees():
    """Test du chargement des données."""
    print("\n" + "="*60)
    print("TEST 1: Chargement des données")
    print("="*60)
    
    donnees = dp.charger_toutes_donnees('.')
    
    assert len(donnees['polices']) > 0, "Polices vides"
    assert len(donnees['sinistres']) > 0, "Sinistres vides"
    assert len(donnees['climat']) > 0, "Climat vide"
    
    print("✓ Toutes les données chargées avec succès\n")
    return donnees


def test_preprocessing(donnees):
    """Test du prétraitement."""
    print("\n" + "="*60)
    print("TEST 2: Prétraitement")
    print("="*60)
    
    df_polices = dp.nettoyer_donnees_polices(donnees['polices'])
    df_sinistres = dp.nettoyer_donnees_sinistres(donnees['sinistres'])
    df_sinistres_agg = dp.agregation_sinistres_par_police(df_sinistres)
    
    assert len(df_polices) > 0, "Polices nettoyées vides"
    assert len(df_sinistres_agg) > 0, "Sinistres agrégés vides"
    
    print("✓ Prétraitement réussi\n")
    return df_polices, df_sinistres_agg


def test_feature_engineering(df_polices, df_sinistres_agg):
    """Test de la construction des features."""
    print("\n" + "="*60)
    print("TEST 3: Construction des features")
    print("="*60)
    
    df_base = fe.joindre_polices_sinistres(df_polices, df_sinistres_agg)
    df_finale = fe.creer_variables_derivees(df_base)
    
    assert 'a_sinistre' in df_finale.columns, "Variable cible manquante"
    assert df_finale['a_sinistre'].mean() > 0, "Taux de sinistralité nul"
    
    print(f"Taux de sinistralité: {df_finale['a_sinistre'].mean()*100:.2f}%")
    print("✓ Features construites avec succès\n")
    return df_finale


def test_modelisation(df_finale):
    """Test de la modélisation."""
    print("\n" + "="*60)
    print("TEST 4: Modélisation")
    print("="*60)
    
    # Sélectionner des features simples
    features = ['pol_bonus', 'pol_duration', 'drv_age1', 'vh_age', 'vh_din', 'vh_value']
    features = [f for f in features if f in df_finale.columns]
    
    # Préparer les données
    X, y, indices = fe.preparer_donnees_modelisation(df_finale, features, target='a_sinistre')
    
    # Diviser
    X_train, X_test, y_train, y_test = md.diviser_donnees(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Entraîner un modèle simple
    model = md.entrainer_logistic_regression(X_train, y_train, max_iter=500)
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Évaluation
    metriques = ev.evaluer_classification(y_test, y_pred, y_pred_proba)
    ev.afficher_metriques_classification(metriques, "Régression Logistique (Test)")
    
    assert metriques['accuracy'] > 0.5, "Accuracy trop faible"
    assert metriques['roc_auc'] > 0.5, "AUC trop faible"
    
    print("✓ Modélisation réussie\n")
    return metriques


def main():
    """Fonction principale."""
    print("\n" + "="*60)
    print("TESTS DU PROJET DE SINISTRALITÉ AUTOMOBILE")
    print("="*60)
    
    try:
        # Test 1: Chargement
        donnees = test_chargement_donnees()
        
        # Test 2: Prétraitement
        df_polices, df_sinistres_agg = test_preprocessing(donnees)
        
        # Test 3: Feature engineering
        df_finale = test_feature_engineering(df_polices, df_sinistres_agg)
        
        # Test 4: Modélisation
        metriques = test_modelisation(df_finale)
        
        print("\n" + "="*60)
        print("TOUS LES TESTS SONT PASSÉS AVEC SUCCÈS ✓")
        print("="*60)
        print("\nLe projet est prêt à être utilisé!")
        print("Pour une analyse complète, ouvrez le notebook:")
        print("  jupyter notebook notebooks/projet_sinistralite_climat.ipynb")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
