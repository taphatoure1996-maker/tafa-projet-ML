# ============================================================================
# SCRIPT GOOGLE COLAB - PROJET SINISTRALITÉ AUTOMOBILE ET CLIMAT
# ============================================================================
# 
# INSTRUCTIONS D'UTILISATION:
# ---------------------------
# 1. Téléchargez les fichiers CSV dans Google Colab:
#    - pg17trainpol.csv
#    - pg17trainclaim.csv  
#    - DataClimatiques.csv
#    - fremuni17.csv
#
# 2. Copiez le code de chaque section "CELLULE X" ci-dessous
# 3. Collez dans une cellule Colab séparée
# 4. Exécutez les cellules dans l'ordre (1, 2, 3, etc.)
#
# Note: Copiez UNIQUEMENT le code entre les lignes de séparation
# ============================================================================


# ============================================================================
# ============================== CELLULE 1 ===================================
# =================== Installation des dépendances ==========================
# ============================================================================
# Copiez tout le code ci-dessous dans la première cellule Colab

!pip install -q pandas numpy scikit-learn matplotlib seaborn xgboost plotly

# ============================================================================


# ============================================================================
# ============================== CELLULE 2 ===================================
# ====================== Import des bibliothèques ============================
# ============================================================================
# Copiez tout le code ci-dessous dans la deuxième cellule Colab

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✓ Bibliothèques importées avec succès!")

# ============================================================================


# ============================================================================
# ============================== CELLULE 3 ===================================
# ================= Fonctions de chargement des données ======================
# ============================================================================
# Copiez tout le code ci-dessous dans la troisième cellule Colab

def charger_polices(chemin='pg17trainpol.csv', sep=';'):
    print("Chargement des polices...")
    df = pd.read_csv(chemin, sep=sep)
    print(f"  - {len(df)} polices chargées")
    return df

def charger_sinistres(chemin='pg17trainclaim.csv', sep=';'):
    print("Chargement des sinistres...")
    df = pd.read_csv(chemin, sep=sep)
    # Nettoyer claim_amount
    df['claim_amount'] = df['claim_amount'].astype(str).str.replace('amount=', '').str.strip()
    df['claim_amount'] = pd.to_numeric(df['claim_amount'], errors='coerce')
    print(f"  - {len(df)} sinistres chargés")
    return df

def charger_climat(chemin='DataClimatiques.csv', sep=';'):
    print("Chargement des données climatiques...")
    df = pd.read_csv(chemin, sep=sep)
    print(f"  - {len(df)} observations climatiques")
    return df

print("✓ Fonctions de chargement définies")

# ============================================================================


# ============================================================================
# ============================== CELLULE 4 ===================================
# ====================== Chargement des données ==============================
# ============================================================================
# Copiez tout le code ci-dessous dans la quatrième cellule Colab

# Charger toutes les données
df_polices = charger_polices()
df_sinistres = charger_sinistres()
df_climat = charger_climat()

print("\n✓ Toutes les données chargées!")

# ============================================================================


# ============================================================================
# ============================== CELLULE 5 ===================================
# ====================== Nettoyage des polices ===============================
# ============================================================================
# Copiez tout le code ci-dessous dans la cinquième cellule Colab

def nettoyer_polices(df):
    df = df.copy()
    
    # Colonnes numériques
    cols_num = ['pol_bonus', 'pol_duration', 'pol_sit_duration',
                'drv_age1', 'drv_age2', 'drv_age_lic1', 'drv_age_lic2',
                'vh_age', 'vh_cyl', 'vh_din', 'vh_value', 'vh_weight']
    
    for col in cols_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Valeurs manquantes conducteur secondaire
    if 'drv_age2' in df.columns:
        df['drv_age2'].fillna(0, inplace=True)
    if 'drv_age_lic2' in df.columns:
        df['drv_age_lic2'].fillna(0, inplace=True)
    
    # Identifiant unique
    df['policy_year_id'] = df['id_policy'] + '_' + df['id_year'].astype(str)
    
    print(f"Polices nettoyées: {len(df)} lignes")
    return df

df_polices_clean = nettoyer_polices(df_polices)
print("✓ Polices nettoyées")

# ============================================================================


# ============================================================================
# ============================== CELLULE 6 ===================================
# ==================== Nettoyage des sinistres ===============================
# ============================================================================
# Copiez tout le code ci-dessous dans la sixième cellule Colab

def nettoyer_sinistres(df):
    df = df.copy()
    
    # Créer identifiant
    df['id_policy'] = df['id_client'] + '-' + df['id_vehicle']
    df['policy_year_id'] = df['id_policy'] + '_' + df['id_year'].astype(str)
    
    # Convertir claim_nb
    df['claim_nb'] = pd.to_numeric(df['claim_nb'], errors='coerce')
    
    # Supprimer sinistres invalides
    df = df[df['claim_amount'].notna()]
    df = df[df['claim_amount'] > 0]
    
    print(f"Sinistres nettoyés: {len(df)} lignes")
    return df

df_sinistres_clean = nettoyer_sinistres(df_sinistres)
print("✓ Sinistres nettoyés")

# ============================================================================


# ============================================================================
# ============================== CELLULE 7 ===================================
# =================== Agrégation des sinistres ===============================
# ============================================================================
# Copiez tout le code ci-dessous dans la septième cellule Colab

def agreger_sinistres(df):
    agg_dict = {
        'claim_nb': 'sum',
        'claim_amount': 'sum',
        'id_claim': 'count'
    }
    
    df_agg = df.groupby('policy_year_id').agg(agg_dict).reset_index()
    df_agg.columns = ['policy_year_id', 'nb_sinistres_total', 'montant_total', 'nb_lignes']
    df_agg['a_sinistre'] = 1
    
    print(f"Sinistres agrégés: {len(df_agg)} polices avec sinistre")
    return df_agg

df_sinistres_agg = agreger_sinistres(df_sinistres_clean)
print("✓ Sinistres agrégés")

# ============================================================================


# ============================================================================
# ============================== CELLULE 8 ===================================
# ================== Jointure polices-sinistres ==============================
# ============================================================================
# Copiez tout le code ci-dessous dans la huitième cellule Colab

def joindre_polices_sinistres(df_polices, df_sinistres_agg):
    df = df_polices.merge(df_sinistres_agg, on='policy_year_id', how='left')
    
    # Compléter les non-sinistres
    df['a_sinistre'] = df['a_sinistre'].fillna(0)
    df['nb_sinistres_total'] = df['nb_sinistres_total'].fillna(0)
    df['montant_total'] = df['montant_total'].fillna(0)
    
    print(f"Base jointe: {len(df)} polices")
    print(f"Taux de sinistralité: {df['a_sinistre'].mean()*100:.2f}%")
    
    return df

df_base = joindre_polices_sinistres(df_polices_clean, df_sinistres_agg)
print("✓ Jointure effectuée")

# ============================================================================


# ============================================================================
# ============================== CELLULE 9 ===================================
# ============== Analyse descriptive - Distribution ==========================
# ============================================================================
# Copiez tout le code ci-dessous dans la neuvième cellule Colab

# Distribution de la sinistralité
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Indicateur binaire
df_base['a_sinistre'].value_counts().plot(kind='bar', ax=axes[0])
axes[0].set_title('Distribution de la Sinistralité (0/1)')
axes[0].set_xlabel('A eu un sinistre')
axes[0].set_ylabel('Nombre de polices')

# Montant des sinistres
df_avec_sinistre = df_base[df_base['montant_total'] > 0]
df_avec_sinistre['montant_total'].hist(bins=50, ax=axes[1])
axes[1].set_title('Distribution du Montant des Sinistres')
axes[1].set_xlabel('Montant (€)')
axes[1].set_ylabel('Fréquence')

plt.tight_layout()
plt.show()

print(f"Taux de sinistralité: {df_base['a_sinistre'].mean()*100:.2f}%")
print(f"Montant moyen: {df_avec_sinistre['montant_total'].mean():.2f}€")

# ============================================================================


# ============================================================================
# ============================== CELLULE 10 ==================================
# ============== Analyse descriptive - Par usage =============================
# ============================================================================
# Copiez tout le code ci-dessous dans la dixième cellule Colab

# Sinistralité par usage
sinistralite_usage = df_base.groupby('pol_usage').agg({
    'a_sinistre': ['mean', 'count'],
    'montant_total': 'mean'
})

print("\nSinistralité par usage du véhicule:")
print(sinistralite_usage)

# Visualisation
fig, ax = plt.subplots(figsize=(10, 6))
df_base.groupby('pol_usage')['a_sinistre'].mean().plot(kind='bar', ax=ax)
ax.set_title('Taux de Sinistralité par Usage')
ax.set_ylabel('Taux')
ax.set_xlabel('Usage')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ============================================================================


# ============================================================================
# ============================== CELLULE 11 ==================================
# ================== Créer variables dérivées ================================
# ============================================================================
# Copiez tout le code ci-dessous dans la onzième cellule Colab

def creer_variables_derivees(df):
    df = df.copy()
    
    # Catégories d'âge véhicule
    if 'vh_age' in df.columns:
        df['vh_age_cat'] = pd.cut(df['vh_age'], 
                                   bins=[0, 2, 5, 10, 20, 100],
                                   labels=['Neuf', 'Recent', 'Moyen', 'Ancien', 'Tres_ancien'])
    
    # Catégories âge conducteur
    if 'drv_age1' in df.columns:
        df['drv_age1_cat'] = pd.cut(df['drv_age1'],
                                     bins=[0, 25, 35, 50, 65, 100],
                                     labels=['Jeune', 'Adulte_jeune', 'Adulte', 'Senior', 'Tres_senior'])
    
    # Indicateur conducteur secondaire
    if 'drv_drv2' in df.columns:
        df['a_conducteur_secondaire'] = (df['drv_drv2'] == 'Yes').astype(int)
    
    # Ratio puissance/poids
    if 'vh_din' in df.columns and 'vh_weight' in df.columns:
        df['ratio_puissance_poids'] = df['vh_din'] / (df['vh_weight'] + 1)
    
    print("Variables dérivées créées")
    return df

df_finale = creer_variables_derivees(df_base)
print("✓ Variables dérivées ajoutées")

# ============================================================================


# ============================================================================
# ============================== CELLULE 12 ==================================
# ============== Analyse par âge véhicule ====================================
# ============================================================================
# Copiez tout le code ci-dessous dans la douzième cellule Colab

if 'vh_age_cat' in df_finale.columns:
    print("\nSinistralité par âge du véhicule:")
    sinistralite_age = df_finale.groupby('vh_age_cat')['a_sinistre'].agg(['mean', 'count'])
    sinistralite_age.columns = ['Taux', 'Nb_polices']
    sinistralite_age['Taux'] = (sinistralite_age['Taux'] * 100).round(2)
    print(sinistralite_age)
    
    # Visualisation
    fig, ax = plt.subplots(figsize=(10, 6))
    df_finale.groupby('vh_age_cat')['a_sinistre'].mean().plot(kind='bar', ax=ax)
    ax.set_title('Taux de Sinistralité par Âge du Véhicule')
    ax.set_ylabel('Taux')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ============================================================================


# ============================================================================
# ============================== CELLULE 13 ==================================
# ================ Préparation pour modélisation =============================
# ============================================================================
# Copiez tout le code ci-dessous dans la treizième cellule Colab

# Sélectionner features
features = ['pol_bonus', 'pol_duration', 'drv_age1', 'drv_age_lic1',
            'vh_age', 'vh_din', 'vh_value', 'vh_weight']

# Filtrer features disponibles
features_disponibles = [f for f in features if f in df_finale.columns]

print(f"Features sélectionnées: {features_disponibles}")

# Préparer X et y
df_model = df_finale[features_disponibles + ['a_sinistre']].dropna()
X = df_model[features_disponibles]
y = df_model['a_sinistre']

print(f"\nDonnées de modélisation: {len(X)} observations")
print(f"Distribution target:")
print(y.value_counts())

# ============================================================================


# ============================================================================
# ============================== CELLULE 14 ==================================
# ==================== Division train/test ===================================
# ============================================================================
# Copiez tout le code ci-dessous dans la quatorzième cellule Colab

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Entraînement: {len(X_train)} observations")
print(f"Test: {len(X_test)} observations")

# ============================================================================


# ============================================================================
# ============================== CELLULE 15 ==================================
# ================ Modèle 1 - Régression Logistique ==========================
# ============================================================================
# Copiez tout le code ci-dessous dans la quinzième cellule Colab

print("\n" + "="*60)
print("RÉGRESSION LOGISTIQUE")
print("="*60)

model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_lr.fit(X_train, y_train)

y_pred_lr = model_lr.predict(X_test)
y_pred_proba_lr = model_lr.predict_proba(X_test)[:, 1]

# Métriques
acc_lr = accuracy_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_pred_proba_lr)

print(f"Accuracy: {acc_lr:.4f}")
print(f"AUC-ROC: {auc_lr:.4f}")

# ============================================================================


# ============================================================================
# ============================== CELLULE 16 ==================================
# =================== Modèle 2 - Random Forest ===============================
# ============================================================================
# Copiez tout le code ci-dessous dans la seizième cellule Colab

print("\n" + "="*60)
print("RANDOM FOREST")
print("="*60)

model_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)
y_pred_proba_rf = model_rf.predict_proba(X_test)[:, 1]

# Métriques
acc_rf = accuracy_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

print(f"Accuracy: {acc_rf:.4f}")
print(f"AUC-ROC: {auc_rf:.4f}")

# ============================================================================


# ============================================================================
# ============================== CELLULE 17 ==================================
# ===================== Modèle 3 - XGBoost ===================================
# ============================================================================
# Copiez tout le code ci-dessous dans la dix-septième cellule Colab

print("\n" + "="*60)
print("XGBOOST")
print("="*60)

model_xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, 
                          random_state=42, n_jobs=-1, eval_metric='logloss')
model_xgb.fit(X_train, y_train)

y_pred_xgb = model_xgb.predict(X_test)
y_pred_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]

# Métriques
acc_xgb = accuracy_score(y_test, y_pred_xgb)
auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)

print(f"Accuracy: {acc_xgb:.4f}")
print(f"AUC-ROC: {auc_xgb:.4f}")

# ============================================================================


# ============================================================================
# ============================== CELLULE 18 ==================================
# ================= Comparaison des modèles ==================================
# ============================================================================
# Copiez tout le code ci-dessous dans la dix-huitième cellule Colab

# Tableau comparatif
resultats = pd.DataFrame({
    'Modèle': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Accuracy': [acc_lr, acc_rf, acc_xgb],
    'AUC-ROC': [auc_lr, auc_rf, auc_xgb]
}).sort_values('AUC-ROC', ascending=False)

print("\n" + "="*60)
print("COMPARAISON DES MODÈLES")
print("="*60)
print(resultats.to_string(index=False))

# Visualisation
fig, ax = plt.subplots(figsize=(10, 6))
resultats.plot(x='Modèle', y='AUC-ROC', kind='bar', ax=ax)
ax.set_title('Comparaison des Modèles - AUC-ROC')
ax.set_ylabel('AUC-ROC')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ============================================================================


# ============================================================================
# ============================== CELLULE 19 ==================================
# ======================= Courbes ROC ========================================
# ============================================================================
# Copiez tout le code ci-dessous dans la dix-neuvième cellule Colab

# Courbes ROC pour tous les modèles
plt.figure(figsize=(10, 8))

for nom, y_proba, color in [('Logistic Regression', y_pred_proba_lr, 'blue'),
                              ('Random Forest', y_pred_proba_rf, 'green'),
                              ('XGBoost', y_pred_proba_xgb, 'red')]:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{nom} (AUC = {auc:.4f})', linewidth=2, color=color)

plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Courbes ROC - Comparaison des Modèles')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================================


# ============================================================================
# ============================== CELLULE 20 ==================================
# =================== Feature Importance =====================================
# ============================================================================
# Copiez tout le code ci-dessous dans la vingtième cellule Colab

# Feature importance pour Random Forest
importance_rf = pd.DataFrame({
    'feature': features_disponibles,
    'importance': model_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*60)
print("FEATURE IMPORTANCE (Random Forest)")
print("="*60)
print(importance_rf.to_string(index=False))

# Visualisation
plt.figure(figsize=(10, 6))
plt.barh(range(len(importance_rf)), importance_rf['importance'])
plt.yticks(range(len(importance_rf)), importance_rf['feature'])
plt.xlabel('Importance')
plt.title('Feature Importance - Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ============================================================================


# ============================================================================
# ============================== CELLULE 21 ==================================
# =================== Matrice de confusion ===================================
# ============================================================================
# Copiez tout le code ci-dessous dans la vingt-et-unième cellule Colab

# Matrice de confusion pour le meilleur modèle
cm = confusion_matrix(y_test, y_pred_xgb)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pas de sinistre', 'Sinistre'],
            yticklabels=['Pas de sinistre', 'Sinistre'])
plt.ylabel('Vraie étiquette')
plt.xlabel('Prédiction')
plt.title('Matrice de Confusion - XGBoost')
plt.tight_layout()
plt.show()

print(f"\nVrais Négatifs: {cm[0,0]}")
print(f"Faux Positifs: {cm[0,1]}")
print(f"Faux Négatifs: {cm[1,0]}")
print(f"Vrais Positifs: {cm[1,1]}")

# ============================================================================


# ============================================================================
# ============================== CELLULE 22 ==================================
# ================ Modélisation de la gravité ================================
# ============================================================================
# Copiez tout le code ci-dessous dans la vingt-deuxième cellule Colab

print("\n" + "="*60)
print("MODÉLISATION DE LA GRAVITÉ")
print("="*60)

# Filtrer seulement les sinistres
df_gravite = df_finale[df_finale['a_sinistre'] == 1].copy()

# Préparer données
df_grav_model = df_gravite[features_disponibles + ['montant_total']].dropna()
X_grav = df_grav_model[features_disponibles]
y_grav = df_grav_model['montant_total']

print(f"Données gravité: {len(X_grav)} observations")
print(f"Montant moyen: {y_grav.mean():.2f}€")

# Division
X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
    X_grav, y_grav, test_size=0.2, random_state=42
)

print(f"Entraînement: {len(X_train_g)} observations")
print(f"Test: {len(X_test_g)} observations")

# ============================================================================


# ============================================================================
# ============================== CELLULE 23 ==================================
# =============== Random Forest Regressor ====================================
# ============================================================================
# Copiez tout le code ci-dessous dans la vingt-troisième cellule Colab

print("\n" + "="*60)
print("RANDOM FOREST REGRESSOR")
print("="*60)

model_rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                      random_state=42, n_jobs=-1)
model_rf_reg.fit(X_train_g, y_train_g)

y_pred_rf_reg = model_rf_reg.predict(X_test_g)

# Métriques
rmse = np.sqrt(mean_squared_error(y_test_g, y_pred_rf_reg))
mae = mean_absolute_error(y_test_g, y_pred_rf_reg)
r2 = r2_score(y_test_g, y_pred_rf_reg)

print(f"RMSE: {rmse:.2f}€")
print(f"MAE: {mae:.2f}€")
print(f"R²: {r2:.4f}")

# ============================================================================


# ============================================================================
# ============================== CELLULE 24 ==================================
# ========== Visualisation prédictions vs réelles ============================
# ============================================================================
# Copiez tout le code ci-dessous dans la vingt-quatrième cellule Colab

# Prédictions vs réelles
plt.figure(figsize=(10, 8))
plt.scatter(y_test_g, y_pred_rf_reg, alpha=0.5, s=10)

# Ligne y=x
min_val = min(y_test_g.min(), y_pred_rf_reg.min())
max_val = max(y_test_g.max(), y_pred_rf_reg.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Prédiction parfaite')

plt.xlabel('Montant Réel (€)')
plt.ylabel('Montant Prédit (€)')
plt.title('Prédictions vs Valeurs Réelles - Gravité')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================================


# ============================================================================
# ============================== CELLULE 25 ==================================
# ====================== Résumé final ========================================
# ============================================================================
# Copiez tout le code ci-dessous dans la vingt-cinquième cellule Colab

print("\n" + "="*70)
print("RÉSUMÉ FINAL DU PROJET")
print("="*70)

print(f"\n1. DONNÉES")
print(f"  - Polices: {len(df_finale):,}")
print(f"  - Taux de sinistralité: {df_finale['a_sinistre'].mean()*100:.2f}%")
print(f"  - Montant moyen: {df_finale[df_finale['montant_total']>0]['montant_total'].mean():.2f}€")

print(f"\n2. MEILLEURS MODÈLES")
print(f"  - Fréquence: XGBoost (AUC = {auc_xgb:.4f})")
print(f"  - Gravité: Random Forest (R² = {r2:.4f})")

print(f"\n3. TOP 5 FEATURES IMPORTANTES")
for i, row in importance_rf.head(5).iterrows():
    print(f"  {i+1}. {row['feature']}: {row['importance']*100:.1f}%")

print("\n" + "="*70)
print("✓ ANALYSE COMPLÈTE TERMINÉE!")
print("="*70)

# ============================================================================
# ========================== FIN DU SCRIPT ===================================
# ============================================================================
