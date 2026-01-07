import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score,
    f1_score, 
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuration MLflow
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("bank-churn-prediction")

print("="*60)
print("CHARGEMENT ET PREPROCESSING DES DONNEES")
print("="*60)

df = pd.read_csv("data/bank_churn.csv")

print(f"Dataset : {len(df)} lignes, {len(df.columns)} colonnes")
print(f"Taux de churn : {df['Exited'].mean():.2%}")

# ANALYSE ET FEATURE ENGINEERING
print("\n🔍 Feature engineering...")

# 1. Création de nouvelles features pertinentes
df['Balance_to_Salary_Ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
df['Products_per_Tenure'] = df['NumOfProducts'] / (df['Tenure'] + 1)
df['CreditScore_Age_Interaction'] = df['CreditScore'] * df['Age'] / 1000
df['Is_High_Value'] = ((df['Balance'] > df['Balance'].median()) & 
                       (df['EstimatedSalary'] > df['EstimatedSalary'].median())).astype(int)

# 2. Catégorisation d'âge (plus informatif que l'âge brut)
df['Age_Group'] = pd.cut(df['Age'], 
                         bins=[0, 25, 35, 45, 55, 65, 100],
                         labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+'])

# Encodage one-hot pour Age_Group
age_dummies = pd.get_dummies(df['Age_Group'], prefix='Age', drop_first=True)
df = pd.concat([df, age_dummies], axis=1)

# Séparation features/target
X = df.drop(['Exited', 'Age_Group'], axis=1)
y = df['Exited']

# GÉRER LE DÉSÉQUILIBRE DE CLASSES AVEC SMOTE
print(f"\n📊 Distribution avant SMOTE: {np.bincount(y)}")
print(f"Ratio classe minoritaire: {np.bincount(y)[1]/len(y):.2%}")

smote = SMOTE(random_state=42, sampling_strategy=0.5)  # 50% de churneurs
X_resampled, y_resampled = smote.fit_resample(X, y)
print(f"📊 Distribution après SMOTE: {np.bincount(y_resampled)}")

# Split train/test avec stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

print(f"\n📈 Train : {len(X_train)} lignes")
print(f"📉 Test  : {len(X_test)} lignes")

# NORMALISATION des features numériques
scaler = StandardScaler()
numerical_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 
                  'Balance_to_Salary_Ratio', 'CreditScore_Age_Interaction']
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# OPTIMISATION HYPERPARAMÈTRES AVEC GRID SEARCH
print("\n" + "="*60)
print("OPTIMISATION DES HYPERPARAMÈTRES (GridSearchCV)")
print("="*60)

with mlflow.start_run(run_name=f"optimized-rf-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
    
    # Définition du modèle et des hyperparamètres à tester
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    # Validation croisée stratifiée
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # GridSearch avec scoring sur ROC AUC (meilleur pour déséquilibre)
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    print("🔍 Recherche des meilleurs hyperparamètres...")
    grid_search.fit(X_train, y_train)
    
    # Meilleur modèle
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"✅ Meilleurs paramètres trouvés:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    
    # PRÉDICTIONS AVEC LE MEILLEUR MODÈLE
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    # CALCUL DES MÉTRIQUES DÉTAILLÉES
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    # Métriques additionnelles
    from sklearn.metrics import balanced_accuracy_score, average_precision_score
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    avg_precision = average_precision_score(y_test, y_proba)
    
    # LOG DANS MLFLOW
    mlflow.log_params(best_params)
    mlflow.log_params({
        'feature_engineering': 'advanced',
        'smote_ratio': 0.5,
        'scaler': 'standard'
    })
    
    mlflow.log_metrics({
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc,
        "average_precision": avg_precision
    })
    
    # VISUALISATIONS AMÉLIORÉES
    
    # 1. Matrice de confusion détaillée
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Churn', 'Churn'],
                yticklabels=['Non-Churn', 'Churn'])
    plt.title('Matrice de Confusion - Modèle Optimisé', fontsize=14)
    plt.ylabel('Vraie Classe', fontsize=12)
    plt.xlabel('Classe Prédite', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix_optimized.png', dpi=300)
    mlflow.log_artifact('confusion_matrix_optimized.png')
    plt.close()
    
    # 2. Feature importance avec seuil
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Garder seulement les features importantes (>1% importance)
    significant_features = feature_importance[feature_importance['importance'] > 0.01]
    
    plt.figure(figsize=(12, 8))
    colors = ['darkred' if 'engineered' in feat else 'steelblue' 
              for feat in significant_features['feature']]
    plt.barh(significant_features['feature'], significant_features['importance'], color=colors)
    plt.xlabel('Importance', fontsize=12)
    plt.title('Top Features Importance (seuil > 1%)', fontsize=14)
    plt.axvline(x=0.01, color='red', linestyle='--', alpha=0.5, label='Seuil 1%')
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_importance_optimized.png', dpi=300)
    mlflow.log_artifact('feature_importance_optimized.png')
    plt.close()
    
    # 3. Courbe ROC
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Courbe ROC', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300)
    mlflow.log_artifact('roc_curve.png')
    plt.close()
    
    # ENREGISTREMENT DU MODÈLE
    mlflow.sklearn.log_model(
        best_model,
        "optimized_model",
        registered_model_name="bank-churn-classifier-optimized"
    )
    
    # Sauvegarde locale
    joblib.dump(best_model, "model/churn_model_optimized.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    
    # Sauvegarde des features importantes
    feature_importance.to_csv("model/feature_importance.csv", index=False)
    
    # TAGS et métadonnées
    mlflow.set_tags({
        "environment": "production",
        "model_type": "RandomForest_Optimized",
        "task": "binary_classification",
        "optimization": "grid_search",
        "feature_engineering": "advanced"
    })
    
    # AFFICHAGE DES RÉSULTATS DÉTAILLÉS
    print("\n" + "="*60)
    print("📊 RÉSULTATS DU MODÈLE OPTIMISÉ")
    print("="*60)
    print(f"🎯 Accuracy           : {accuracy:.4f}")
    print(f"⚖️  Balanced Accuracy  : {balanced_acc:.4f}")
    print(f"📐 Precision          : {precision:.4f}")
    print(f"📈 Recall             : {recall:.4f}")
    print(f"📊 F1 Score           : {f1:.4f}")
    print(f"📉 ROC AUC            : {auc:.4f}")
    print(f"🎯 Average Precision  : {avg_precision:.4f}")
    print("="*60)
    
    print(f"\n📋 Rapport de classification:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Non-Churn', 'Churn']))
    
    print(f"\n🏆 Top 5 features les plus importantes:")
    for i, row in feature_importance.head().iterrows():
        print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    print(f"\n💾 Modèle sauvegardé dans : model/churn_model_optimized.pkl")
    print(f"📊 MLflow UI : mlflow ui --port 5000")
    print("="*60)