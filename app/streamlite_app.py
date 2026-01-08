import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import io
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Configuration de la page
st.set_page_config(
    page_title="Bank Churn Prediction Dashboard",
    page_icon="🏦",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #3B82F6;
    }
    .drift-warning {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #F59E0B;
    }
    .drift-ok {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #10B981;
    }
    .stButton>button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-header">🏦 Bank Churn Prediction Dashboard</h1>', unsafe_allow_html=True)

# Configuration API
API_BASE_URL = st.sidebar.text_input("API URL", " https://bank-churn.nicehill-253897a0.francecentral.azurecontainerapps.io")

# Vérification de la connexion API
def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# Sidebar
st.sidebar.title("Configuration")
st.sidebar.markdown("---")

if check_api_health():
    st.sidebar.success("✅ API connectée")
else:
    st.sidebar.error("❌ API non disponible")

# Onglets
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Prédiction Client Unique",
    "📈 Prédiction Batch",
    "📉 Détection de Drift",
    "📋 Historique des Prédictions"
])

# ============================================
# ONGLET 1: PRÉDICTION CLIENT UNIQUE
# ============================================

with tab1:
    st.markdown('<h2 class="sub-header">Prédiction pour un client unique</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Informations démographiques")
        credit_score = st.slider("Score de crédit", 300, 850, 650)
        age = st.slider("Âge", 18, 92, 35)
        tenure = st.slider("Ancienneté (années)", 0, 10, 5)
        
        geography = st.selectbox("Pays", ["France", "Germany", "Spain"])
        geography_germany = 1 if geography == "Germany" else 0
        geography_spain = 1 if geography == "Spain" else 0
    
    with col2:
        st.markdown("### Informations financières")
        balance = st.number_input("Solde du compte", 0.0, 500000.0, 10000.0)
        num_of_products = st.selectbox("Nombre de produits", [1, 2, 3, 4])
        has_cr_card = st.checkbox("Possède une carte de crédit", value=True)
        is_active_member = st.checkbox("Membre actif", value=True)
        estimated_salary = st.number_input("Salaire estimé", 0.0, 500000.0, 50000.0)
    
    if st.button("🔮 Prédire le churn", key="predict_single"):
        with st.spinner("Calcul en cours..."):
            # Préparation des données
            features = {
                "CreditScore": credit_score,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_of_products,
                "HasCrCard": int(has_cr_card),
                "IsActiveMember": int(is_active_member),
                "EstimatedSalary": estimated_salary,
                "Geography_Germany": geography_germany,
                "Geography_Spain": geography_spain
            }
            
            try:
                # Appel API
                response = requests.post(
                    f"{API_BASE_URL}/predict",
                    json=features,
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Affichage des résultats
                    st.markdown("### Résultats de la prédiction")
                    
                    # Création de colonnes pour les métriques
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric(
                            label="Probabilité de churn",
                            value=f"{result['churn_probability']*100:.1f}%",
                            delta=f"{result['churn_probability']*100:.1f}%"
                        )
                    
                    with metric_col2:
                        prediction_text = "⚠️ Va quitter" if result['prediction'] == 1 else "✅ Restera"
                        st.metric(
                            label="Prédiction",
                            value=prediction_text
                        )
                    
                    with metric_col3:
                        color_map = {
                            "Low": "green",
                            "Medium": "orange",
                            "High": "red"
                        }
                        st.metric(
                            label="Niveau de risque",
                            value=result['risk_level']
                        )
                    
                    # Barre de progression pour la probabilité
                    st.progress(float(result['churn_probability']))
                    
                    # Recommendations basées sur le risque
                    st.markdown("### Recommandations")
                    
                    if result['risk_level'] == "High":
                        st.warning("""
                        **Actions recommandées :**
                        - Contacter le client immédiatement
                        - Offrir des avantages personnalisés
                        - Programme de fidélisation spécial
                        """)
                    elif result['risk_level'] == "Medium":
                        st.info("""
                        **Actions recommandées :**
                        - Surveillance accrue
                        - Offres de produits complémentaires
                        - Enquête de satisfaction
                        """)
                    else:
                        st.success("""
                        **Actions recommandées :**
                        - Maintenir l'engagement
                        - Proposer des mises à niveau
                        - Programme de parrainage
                        """)
                    
                    # Stocker la prédiction pour l'historique
                    if 'predictions_history' not in st.session_state:
                        st.session_state.predictions_history = []
                    
                    st.session_state.predictions_history.append({
                        **features,
                        **result,
                        "timestamp": datetime.now().isoformat(),
                        "type": "single"
                    })
                    
                else:
                    st.error(f"Erreur API: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Erreur de connexion: {str(e)}")

# ============================================
# ONGLET 2: PRÉDICTION BATCH
# ============================================

with tab2:
    st.markdown('<h2 class="sub-header">Prédiction batch - Tableau de clients</h2>', unsafe_allow_html=True)
    
    # Mode de saisie
    input_mode = st.radio(
        "Mode de saisie :",
        ["📝 Saisie manuelle", "📁 Upload CSV"]
    )
    
    if input_mode == "📝 Saisie manuelle":
        # Configuration du nombre de lignes
        num_clients = st.number_input("Nombre de clients à ajouter", 1, 100, 5)
        
        # Initialisation du dataframe dans session state
        if 'batch_data' not in st.session_state:
            st.session_state.batch_data = pd.DataFrame(columns=[
                'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                'Geography_Germany', 'Geography_Spain'
            ])
        
        # Formulaire pour chaque client
        for i in range(num_clients):
            with st.expander(f"Client {i+1}", expanded=(i == 0)):
                cols = st.columns(5)
                
                with cols[0]:
                    credit_score = st.slider(f"Score crédit C{i+1}", 300, 850, 650, key=f"cs{i}")
                with cols[1]:
                    age = st.slider(f"Âge C{i+1}", 18, 92, 35, key=f"age{i}")
                with cols[2]:
                    tenure = st.slider(f"Ancienneté C{i+1}", 0, 10, 5, key=f"ten{i}")
                with cols[3]:
                    balance = st.number_input(f"Solde C{i+1}", 0.0, 500000.0, 10000.0, key=f"bal{i}")
                with cols[4]:
                    num_products = st.selectbox(f"Produits C{i+1}", [1, 2, 3, 4], key=f"prod{i}")
                
                cols2 = st.columns(4)
                with cols2[0]:
                    has_card = st.checkbox(f"Carte C{i+1}", value=True, key=f"card{i}")
                with cols2[1]:
                    active = st.checkbox(f"Actif C{i+1}", value=True, key=f"act{i}")
                with cols2[2]:
                    salary = st.number_input(f"Salaire C{i+1}", 0.0, 500000.0, 50000.0, key=f"sal{i}")
                with cols2[3]:
                    geography = st.selectbox(f"Pays C{i+1}", ["France", "Germany", "Spain"], key=f"geo{i}")
                    geo_germany = 1 if geography == "Germany" else 0
                    geo_spain = 1 if geography == "Spain" else 0
                
                if st.button(f"➕ Ajouter Client {i+1}", key=f"add{i}"):
                    new_row = {
                        'CreditScore': credit_score,
                        'Age': age,
                        'Tenure': tenure,
                        'Balance': balance,
                        'NumOfProducts': num_products,
                        'HasCrCard': int(has_card),
                        'IsActiveMember': int(active),
                        'EstimatedSalary': salary,
                        'Geography_Germany': geo_germany,
                        'Geography_Spain': geo_spain
                    }
                    st.session_state.batch_data = pd.concat(
                        [st.session_state.batch_data, pd.DataFrame([new_row])],
                        ignore_index=True
                    )
                    st.success(f"Client {i+1} ajouté !")
        
        # Affichage du tableau
        if not st.session_state.batch_data.empty:
            st.markdown("### Clients ajoutés")
            st.dataframe(st.session_state.batch_data, use_container_width=True)
            
            # Bouton pour prédire tout le batch
            if st.button("🚀 Lancer les prédictions batch", key="predict_batch"):
                with st.spinner("Prédiction en cours..."):
                    try:
                        # Conversion en format API
                        clients_list = st.session_state.batch_data.to_dict('records')
                        
                        response = requests.post(
                            f"{API_BASE_URL}/predict/batch",
                            json=clients_list,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            results = response.json()
                            predictions = results['predictions']
                            
                            # Ajout des prédictions au dataframe
                            result_df = st.session_state.batch_data.copy()
                            result_df['Churn_Probability'] = [p['churn_probability'] for p in predictions]
                            result_df['Prediction'] = [p['prediction'] for p in predictions]
                            result_df['Risk'] = [
                                'High' if p['churn_probability'] > 0.7 
                                else 'Medium' if p['churn_probability'] > 0.3 
                                else 'Low' 
                                for p in predictions
                            ]
                            
                            st.markdown("### Résultats des prédictions")
                            st.dataframe(result_df, use_container_width=True)
                            
                            # Statistiques
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Clients à risque",
                                    f"{sum([p['prediction'] for p in predictions])}/{len(predictions)}"
                                )
                            with col2:
                                avg_prob = np.mean([p['churn_probability'] for p in predictions])
                                st.metric("Probabilité moyenne", f"{avg_prob*100:.1f}%")
                            with col3:
                                high_risk = sum([1 for p in predictions if p['churn_probability'] > 0.7])
                                st.metric("Risque élevé", high_risk)
                            
                            # Téléchargement des résultats
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Télécharger les résultats",
                                data=csv,
                                file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                            # Stocker dans l'historique
                            for i, (client, pred) in enumerate(zip(clients_list, predictions)):
                                if 'predictions_history' not in st.session_state:
                                    st.session_state.predictions_history = []
                                
                                st.session_state.predictions_history.append({
                                    **client,
                                    **pred,
                                    "risk_level": 'High' if pred['churn_probability'] > 0.7 
                                                 else 'Medium' if pred['churn_probability'] > 0.3 
                                                 else 'Low',
                                    "timestamp": datetime.now().isoformat(),
                                    "type": "batch"
                                })
                            
                        else:
                            st.error(f"Erreur API: {response.status_code}")
                            
                    except Exception as e:
                        st.error(f"Erreur: {str(e)}")
        
    else:  # Mode upload CSV
        st.markdown("### Upload de fichier CSV")
        uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.markdown("### Aperçu des données")
                st.dataframe(df.head(), use_container_width=True)
                
                # Vérification des colonnes requises
                required_cols = [
                    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                    'Geography_Germany', 'Geography_Spain'
                ]
                
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"Colonnes manquantes: {missing_cols}")
                else:
                    st.success(f"✅ Fichier valide ({len(df)} clients)")
                    
                    if st.button("🚀 Lancer les prédictions sur le fichier"):
                        with st.spinner("Prédiction en cours..."):
                            try:
                                clients_list = df.to_dict('records')
                                
                                response = requests.post(
                                    f"{API_BASE_URL}/predict/batch",
                                    json=clients_list,
                                    timeout=30
                                )
                                
                                if response.status_code == 200:
                                    results = response.json()
                                    predictions = results['predictions']
                                    
                                    # Ajout des prédictions
                                    result_df = df.copy()
                                    result_df['Churn_Probability'] = [p['churn_probability'] for p in predictions]
                                    result_df['Prediction'] = [p['prediction'] for p in predictions]
                                    
                                    st.markdown("### Résultats")
                                    st.dataframe(result_df, use_container_width=True)
                                    
                                    # Graphique des prédictions
                                    fig, ax = plt.subplots(figsize=(10, 4))
                                    result_df['Prediction'].value_counts().plot(
                                        kind='bar',
                                        color=['green', 'red'],
                                        ax=ax
                                    )
                                    ax.set_title('Distribution des prédictions de churn')
                                    ax.set_xlabel('Prédiction (0=Non, 1=Oui)')
                                    ax.set_ylabel('Nombre de clients')
                                    st.pyplot(fig)
                                    
                            except Exception as e:
                                st.error(f"Erreur: {str(e)}")
            except Exception as e:
                st.error(f"Erreur lors de la lecture du fichier: {str(e)}")

# ============================================
# ONGLET 3: DÉTECTION DE DRIFT
# ============================================

with tab3:
    st.markdown('<h2 class="sub-header">Détection de Data Drift</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_prod_file = st.file_uploader(
            "Uploader les données de production (CSV)",
            type=['csv'],
            key="drift_upload"
        )
        
        threshold = st.slider(
            "Seuil de détection (p-value)",
            min_value=0.01,
            max_value=0.10,
            value=0.05,
            step=0.01,
            help="Plus la valeur est basse, plus la détection est stricte"
        )
    
    with col2:
        st.markdown("### Référence")
        st.info("""
        **Données de référence :**
        - Fichier: `data/bank_churn.csv`
        - Contient les données historiques
        """)
    
    if uploaded_prod_file is not None:
        # Sauvegarder le fichier uploadé
        prod_data_path = "../data/production_data.csv"
        
        try:
            # Lire et sauvegarder le fichier
            prod_df = pd.read_csv(uploaded_prod_file)
            prod_df.to_csv(prod_data_path, index=False)
            
            st.success(f"✅ Fichier chargé ({len(prod_df)} lignes, {len(prod_df.columns)} colonnes)")
            
            # Aperçu des données
            with st.expander("Aperçu des données de production"):
                st.dataframe(prod_df.head())
            
            # Bouton pour lancer la détection
            if st.button("🔍 Détecter le drift", key="detect_drift"):
                with st.spinner("Analyse en cours..."):
                    try:
                        # Appel API
                        response = requests.post(
                            f"{API_BASE_URL}/drift/check",
                            params={"threshold": threshold},
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Affichage des résultats
                            st.markdown("### Résultats de détection")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Features analysées",
                                    result['features_analyzed']
                                )
                            
                            with col2:
                                st.metric(
                                    "Features avec drift",
                                    result['features_drifted']
                                )
                            
                            with col3:
                                drift_percent = (result['features_drifted'] / result['features_analyzed']) * 100
                                st.metric(
                                    "Pourcentage de drift",
                                    f"{drift_percent:.1f}%"
                                )
                            
                            # Évaluation du risque
                            if drift_percent > 50:
                                st.error("""
                                ⚠️ **RISQUE ÉLEVÉ DE DRIFT**
                                
                                Plus de 50% des features présentent un drift.
                                Considérations :
                                - Le modèle pourrait être obsolète
                                - Re-entraînement recommandé
                                - Analyse approfondie nécessaire
                                """)
                            elif drift_percent > 20:
                                st.warning("""
                                ⚠️ **RISQUE MODÉRÉ DE DRIFT**
                                
                                Entre 20% et 50% des features présentent un drift.
                                Considérations :
                                - Surveillance accrue recommandée
                                - Vérifier les données d'entrée
                                - Planifier un re-entraînement
                                """)
                            else:
                                st.success("""
                                ✅ **DRIFT FAIBLE**
                                
                                Moins de 20% des features présentent un drift.
                                Le modèle est toujours adapté aux données.
                                """)
                            
                            # Alert manuelle
                            st.markdown("### Alertes")
                            alert_col1, alert_col2 = st.columns(2)
                            
                            with alert_col1:
                                if st.button("🚨 Envoyer alerte de drift", key="send_alert"):
                                    try:
                                        alert_response = requests.post(
                                            f"{API_BASE_URL}/drift/alert",
                                            params={
                                                "message": f"Drift détecté: {result['features_drifted']}/{result['features_analyzed']} features",
                                                "severity": "warning"
                                            }
                                        )
                                        if alert_response.status_code == 200:
                                            st.success("Alerte envoyée avec succès!")
                                    except Exception as e:
                                        st.error(f"Erreur lors de l'envoi de l'alerte: {str(e)}")
                            
                            with alert_col2:
                                if st.button("📧 Notifier l'équipe"):
                                    st.info("""
                                    **Notification envoyée :**
                                    - Équipe Data Science
                                    - Équipe Métier
                                    - Administrateurs
                                    """)
                            
                        else:
                            st.error(f"Erreur API: {response.status_code}")
                            
                    except Exception as e:
                        st.error(f"Erreur lors de la détection: {str(e)}")
        
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier: {str(e)}")
    
    # Section de visualisation
    st.markdown("---")
    st.markdown("### Visualisation du drift")
    
    # Exemple de visualisation statique (à adapter avec vos données réelles)
    if st.checkbox("Afficher l'exemple de visualisation"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Exemple de distributions
        np.random.seed(42)
        ref_data = np.random.normal(100, 15, 1000)
        prod_data = np.random.normal(110, 20, 800)
        
        axes[0, 0].hist(ref_data, alpha=0.5, label='Référence', bins=30)
        axes[0, 0].hist(prod_data, alpha=0.5, label='Production', bins=30)
        axes[0, 0].set_title('Distribution des scores de crédit')
        axes[0, 0].legend()
        
        # Exemple de p-values
        features = ['CreditScore', 'Age', 'Balance', 'Tenure']
        p_values = [0.03, 0.45, 0.12, 0.67]
        
        axes[0, 1].barh(features, p_values, color=['red', 'green', 'orange', 'green'])
        axes[0, 1].axvline(x=threshold, color='blue', linestyle='--', label=f'Seuil ({threshold})')
        axes[0, 1].set_xlabel('P-value')
        axes[0, 1].set_title('P-values par feature')
        axes[0, 1].legend()
        
        # Exemple de heatmap
        drift_matrix = np.random.rand(5, 5)
        sns.heatmap(drift_matrix, annot=True, fmt='.2f', ax=axes[1, 0])
        axes[1, 0].set_title('Matrice de corrélation du drift')
        
        # Exemple de timeline
        dates = pd.date_range('2024-01-01', periods=10, freq='M')
        drift_percentages = np.random.uniform(10, 60, 10)
        
        axes[1, 1].plot(dates, drift_percentages, marker='o')
        axes[1, 1].axhline(y=50, color='red', linestyle='--', label='Seuil critique')
        axes[1, 1].axhline(y=20, color='orange', linestyle='--', label='Seuil avertissement')
        axes[1, 1].set_title('Évolution du drift dans le temps')
        axes[1, 1].set_ylabel('% de features avec drift')
        axes[1, 1].legend()
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)

# ============================================
# ONGLET 4: HISTORIQUE DES PRÉDICTIONS
# ============================================

with tab4:
    st.markdown('<h2 class="sub-header">Historique des prédictions</h2>', unsafe_allow_html=True)
    
    if 'predictions_history' in st.session_state and st.session_state.predictions_history:
        # Convertir en DataFrame
        history_df = pd.DataFrame(st.session_state.predictions_history)
        
        # Statistiques
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_predictions = len(history_df)
            st.metric("Prédictions totales", total_predictions)
        
        with col2:
            churn_count = history_df['prediction'].sum() if 'prediction' in history_df.columns else 0
            st.metric("Prédictions de churn", churn_count)
        
        with col3:
            avg_prob = history_df['churn_probability'].mean() if 'churn_probability' in history_df.columns else 0
            st.metric("Probabilité moyenne", f"{avg_prob*100:.1f}%")
        
        with col4:
            recent_predictions = history_df.tail(10) if len(history_df) >= 10 else history_df
            recent_churn = recent_predictions['prediction'].sum() if 'prediction' in recent_predictions.columns else 0
            st.metric("Churn récent (10 derniers)", recent_churn)
        
        # Filtres
        st.markdown("### Filtres")
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            min_prob = st.slider(
                "Probabilité minimale",
                0.0, 1.0, 0.0, 0.1
            )
        
        with filter_col2:
            prediction_filter = st.selectbox(
                "Type de prédiction",
                ["Toutes", "Churn (1)", "Non-churn (0)"]
            )
        
        with filter_col3:
            risk_filter = st.multiselect(
                "Niveau de risque",
                ["Low", "Medium", "High"],
                default=["Low", "Medium", "High"]
            )
        
        # Application des filtres
        filtered_df = history_df.copy()
        
        if 'churn_probability' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['churn_probability'] >= min_prob]
        
        if prediction_filter == "Churn (1)" and 'prediction' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['prediction'] == 1]
        elif prediction_filter == "Non-churn (0)" and 'prediction' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['prediction'] == 0]
        
        if 'risk_level' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['risk_level'].isin(risk_filter)]
        
        # Affichage des données
        st.markdown(f"### Résultats filtrés ({len(filtered_df)}/{total_predictions})")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Visualisations
        st.markdown("### Visualisations")
        
        if len(filtered_df) > 0:
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                if 'prediction' in filtered_df.columns:
                    fig1, ax1 = plt.subplots(figsize=(8, 4))
                    filtered_df['prediction'].value_counts().plot(
                        kind='pie',
                        autopct='%1.1f%%',
                        colors=['green', 'red'],
                        ax=ax1
                    )
                    ax1.set_title('Distribution des prédictions')
                    ax1.set_ylabel('')
                    st.pyplot(fig1)
            
            with viz_col2:
                if 'churn_probability' in filtered_df.columns:
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    ax2.hist(filtered_df['churn_probability'], bins=20, edgecolor='black')
                    ax2.set_xlabel('Probabilité de churn')
                    ax2.set_ylabel('Nombre de clients')
                    ax2.set_title('Distribution des probabilités')
                    st.pyplot(fig2)
            
            # Graphique temporel
            if 'timestamp' in filtered_df.columns:
                filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])
                filtered_df = filtered_df.sort_values('timestamp')
                
                fig3, ax3 = plt.subplots(figsize=(10, 4))
                ax3.plot(filtered_df['timestamp'], filtered_df['churn_probability'], 
                        marker='o', linestyle='-', alpha=0.5)
                ax3.set_xlabel('Date')
                ax3.set_ylabel('Probabilité de churn')
                ax3.set_title('Évolution des probabilités dans le temps')
                plt.xticks(rotation=45)
                st.pyplot(fig3)
        
        # Options d'export
        st.markdown("### Export des données")
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger CSV",
                data=csv,
                file_name=f"predictions_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with export_col2:
            if st.button("🧹 Effacer l'historique"):
                st.session_state.predictions_history = []
                st.rerun()
    
    else:
        st.info("""
        📝 **Aucun historique disponible**
        
        Les prédictions effectuées dans les onglets précédents 
        apparaîtront ici automatiquement.
        
        Pour commencer :
        1. Allez dans l'onglet "Prédiction Client Unique"
        2. Saisissez les informations d'un client
        3. Cliquez sur "Prédire le churn"
        
        Les résultats seront stockés dans cet historique.
        """)

# ============================================
# PIED DE PAGE
# ============================================

st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**🏦 Bank Churn API**")
    st.markdown("Version 1.0.0")

with footer_col2:
    st.markdown("**🔗 Connecté à**")
    st.markdown(f"`{API_BASE_URL}`")

with footer_col3:
    st.markdown("**📊 Dernière mise à jour**")
    st.markdown(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))