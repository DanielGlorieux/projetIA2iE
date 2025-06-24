import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Prédiction PE Centrale",
    layout="wide",
    page_icon="🔌"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert > div {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Titre avec style
st.title("🔌 Prédiction de la Production Électrique (PE)")
st.markdown("### Application d'analyse et de prédiction avec Machine Learning")

# Sidebar pour la navigation
st.sidebar.title("🎛️ Configuration")
show_data_analysis = st.sidebar.checkbox("Analyse exploratoire", True)
model_selection = st.sidebar.multiselect(
    "Sélectionner les modèles à entraîner",
    ["Random Forest", "SVR", "Régression Linéaire", "MLP Neural Network"],
    default=["Random Forest", "SVR", "MLP Neural Network"]
)

# Paramètres d'entraînement
st.sidebar.subheader("Paramètres")
test_size = st.sidebar.slider("Taille du jeu de test (%)", 10, 40, 20) / 100
random_state = st.sidebar.number_input("Random State", 0, 100, 42)


# Chargement des données avec gestion d'erreur
@st.cache_data
def load_data():
    try:
        data = pd.read_excel("Folds5x2_pp.xlsx")
        return data, None
    except FileNotFoundError:
        # Générer des données d'exemple si le fichier n'existe pas
        np.random.seed(42)
        n_samples = 1000

        # Variables d'entrée simulées (température, pression, humidité, vitesse du vent)
        temperature = np.random.normal(15, 5, n_samples)  # °C
        pression = np.random.normal(1013, 10, n_samples)  # hPa
        humidite = np.random.uniform(25, 95, n_samples)  # %
        vitesse_vent = np.random.exponential(2, n_samples)  # m/s

        # Production électrique simulée (relation complexe)
        pe = (450 - 0.5 * temperature - 0.02 * pression +
              0.1 * humidite + 2 * vitesse_vent +
              np.random.normal(0, 5, n_samples))

        data = pd.DataFrame({
            'AT': temperature,  # Température ambiante
            'V': vitesse_vent,  # Vitesse du vent
            'AP': pression,  # Pression atmosphérique
            'RH': humidite,  # Humidité relative
            'PE': pe  # Production électrique
        })

        return data, "⚠️ Fichier Excel non trouvé. Utilisation de données simulées pour la démonstration."
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return None, str(e)


# Chargement des données
data, error_msg = load_data()
if error_msg:
    st.warning(error_msg)

if data is not None:
    # Analyse exploratoire
    if show_data_analysis:
        st.subheader("📊 Analyse Exploratoire des Données")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.dataframe(data.head(10), use_container_width=True)

        with col2:
            st.write("**Statistiques descriptives**")
            st.dataframe(data.describe().round(2))

        # Visualisations
        st.subheader("📈 Visualisations")

        # Matrice de corrélation
        fig_corr = px.imshow(
            data.corr(),
            text_auto=True,
            aspect="auto",
            title="Matrice de Corrélation",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # Distribution des variables
        col1, col2 = st.columns(2)

        with col1:
            fig_hist = px.histogram(
                data,
                x='PE',
                nbins=30,
                title="Distribution de la Production Électrique (PE)",
                marginal="box"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            # Scatter plot avec variable cible
            feature_select = st.selectbox(
                "Sélectionner une variable pour le scatter plot",
                options=[col for col in data.columns if col != 'PE']
            )
            fig_scatter = px.scatter(
                data,
                x=feature_select,
                y='PE',
                title=f"Relation entre {feature_select} et PE",
                trendline="ols"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    # Préparation des données
    st.subheader("🔧 Prétraitement des Données")

    # Séparation des features et de la cible
    feature_columns = [col for col in data.columns if col != 'PE']
    X = data[feature_columns].values
    y = data['PE'].values

    # Informations sur les données
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nombre d'échantillons", len(data))
    col2.metric("Nombre de features", X.shape[1])
    col3.metric("Valeurs manquantes", data.isnull().sum().sum())
    col4.metric("Taille du test", f"{int(test_size * 100)}%")

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    st.success(f"✅ Données préparées : {len(X_train)} échantillons d'entraînement, {len(X_test)} de test")


    # Fonction d'entraînement des modèles
    @st.cache_data
    def train_models(models_to_train, X_train, y_train, X_test, random_state):
        results = {}

        if "Random Forest" in models_to_train:
            rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            results["Random Forest"] = y_pred_rf

        if "SVR" in models_to_train:
            svr = SVR(kernel='rbf', C=1.0, gamma='scale')
            svr.fit(X_train, y_train)
            y_pred_svr = svr.predict(X_test)
            results["SVR"] = y_pred_svr

        if "Régression Linéaire" in models_to_train:
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_pred_lr = lr.predict(X_test)
            results["Régression Linéaire"] = y_pred_lr

        if "MLP Neural Network" in models_to_train:
            mlp = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.1
            )
            mlp.fit(X_train, y_train)
            y_pred_mlp = mlp.predict(X_test)
            results["MLP Neural Network"] = y_pred_mlp

        return results


    # Interface d'entraînement
    st.subheader("🤖 Entraînement des Modèles")

    if len(model_selection) == 0:
        st.warning("Veuillez sélectionner au moins un modèle dans la sidebar.")
    else:
        if st.button("🚀 Entraîner les modèles sélectionnés", type="primary"):
            with st.spinner("Entraînement en cours..."):
                predictions = train_models(model_selection, X_train, y_train, X_test, random_state)
                st.success("✅ Entraînement terminé !")

                # Évaluation des modèles
                st.subheader("📊 Résultats d'Évaluation")

                # Tableau comparatif
                results_data = []
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

                for i, (model_name, y_pred) in enumerate(predictions.items()):
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)

                    results_data.append({
                        'Modèle': model_name,
                        'R² Score': round(r2, 4),
                        'RMSE': round(rmse, 2),
                        'MAE': round(mae, 2)
                    })

                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)

                # Graphiques de performance
                col1, col2 = st.columns(2)

                with col1:
                    # Graphique en barres pour R²
                    fig_r2 = px.bar(
                        results_df,
                        x='Modèle',
                        y='R² Score',
                        title="Comparaison R² Score",
                        color='Modèle'
                    )
                    fig_r2.update_layout(showlegend=False)
                    st.plotly_chart(fig_r2, use_container_width=True)

                with col2:
                    # Graphique en barres pour RMSE
                    fig_rmse = px.bar(
                        results_df,
                        x='Modèle',
                        y='RMSE',
                        title="Comparaison RMSE",
                        color='Modèle'
                    )
                    fig_rmse.update_layout(showlegend=False)
                    st.plotly_chart(fig_rmse, use_container_width=True)

                # Graphique de comparaison des prédictions
                st.subheader("📈 Comparaison Visuelle des Prédictions")

                # Limiter à 100 points pour la lisibilité
                n_points = min(100, len(y_test))
                indices = np.arange(n_points)

                fig = go.Figure()

                # Valeurs réelles
                fig.add_trace(go.Scatter(
                    x=indices,
                    y=y_test[:n_points],
                    mode='lines+markers',
                    name='Valeurs Réelles',
                    line=dict(color='black', width=3)
                ))

                # Prédictions de chaque modèle
                for i, (model_name, y_pred) in enumerate(predictions.items()):
                    fig.add_trace(go.Scatter(
                        x=indices,
                        y=y_pred[:n_points],
                        mode='lines',
                        name=f'Prédictions {model_name}',
                        line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                    ))

                fig.update_layout(
                    title=f"Comparaison des Prédictions ({n_points} premiers points)",
                    xaxis_title="Index",
                    yaxis_title="Production Électrique (PE)",
                    hovermode='x unified',
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                # Graphiques de dispersion (prédictions vs réelles)
                st.subheader("🎯 Précision des Prédictions")

                n_models = len(predictions)
                if n_models == 1:
                    cols = [st.container()]
                elif n_models == 2:
                    cols = st.columns(2)
                else:
                    cols = st.columns(min(3, n_models))

                for i, (model_name, y_pred) in enumerate(predictions.items()):
                    with cols[i % len(cols)]:
                        fig_scatter = px.scatter(
                            x=y_test,
                            y=y_pred,
                            title=f"Prédictions vs Réelles - {model_name}",
                            labels={'x': 'Valeurs Réelles', 'y': 'Prédictions'},
                            trendline="ols"
                        )

                        # Ligne de référence y=x
                        min_val = min(y_test.min(), y_pred.min())
                        max_val = max(y_test.max(), y_pred.max())
                        fig_scatter.add_shape(
                            type="line",
                            x0=min_val, y0=min_val,
                            x1=max_val, y1=max_val,
                            line=dict(color="red", width=2, dash="dash"),
                        )

                        st.plotly_chart(fig_scatter, use_container_width=True)

                # Analyse des résidus
                st.subheader("📉 Analyse des Résidus")

                best_model = results_df.loc[results_df['R² Score'].idxmax(), 'Modèle']
                st.info(f"🏆 Meilleur modèle (R² Score) : **{best_model}**")

                best_predictions = predictions[best_model]
                residuals = y_test - best_predictions

                col1, col2 = st.columns(2)

                with col1:
                    fig_residuals = px.scatter(
                        x=best_predictions,
                        y=residuals,
                        title=f"Résidus - {best_model}",
                        labels={'x': 'Prédictions', 'y': 'Résidus'}
                    )
                    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_residuals, use_container_width=True)

                with col2:
                    fig_hist_residuals = px.histogram(
                        x=residuals,
                        nbins=30,
                        title=f"Distribution des Résidus - {best_model}",
                        marginal="box"
                    )
                    st.plotly_chart(fig_hist_residuals, use_container_width=True)

else:
    st.error("Impossible de charger les données. Veuillez vérifier le fichier Excel.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Application développée avec ❤️ using Streamlit et Scikit-learn</p>
    <p><em>Prédiction de Production Électrique - Version sans TensorFlow</em></p>
</div>
""", unsafe_allow_html=True)