import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration de la page
st.set_page_config(
    page_title="Prédiction PE Centrale",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Classes et structures de données
@dataclass
class ModelResults:
    """Structure pour stocker les résultats d'un modèle"""
    name: str
    predictions: np.ndarray
    r2_score: float
    rmse: float
    mae: float
    training_time: float


@dataclass
class DatasetInfo:
    """Structure pour les informations du dataset"""
    shape: Tuple[int, int]
    features: List[str]
    target: str
    missing_values: int
    description: str


@dataclass
class VariableInfo:
    """Structure pour les informations des variables"""
    name: str
    full_name: str
    description: str
    unit: str
    type: str
    range_info: str


class DataLoader:
    """Classe pour gérer le chargement et la validation des données"""

    @staticmethod
    @st.cache_data
    def load_data(file_path: str = "Folds5x2_pp.xlsx") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Charge les données depuis un fichier Excel avec gestion d'erreurs

        Returns:
            Tuple[DataFrame, error_message]: Les données et un message d'erreur éventuel
        """
        try:
            if Path(file_path).exists():
                data = pd.read_excel(file_path)
                logger.info(f"Données chargées depuis {file_path}: {data.shape}")
                return data, None
            else:
                logger.warning(f"Fichier {file_path} non trouvé, génération de données simulées")
                return DataLoader._generate_synthetic_data(), "Fichier non trouvé, données simulées utilisées"

        except Exception as e:
            logger.error(f"Erreur lors du chargement: {e}")
            return None, f"Erreur: {str(e)}"

    @staticmethod
    def _generate_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
        """Génère des données synthétiques pour la démonstration"""
        np.random.seed(42)

        # Simulation basée sur des relations physiques réalistes
        temperature = np.random.normal(15, 8, n_samples)  # Température ambiante (°C)
        humidity = np.random.uniform(25, 95, n_samples)  # Humidité relative (%)
        pressure = np.random.normal(1013, 15, n_samples)  # Pression atmosphérique (hPa)
        wind_speed = np.random.exponential(3, n_samples)  # Vitesse du vent (m/s)

        # Relation complexe pour la production électrique
        pe = (
                480 - 1.2 * temperature - 0.03 * pressure +
                0.15 * humidity + 3 * wind_speed +
                0.01 * temperature * humidity +
                np.random.normal(0, 8, n_samples)
        )

        return pd.DataFrame({
            'AT': temperature,  # Ambient Temperature
            'V': wind_speed,  # Wind Speed
            'AP': pressure,  # Atmospheric Pressure
            'RH': humidity,  # Relative Humidity
            'PE': pe  # Power Output
        })

    @staticmethod
    def get_dataset_info(data: pd.DataFrame) -> DatasetInfo:
        """Retourne les informations sur le dataset"""
        return DatasetInfo(
            shape=data.shape,
            features=[col for col in data.columns if col != 'PE'],
            target='PE',
            missing_values=data.isnull().sum().sum(),
            description="Dataset de production électrique d'une centrale à cycle combiné"
        )

    @staticmethod
    def get_variables_info() -> List[VariableInfo]:
        """Retourne les informations détaillées sur chaque variable"""
        return [
            VariableInfo(
                name="AT",
                full_name="Ambient Temperature",
                description="Température ambiante mesurée à l'entrée de la turbine. Cette variable influence directement l'efficacité de la centrale car une température plus élevée réduit la densité de l'air et donc la performance.",
                unit="°C (Celsius)",
                type="Variable continue",
                range_info="Typiquement entre 1.81°C et 37.11°C"
            ),
            VariableInfo(
                name="V",
                full_name="Exhaust Vacuum",
                description="Vide d'échappement dans la turbine à vapeur. Un vide plus important améliore l'efficacité de la turbine en réduisant la contre-pression sur l'échappement.",
                unit="cm Hg (centimètres de mercure)",
                type="Variable continue",
                range_info="Typiquement entre 25.36 et 81.56 cm Hg"
            ),
            VariableInfo(
                name="AP",
                full_name="Ambient Pressure",
                description="Pression atmosphérique ambiante. Affecte la densité de l'air entrant dans la turbine à gaz et donc la combustion et la performance globale.",
                unit="mbar (millibars)",
                type="Variable continue",
                range_info="Typiquement entre 992.89 et 1033.30 mbar"
            ),
            VariableInfo(
                name="RH",
                full_name="Relative Humidity",
                description="Humidité relative de l'air ambiant. L'humidité affecte la densité de l'air et peut influencer la performance de la combustion dans la turbine à gaz.",
                unit="% (pourcentage)",
                type="Variable continue",
                range_info="Typiquement entre 25.56% et 100.16%"
            ),
            VariableInfo(
                name="PE",
                full_name="Power Output",
                description="Production électrique nette de la centrale. C'est la variable cible que nous cherchons à prédire en fonction des conditions ambiantes.",
                unit="MW (Mégawatts)",
                type="Variable cible (continue)",
                range_info="Typiquement entre 420.26 et 495.76 MW"
            )
        ]


class DataPreprocessor:
    """Classe pour le prétraitement des données"""

    def __init__(self):
        self.scaler = None
        self.feature_names = None

    def fit_transform(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Applique la normalisation et garde les noms des features"""
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        self.feature_names = feature_names
        return self.scaler.fit_transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Applique la transformation sur de nouvelles données"""
        if self.scaler is None:
            raise ValueError("Le preprocessor doit être fitté avant la transformation")
        return self.scaler.transform(X)

    def get_feature_importance_data(self, model, model_type: str) -> pd.DataFrame:
        """Extrait l'importance des features selon le type de modèle"""
        if model_type == "Random Forest" and hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif model_type == "Linear Regression" and hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            return pd.DataFrame()

        return pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)


class ModelTrainer:
    """Classe pour l'entraînement et l'évaluation des modèles"""

    def __init__(self):
        self.models = {}
        self.results = {}

    @st.cache_data
    def train_models(_self, model_configs: Dict, X_train: np.ndarray,
                     y_train: np.ndarray, X_test: np.ndarray,
                     y_test: np.ndarray) -> Dict[str, ModelResults]:
        """
        Entraîne plusieurs modèles et retourne les résultats

        Args:
            model_configs: Configuration des modèles à entraîner
            X_train, y_train: Données d'entraînement
            X_test, y_test: Données de test

        Returns:
            Dict contenant les résultats pour chaque modèle
        """
        results = {}

        for model_name, config in model_configs.items():
            try:
                start_time = datetime.now()

                # Créer et entraîner le modèle
                model = _self._create_model(model_name, config)
                model.fit(X_train, y_train)

                # Prédictions
                y_pred = model.predict(X_test)

                # Métriques
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)

                training_time = (datetime.now() - start_time).total_seconds()

                results[model_name] = ModelResults(
                    name=model_name,
                    predictions=y_pred,
                    r2_score=r2,
                    rmse=rmse,
                    mae=mae,
                    training_time=training_time
                )

                # Stocker le modèle pour l'analyse
                _self.models[model_name] = model

                logger.info(f"Modèle {model_name} entraîné - R²: {r2:.4f}")

            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement de {model_name}: {e}")
                st.error(f"Erreur avec {model_name}: {str(e)}")

        return results

    def _create_model(self, model_name: str, config: Dict):
        """Factory method pour créer les modèles"""
        if model_name == "Random Forest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**config)
        elif model_name == "SVR":
            from sklearn.svm import SVR
            return SVR(**config)
        elif model_name == "Linear Regression":
            from sklearn.linear_model import LinearRegression
            return LinearRegression(**config)
        elif model_name == "MLP Neural Network":
            from sklearn.neural_network import MLPRegressor
            return MLPRegressor(**config)
        elif model_name == "Gradient Boosting":
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(**config)
        else:
            raise ValueError(f"Modèle {model_name} non supporté")


class Visualizer:
    """Classe pour toutes les visualisations"""

    @staticmethod
    def plot_variables_description(variables_info: List[VariableInfo]) -> None:
        """Affiche la description détaillée des variables"""
        st.subheader("📋 Description des Variables")

        # Créer un DataFrame pour l'affichage
        df_variables = pd.DataFrame([
            {
                "Variable": var.name,
                "Nom Complet": var.full_name,
                "Unité": var.unit,
                "Type": var.type,
                "Plage Typique": var.range_info
            }
            for var in variables_info
        ])

        # Affichage du tableau avec style
        st.dataframe(
            df_variables,
            use_container_width=True,
            hide_index=True
        )

        # Descriptions détaillées dans des expandeurs
        st.subheader("📖 Descriptions Détaillées")

        for var in variables_info:
            with st.expander(f"**{var.name}** - {var.full_name}"):
                st.write(var.description)

                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Unité:** {var.unit}")
                    st.write(f"**Type:** {var.type}")
                with col2:
                    st.write(f"**Plage typique:** {var.range_info}")

    @staticmethod
    def plot_data_overview(data: pd.DataFrame) -> None:
        """Affiche une vue d'ensemble des données"""
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📊 Aperçu des Données")
            st.dataframe(data.head(10), use_container_width=True)

        with col2:
            st.subheader("📈 Statistiques")
            st.dataframe(data.describe().round(2))

    @staticmethod
    def plot_correlation_matrix(data: pd.DataFrame) -> None:
        """Affiche la matrice de corrélation"""
        st.subheader("🔗 Matrice de Corrélation")

        corr_matrix = data.corr()

        # Plotly heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Corrélations entre les variables",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_feature_distributions(data: pd.DataFrame) -> None:
        """Affiche la distribution des features"""
        st.subheader("📊 Distribution des Variables")

        numeric_columns = data.select_dtypes(include=[np.number]).columns
        n_cols = 2
        n_rows = (len(numeric_columns) + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f"Distribution de {col}" for col in numeric_columns],
            vertical_spacing=0.1
        )

        for i, col in enumerate(numeric_columns):
            row = i // n_cols + 1
            col_idx = i % n_cols + 1

            fig.add_trace(
                go.Histogram(x=data[col], name=col, showlegend=False),
                row=row, col=col_idx
            )

        fig.update_layout(height=300 * n_rows, title_text="Distributions des Variables")
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_model_comparison(results: Dict[str, ModelResults]) -> None:
        """Compare les performances des modèles"""
        st.subheader("📊 Comparaison des Modèles")

        # Préparer les données
        metrics_data = []
        for result in results.values():
            metrics_data.append({
                'Modèle': result.name,
                'R² Score': result.r2_score,
                'RMSE': result.rmse,
                'MAE': result.mae,
                'Temps (s)': result.training_time
            })

        df_metrics = pd.DataFrame(metrics_data)

        # Tableau des résultats
        st.dataframe(df_metrics.round(4), use_container_width=True)

        # Graphiques de comparaison
        col1, col2 = st.columns(2)

        with col1:
            fig_r2 = px.bar(
                df_metrics, x='Modèle', y='R² Score',
                title="R² Score par Modèle",
                color='R² Score',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_r2, use_container_width=True)

        with col2:
            fig_rmse = px.bar(
                df_metrics, x='Modèle', y='RMSE',
                title="RMSE par Modèle",
                color='RMSE',
                color_continuous_scale='reds'
            )
            st.plotly_chart(fig_rmse, use_container_width=True)

    @staticmethod
    def plot_predictions_comparison(y_test: np.ndarray, results: Dict[str, ModelResults],
                                    n_points: int = 100) -> None:
        """Compare les prédictions des modèles"""
        st.subheader("📈 Comparaison des Prédictions")

        indices = np.arange(min(n_points, len(y_test)))

        fig = go.Figure()

        # Valeurs réelles
        fig.add_trace(go.Scatter(
            x=indices, y=y_test[:len(indices)],
            mode='lines+markers',
            name='Valeurs Réelles',
            line=dict(color='black', width=3)
        ))

        # Prédictions
        colors = px.colors.qualitative.Set1
        for i, result in enumerate(results.values()):
            fig.add_trace(go.Scatter(
                x=indices, y=result.predictions[:len(indices)],
                mode='lines',
                name=f'{result.name}',
                line=dict(color=colors[i % len(colors)], width=2)
            ))

        fig.update_layout(
            title=f"Comparaison des Prédictions (premiers {len(indices)} points)",
            xaxis_title="Index",
            yaxis_title="Production Électrique (PE)",
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_residuals_analysis(y_test: np.ndarray, best_result: ModelResults) -> None:
        """Analyse des résidus du meilleur modèle"""
        st.subheader("🔍 Analyse des Résidus")

        residuals = y_test - best_result.predictions

        col1, col2 = st.columns(2)

        with col1:
            fig_scatter = px.scatter(
                x=best_result.predictions, y=residuals,
                title=f"Résidus vs Prédictions - {best_result.name}",
                labels={'x': 'Prédictions', 'y': 'Résidus'}
            )
            fig_scatter.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col2:
            fig_hist = px.histogram(
                x=residuals, nbins=30,
                title=f"Distribution des Résidus - {best_result.name}",
                marginal="box"
            )
            st.plotly_chart(fig_hist, use_container_width=True)


class StreamlitApp:
    """Classe principale de l'application Streamlit"""

    def __init__(self):
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()
        self.visualizer = Visualizer()

        # Configuration par défaut des modèles
        self.default_model_configs = {
            "Random Forest": {
                "n_estimators": 100,
                "max_depth": None,
                "random_state": 42,
                "n_jobs": -1
            },
            "SVR": {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale"
            },
            "Linear Regression": {},
            "MLP Neural Network": {
                "hidden_layer_sizes": (100, 50),
                "max_iter": 500,
                "random_state": 42,
                "early_stopping": True,
                "validation_fraction": 0.1
            },
            "Gradient Boosting": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42
            }
        }

    def run(self):
        """Point d'entrée principal de l'application"""
        self._setup_ui()

        # Section Description des Variables (NOUVELLE SECTION)
        self._display_variables_description()

        # Chargement des données
        data, error_msg = self.data_loader.load_data()

        if error_msg:
            st.warning(error_msg)

        if data is not None:
            self._display_data_analysis(data)
            self._display_model_training(data)
        else:
            st.error("Impossible de charger les données")

    def _setup_ui(self):
        """Configuration de l'interface utilisateur"""
        st.title("🔌 Prédiction de Production Électrique")
        st.markdown("### Application ML Professionnelle avec Architecture Améliorée")

        # Description du contexte
        st.markdown("""
        Cette application utilise des données d'une centrale électrique à cycle combiné pour prédire 
        la production d'énergie électrique (PE) en fonction de variables environnementales. 
        Une centrale à cycle combiné utilise à la fois une turbine à gaz et une turbine à vapeur 
        pour maximiser l'efficacité énergétique.
        """)

        # Sidebar
        st.sidebar.title("⚙️ Configuration")

        # Paramètres globaux dans le state
        if 'config' not in st.session_state:
            st.session_state.config = {
                'test_size': 0.2,
                'random_state': 42,
                'selected_models': ["Random Forest", "SVR", "MLP Neural Network"]
            }

    def _display_variables_description(self):
        """Affiche la section de description des variables"""
        st.markdown("---")
        variables_info = self.data_loader.get_variables_info()
        self.visualizer.plot_variables_description(variables_info)
        st.markdown("---")

    def _display_data_analysis(self, data: pd.DataFrame):
        """Affiche l'analyse des données"""
        dataset_info = self.data_loader.get_dataset_info(data)

        # Métriques du dataset
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Échantillons", dataset_info.shape[0])
        col2.metric("Features", len(dataset_info.features))
        col3.metric("Valeurs manquantes", dataset_info.missing_values)
        col4.metric("Cible", dataset_info.target)

        # Visualisations
        with st.expander("🔍 Analyse Exploratoire", expanded=True):
            self.visualizer.plot_data_overview(data)
            self.visualizer.plot_correlation_matrix(data)
            self.visualizer.plot_feature_distributions(data)

    def _display_model_training(self, data: pd.DataFrame):
        """Interface d'entraînement des modèles"""
        st.subheader("🤖 Entraînement des Modèles")

        # Configuration dans la sidebar
        st.sidebar.subheader("Paramètres d'Entraînement")

        test_size = st.sidebar.slider("Taille du test (%)", 10, 40, 20) / 100
        random_state = st.sidebar.number_input("Random State", 0, 100, 42)

        selected_models = st.sidebar.multiselect(
            "Modèles à entraîner",
            list(self.default_model_configs.keys()),
            default=["Random Forest", "SVR", "MLP Neural Network"]
        )

        if not selected_models:
            st.warning("Veuillez sélectionner au moins un modèle")
            return

        # Préparation des données
        X = data.drop('PE', axis=1).values
        y = data['PE'].values

        feature_names = [col for col in data.columns if col != 'PE']
        X_scaled = self.preprocessor.fit_transform(X, feature_names)

        # Division train/test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )

        # Bouton d'entraînement
        if st.button("🚀 Entraîner les Modèles", type="primary"):
            # Configuration des modèles sélectionnés
            selected_configs = {
                name: config for name, config in self.default_model_configs.items()
                if name in selected_models
            }

            # Entraînement
            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner("Entraînement en cours..."):
                results = self.trainer.train_models(
                    selected_configs, X_train, y_train, X_test, y_test
                )
                progress_bar.progress(100)

            if results:
                st.success(f"✅ {len(results)} modèles entraînés avec succès!")

                # Affichage des résultats
                self.visualizer.plot_model_comparison(results)
                self.visualizer.plot_predictions_comparison(y_test, results)

                # Analyse du meilleur modèle
                best_result = max(results.values(), key=lambda x: x.r2_score)
                st.info(f"🏆 Meilleur modèle: **{best_result.name}** (R² = {best_result.r2_score:.4f})")

                self.visualizer.plot_residuals_analysis(y_test, best_result)

                # Sauvegarde dans le state pour réutilisation
                st.session_state.results = results
                st.session_state.best_model = best_result
            else:
                st.error("Aucun modèle n'a pu être entraîné")


# Point d'entrée de l'application
def main():
    """Fonction principale"""
    try:
        app = StreamlitApp()
        app.run()
    except Exception as e:
        st.error(f"Erreur dans l'application: {e}")
        logger.error(f"Erreur application: {e}", exc_info=True)


if __name__ == "__main__":
    main()