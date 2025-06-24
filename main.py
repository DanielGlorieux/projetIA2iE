import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


# Chargement du dataset
dataset = pd.read_excel('Folds5x2_pp.xlsx')

# Séparation des variables explicatives et de la variable cible
X = dataset.iloc[:, :-1].values  # AT, V, AP, RH
y = dataset.iloc[:, -1].values   # PE

# Normalisation
sc = StandardScaler()
X = sc.fit_transform(X)

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Construction du modèle ANN
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu', input_shape=(X_train.shape[1],)))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1))

# Compilation du modèle
ann.compile(optimizer='adam', loss='mean_squared_error')

# Entraînement du modèle
ann.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1)

# Prédiction
y_pred = ann.predict(X_test)

# Affichage comparatif des prédictions et valeurs réelles
# np.set_printoptions(precision=2)
# comparison = np.concatenate((y_pred.reshape(-1,1), y_test.reshape(-1,1)), axis=1)
# print("Prédictions vs Réel (PE):")
# print(comparison)

# Evaluation des performances du modèle
# print("R² score:", r2_score(y_test, y_pred))
# print("MAE:", mean_absolute_error(y_test, y_pred))

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# SVR avec noyau RBF (Radial Basis Function)
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_train)
y_pred_svr = svr_model.predict(X_test)

def evaluate_model(name, y_true, y_pred):
    print(f"\n📊 Résultats pour {name}:")
    print("R² score:", r2_score(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("MAE:", mean_absolute_error(y_true, y_pred))

# Évaluations
evaluate_model("ANN", y_test, y_pred.flatten())
evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("SVR", y_test, y_pred_svr)

# Évaluation visuelle
plt.figure(figsize=(10,6))
plt.plot(y_test[:100], label='Réel', color='blue')
plt.plot(y_pred[:100], label='ANN', linestyle='dashed')
plt.plot(y_pred_rf[:100], label='Random Forest', linestyle='dotted')
plt.plot(y_pred_svr[:100], label='SVR', linestyle='dashdot')
plt.title('Comparaison des prédictions (échantillon de 100 points)')
plt.xlabel('Index')
plt.ylabel('Production Électrique (PE)')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()


