import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import os

# Rutas de los archivos
train_path = r"C:\Users\Emil Martinez\Desktop\Piton-assemble\Proyecto-Final-Emil-Martinez\Proyecto-Final-main\train.csv"
test_path = r"C:\Users\Emil Martinez\Desktop\Piton-assemble\Proyecto-Final-Emil-Martinez\Proyecto-Final-main\test.csv"

# Verificar si los archivos existen
if not os.path.exists(train_path):
    raise FileNotFoundError(f"El archivo {train_path} no existe. Verifica la ruta.")
if not os.path.exists(test_path):
    raise FileNotFoundError(f"El archivo {test_path} no existe. Verifica la ruta.")

# Cargar los datos
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Mostrar columnas para verificar que se cargaron correctamente
print("Columnas en train.csv:", train.columns)
print("Columnas en test.csv:", test.columns)

# Separar características (X) y variable objetivo (y) en el dataset de entrenamiento
X = train.drop(['TARGET(PRICE_IN_LACS)', 'ADDRESS'], axis=1)  # Quitar la variable objetivo y 'ADDRESS'
y = train['TARGET(PRICE_IN_LACS)']

# Convertir columnas categóricas en variables dummies
X = pd.get_dummies(X, drop_first=True)

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modelo 1: Regresión Lineal
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_val)
mse_linear = mean_squared_error(y_val, y_pred_linear)

# Modelo 2: Regresión Logística (requiere transformar el objetivo en categorías)
bins = [0, 50, 100, np.inf]  # Divisiones para categorizar los precios
labels = [0, 1, 2]  # Etiquetas de las categorías
y_train_class = pd.cut(y_train, bins=bins, labels=labels)
y_val_class = pd.cut(y_val, bins=bins, labels=labels)

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train_class)
y_pred_logistic = logistic_model.predict(X_val)

# Evaluación de la Regresión Logística
accuracy_logistic = accuracy_score(y_val_class, y_pred_logistic)
classification_rep = classification_report(y_val_class, y_pred_logistic)

# Resultados
print(f'Linear Regression MSE: {mse_linear}')
print(f'Logistic Regression Accuracy: {accuracy_logistic}')
print('\nClassification Report for Logistic Regression:\n')
print(classification_rep)

# Graficar resultados
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))

# Regresión Lineal: Predicciones vs Reales
plt.subplot(1, 2, 1)
plt.scatter(y_val, y_pred_linear, alpha=0.3)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=3)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Linear Regression: Actual vs Predicted')

# Regresión Logística: Matriz de confusión
from sklearn.metrics import ConfusionMatrixDisplay
plt.subplot(1, 2, 2)
ConfusionMatrixDisplay.from_predictions(y_val_class, y_pred_logistic, ax=plt.gca())
plt.title('Logistic Regression Confusion Matrix')

plt.tight_layout()
plt.show()
