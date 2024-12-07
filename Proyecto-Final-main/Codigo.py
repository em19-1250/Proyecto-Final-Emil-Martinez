import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Cargar los datos
train_path = 'train.csv'
test_path = 'test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Separar características y la variable objetivo
X = train.drop(['TARGET(PRICE_IN_LACS)', 'ADDRESS'], axis=1)
y = train['TARGET(PRICE_IN_LACS)']

# Convertir columnas categóricas en variables dummies
X = pd.get_dummies(X, drop_first=True)

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 1. Regresión Lineal
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_val)
mse_linear = mean_squared_error(y_val, y_pred_linear)

# 2. Regresión Logística (requiere transformar el objetivo en categorías)
# Dividir los precios en categorías
bins = [0, 50, 100, np.inf]  # Ajustar según el rango de precios
labels = [0, 1, 2]  # Etiquetas de las categorías
y_train_class = pd.cut(y_train, bins=bins, labels=labels)
y_val_class = pd.cut(y_val, bins=bins, labels=labels)

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train_class)
y_pred_logistic = logistic_model.predict(X_val)

# Evaluación de la Regresión Logística
accuracy_logistic = accuracy_score(y_val_class, y_pred_logistic)
classification_rep = classification_report(y_val_class, y_pred_logistic)

# Imprimir resultados
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
