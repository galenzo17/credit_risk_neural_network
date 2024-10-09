¡Hola de nuevo! Ahora vamos a llevar nuestro análisis de riesgo crediticio al siguiente nivel utilizando **redes neuronales artificiales (ANNs)**. Las ANNs son modelos más potentes que pueden capturar relaciones no lineales en los datos, lo que puede mejorar la precisión de nuestras predicciones.

## Pasos para Implementar una Red Neuronal para el Análisis de Riesgo Crediticio

1. **Importar las bibliotecas necesarias**.
2. **Cargar y preprocesar los datos**.
3. **Dividir los datos en conjuntos de entrenamiento y prueba**.
4. **Construir y compilar la red neuronal**.
5. **Entrenar el modelo**.
6. **Evaluar el rendimiento del modelo**.
7. **Visualizar el proceso de entrenamiento**.
8. **Hacer predicciones con nuevos datos**.

Ahora, vamos a detallar cada uno de estos pasos y proporcionar el código correspondiente.

### 1. Importar las Bibliotecas Necesarias

Instalamos e importamos las bibliotecas que utilizaremos, incluyendo `tensorflow` y `keras` para construir nuestra red neuronal.

```python
# Instalar las bibliotecas necesarias
!pip install pandas numpy scikit-learn matplotlib tensorflow

# Importar las bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

### 2. Cargar y Preprocesar los Datos

Al igual que antes, cargamos el conjunto de datos y realizamos el preprocesamiento necesario.

```python
def load_data(file_path):
    """
    Cargar el conjunto de datos desde un archivo CSV.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Limpiar y normalizar los datos.
    """
    # Eliminar filas con valores faltantes en la variable objetivo
    df = df.dropna(subset=['loan_status'])
    
    # Convertir la variable objetivo a entero
    df['loan_status'] = df['loan_status'].astype(int)
    
    # Separar características y variable objetivo
    target = df['loan_status']
    features = df.drop('loan_status', axis=1)
    
    # Manejar variables categóricas
    features = pd.get_dummies(features, drop_first=True)
    feature_columns = features.columns
    
    # Imputación de valores faltantes
    features = features.fillna(features.mean())
    
    # Escalado de características
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, target, feature_columns, scaler
```

### 3. Dividir los Datos

Dividimos los datos en conjuntos de entrenamiento y prueba.

```python
def split_data(features, target):
    """
    Dividir los datos en conjuntos de entrenamiento y prueba.
    """
    return train_test_split(features, target, test_size=0.2, random_state=42)
```

### 4. Construir y Compilar la Red Neuronal

Construimos una red neuronal simple utilizando `Keras`.

```python
def build_model(input_dim):
    """
    Construir y compilar el modelo de red neuronal.
    """
    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=input_dim))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Salida para clasificación binaria
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
```

- **Capas ocultas**: Hemos utilizado dos capas ocultas con 16 y 8 neuronas respectivamente.
- **Función de activación**: Usamos `relu` para las capas ocultas y `sigmoid` para la capa de salida.
- **Compilación del modelo**: Utilizamos el optimizador `adam` y la función de pérdida `binary_crossentropy`, adecuada para clasificación binaria.

### 5. Entrenar el Modelo

Entrenamos el modelo y guardamos el historial de entrenamiento para visualizarlo posteriormente.

```python
def train_model(model, X_train, y_train, X_val, y_val):
    """
    Entrenar el modelo de red neuronal.
    """
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val)
    )
    return history
```

- **Epochs**: Número de veces que el modelo verá todo el conjunto de entrenamiento.
- **Batch size**: Número de muestras que el modelo procesará antes de actualizar los pesos.

### 6. Evaluar el Rendimiento del Modelo

Evaluamos el modelo en el conjunto de prueba y mostramos métricas de rendimiento.

```python
def evaluate_model(model, X_test, y_test):
    """
    Evaluar el modelo en el conjunto de prueba.
    """
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print(classification_report(y_test, y_pred))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,5))
    plt.imshow(cm, cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.colorbar()
    plt.show()
```

- **Threshold**: Convertimos las probabilidades en etiquetas binarias utilizando un umbral de 0.5.
- **Métricas**: Mostramos precisión, recall, F1-score y soporte.
- **Matriz de confusión**: Visualizamos la matriz de confusión para entender mejor los errores del modelo.

### 7. Visualizar el Proceso de Entrenamiento

Graficamos la pérdida y la precisión durante el entrenamiento y la validación.

```python
def plot_training_history(history):
    """
    Visualizar la pérdida y precisión durante el entrenamiento.
    """
    # Pérdida
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida durante el Entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    
    # Precisión
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Precisión durante el Entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    
    plt.show()
```

### 8. Hacer Predicciones con Nuevos Datos

Creamos funciones para preprocesar nuevos datos y hacer predicciones.

```python
def preprocess_prospect(prospect_df, feature_columns, scaler):
    """
    Preprocesar los datos del prospecto para que coincidan con las características del modelo.
    """
    # Manejar variables categóricas
    prospect_features = pd.get_dummies(prospect_df, drop_first=True)
    prospect_features = prospect_features.reindex(columns=feature_columns, fill_value=0)
    
    # Escalado de características
    prospect_features_scaled = scaler.transform(prospect_features)
    
    return prospect_features_scaled

def evaluate_prospect(model, prospect, feature_columns, scaler):
    """
    Evaluar un nuevo prospecto utilizando el modelo entrenado.
    """
    prospect_df = pd.DataFrame([prospect])
    prospect_features_scaled = preprocess_prospect(prospect_df, feature_columns, scaler)
    prediction = (model.predict(prospect_features_scaled) > 0.5).astype("int32")
    prediction_proba = model.predict(prospect_features_scaled)
    
    return prediction, prediction_proba
```

### Código Principal

Finalmente, unimos todo en una función `main()` para ejecutar el flujo completo.

```python
def main():
    # Ruta al conjunto de datos
    data_file = 'credit_risk_dataset.csv'
    df = load_data(data_file)
    
    # Preprocesamiento
    features, target, feature_columns, scaler = preprocess_data(df)
    
    # División de datos
    X_train, X_test, y_train, y_test = split_data(features, target)
    
    # Construcción y entrenamiento del modelo
    input_dim = X_train.shape[1]
    model = build_model(input_dim)
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Evaluación del modelo
    evaluate_model(model, X_test, y_test)
    
    # Visualización del entrenamiento
    plot_training_history(history)
    
    # Guardar el modelo y el escalador
    model.save('credit_risk_ann_model.h5')
    joblib.dump((feature_columns, scaler), 'model_utils.joblib')
    
    # Evaluar un nuevo prospecto
    prospect = {
        'person_age': 30,
        'person_income': 60000,
        'person_home_ownership': 'RENT',
        'person_emp_length': 10,
        'loan_intent': 'PERSONAL',
        'loan_grade': 'B',
        'loan_amnt': 10000,
        'loan_int_rate': 10.0,
        'loan_percent_income': 0.15,
        'cb_person_default_on_file': 'N',
        'cb_person_cred_hist_length': 5
    }
    
    prediction, prediction_proba = evaluate_prospect(model, prospect, feature_columns, scaler)
    print(f'Prediction: {prediction[0][0]}')
    print(f'Prediction Probability: {prediction_proba[0][0]}')

main()
```

## Análisis de los Pasos para Probar, Visualizar y Medir la Calidad del Modelo

### Probar el Modelo

- **División de Datos**: Al dividir los datos en conjuntos de entrenamiento y prueba, podemos evaluar el rendimiento del modelo en datos que no ha visto antes.
- **Evaluación en el Conjunto de Prueba**: Utilizamos métricas como precisión, recall y F1-score para evaluar el modelo.

### Visualizar el Proceso de Entrenamiento

- **Gráficas de Pérdida y Precisión**: Al graficar la pérdida y precisión durante el entrenamiento y la validación, podemos detectar problemas como sobreajuste o infraajuste.
  - **Sobreajuste**: Si la pérdida en entrenamiento disminuye pero la pérdida en validación aumenta, es posible que el modelo esté sobreajustando.
  - **Infraajuste**: Si tanto la pérdida en entrenamiento como en validación son altas, el modelo puede ser demasiado simple.

### Medir la Calidad del Modelo

- **Matriz de Confusión**: Nos permite ver cuántos casos el modelo clasificó correctamente y dónde se equivocó.
- **Reporte de Clasificación**: Incluye precisión, recall y F1-score para cada clase.
- **Curva ROC y AUC** (Opcional): Podríamos agregar una curva ROC para evaluar la capacidad del modelo para distinguir entre clases.

## Posibles Mejoras

- **Ajustar Hiperparámetros**: Experimentar con el número de capas, neuronas, funciones de activación y optimizadores.
- **Regularización**: Agregar regularización L1/L2 o dropout para prevenir el sobreajuste.
- **Más Datos**: Si es posible, utilizar más datos para entrenar el modelo.
- **Validación Cruzada**: Utilizar validación cruzada para obtener una estimación más robusta del rendimiento del modelo.

## Conclusión

Implementar una red neuronal para el análisis de riesgo crediticio nos permite capturar relaciones más complejas en los datos, potencialmente mejorando la precisión de nuestras predicciones. Al seguir cuidadosamente los pasos de preprocesamiento, construcción del modelo, entrenamiento y evaluación, podemos desarrollar un modelo sólido y aplicarlo en situaciones reales.

¡Espero que este nuevo enfoque te sea útil en tu viaje de aprendizaje!
