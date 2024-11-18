import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Datos de entrada y salida
casa = np.array([[1, 1000000, 30, 10, 1], [1, 1000000, 10, 10, 1], [1, 1000000, 0.5, 20, 1],
                 [1, 900000, 0.6, 24, 1], [2, 1500000, 0.7, 11, 1], [3, 2000000, 2, 11, 1],
                 [1, 1000000, 0.6, 11, 1], [3, 100000, 2, 33, 4], [2, 1400000, 0.8, 60, 3],
                 [1, 3000000, 1, 54, 2], [3, 2000000, 0.3, 70, 1], [1, 1000000, 1, 20, 1],
                 [2, 1000000, 0.9, 65, 2], [3, 2000000, 0.7, 24, 1], [2, 1000000, 1, 45, 1],
                 [1, 4000000, 0.5, 23, 2], [1, 2000000, 0.5, 10, 1], [1, 1500000, 0.7, 10, 1],
                 [1, 5000000, 0.5, 20, 1], [1, 900000, 0.6, 24, 1], [2, 1500000, 0.7, 11, 1],
                 [3, 4000000, 1, 11, 1], [1, 1000000, 0.6, 11, 1], [3, 100000, 1, 23, 4],
                 [2, 1400000, 0.8, 10, 3], [1, 7000000, 1, 46, 2], [3, 2000000, 0.3, 60, 1],
                 [1, 3000000, 1, 20, 1], [2, 4000000, 0.9, 35, 2], [3, 2500000, 0.7, 24, 1],
                 [2, 1400000, 1, 45, 1], [1, 3000000, 0.5, 20, 2], [2, 1000000, 1, 20, 1],
                 [1, 6000000, 1, 1, 1], [1, 1, 1, 1, 1], [1, 2000000, 1, 2, 1]], dtype=float)

valor_casa = np.array([300000000, 100000000, 10000000, 12960000, 23100000, 132000000, 6600000,
                       79200000, 403200000, 324000000, 126000000, 20000000, 234000000, 100800000,
                       90000000, 92000000, 10000000, 10500000, 50000000, 12960000, 23100000,
                       132000000, 6600000, 27600000, 67200000, 644000000, 108000000, 60000000,
                       504000000, 126000000, 126000000, 60000000, 40000000, 6000000, 1, 4000000], dtype=float)

# Normalización
scaler_casa_standard = StandardScaler()
scaler_casa_minmax = MinMaxScaler()
scaler_valor = StandardScaler()

# Escalamos las entradas en dos pasos
casa_scaled_standard = scaler_casa_standard.fit_transform(casa)
casa_normalizada = scaler_casa_minmax.fit_transform(casa_scaled_standard)

# Escalar la salida
valor_casa_normalizado = scaler_valor.fit_transform(valor_casa.reshape(-1, 1))

# Modelo de red neuronal
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=5, activation='relu', input_shape=(casa.shape[1],)),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='relu')
])

# Función para disminuir la tasa de aprendizaje
def ajustar_tasa_de_aprendizaje(epoch):
    if epoch < 600:
        return 0.001
    elif epoch < 900:
        return 0.0005
    elif epoch < 1500:
        return 0.0001
    else:
        return 0.000005
    
# Compilación y entrenamiento
lr_scheduler = LearningRateScheduler(ajustar_tasa_de_aprendizaje)
modelo.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')
historial = modelo.fit(casa_normalizada, valor_casa_normalizado, epochs=2300, verbose=1, callbacks=[lr_scheduler])

# Predicciones en el conjunto de entrenamiento
predicciones_normalizadas = modelo.predict(casa_normalizada)
predicciones = scaler_valor.inverse_transform(predicciones_normalizadas)

# Comparación de predicciones con valores reales
for i in range(len(valor_casa)):
    print(f"Predicción: {predicciones[i][0]:.2f}, Valor Real: {valor_casa[i]:.2f}")

# Gráfica de la pérdida
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])
plt.show()

# Predicción de un nuevo valor
entrada = np.array([[1, 1000000, 0.6, 10, 1]])
entrada_scaled_standard = scaler_casa_standard.transform(entrada)
entrada_normalizada = scaler_casa_minmax.transform(entrada_scaled_standard)

resultado_normalizado = modelo.predict(entrada_normalizada)
resultado = scaler_valor.inverse_transform(resultado_normalizado)

print(f"El resultado es {resultado[0][0]:.2f} valor!")
