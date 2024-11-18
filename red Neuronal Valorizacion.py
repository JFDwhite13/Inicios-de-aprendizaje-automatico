import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

casa = np.array([[1, 1000000, 1, 10, 1],
                [1, 1000000, 0.2, 100, 1],
                [1, 1000000, 0.5, 20, 1],
                [1, 900000, 0.6, 24, 1],
                [2, 1500000, 0.7, 11, 1],
                [3,2000000,2,11,1],
                [1,1000000,0.6,11,1],
                [3,100000,2,33,4],
                [2,1400000,0.8,60,3],
                [1,3000000,1,54,2],
                [3,2000000,0.3,70,1],
                [1, 1000000, 1, 20, 1],
                [2,1000000,0.9,65,2],
                [3,2000000,0.7,24,1],
                [2,1000000,1,45,1],
                [1,4000000,0.5,23,2],
                [1, 2000000, 0.5, 10, 1],
                [1, 1500000, 0.7, 10, 1],
                [1, 5000000, 0.5, 20, 1],
                [1, 900000, 0.6, 24, 1],
                [2, 1500000, 0.7, 11, 1],
                [3,4000000,1,11,1],
                [1,1000000,0.6,11,1],
                [3,100000,1,23,4],
                [2,1400000,0.8,10,3],
                [1,7000000,1,46,2],
                [3,2000000,0.3,60,1],
                [1, 3000000, 1, 20, 1],
                [2,4000000,0.9,35,2],
                [3,2500000,0.7,24,1],
                [2,1400000,1,45,1],
                [1,3000000,0.5,20,2],
                [2,1000000,1,20,1],
                [1,6000000,1,1,1],
                [1,1,1,1,1],
                [1,2000000,1,2,1],
                [2,2500000,1,2,1],
                [2,3000000,0.3,15,1]], dtype=float)

valor_casa = np.array([10000000, 
                    20000000, 
                    10000000, 
                    12960000, 
                    23100000,
                    132000000,
                    6600000,
                    79200000,
                    403200000,
                    324000000,
                    126000000,
                    20000000,
                    234000000,
                    100800000,
                    90000000,
                    92000000,
                    10000000,
                    10500000,
                    50000000,
                    12960000,
                    23100000,
                    132000000,
                    6600000,
                    27600000,
                    67200000,
                    644000000,
                    108000000,
                    60000000,
                    504000000,
                    126000000,
                    126000000,
                    60000000,
                    40000000,
                    6000000,
                    1,
                    4000000,
                    10000000,
                    27000000], dtype=float)

# Normalización de entradas y etiquetas
scaler_casa = MinMaxScaler()
scaler_valor = MinMaxScaler()

casa_normalizada = scaler_casa.fit_transform(casa)
valor_casa_normalizado = scaler_valor.fit_transform(valor_casa.reshape(-1, 1))

#modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=5, activation='relu', input_shape=(casa.shape[1],)),
    tf.keras.layers.Dense(units=32, activation = 'relu'),
    tf.keras.layers.Dense(units=32, activation = 'relu'),
    tf.keras.layers.Dense(units=32, activation = 'relu'),
    tf.keras.layers.Dropout(0.01),
    tf.keras.layers.Dense(units=32, activation = 'relu'),
    tf.keras.layers.Dense(units=32, activation = 'relu'),
    tf.keras.layers.Dense(units=32, activation = 'relu'),
    tf.keras.layers.Dense(units=32, activation = 'relu'),
    tf.keras.layers.Dense(units=32, activation = 'relu'),
    tf.keras.layers.Dense(units=32, activation = 'relu'),
    tf.keras.layers.Dense(units=32, activation = 'relu'),
    tf.keras.layers.Dense(units=32, activation = 'relu'),
    tf.keras.layers.Dense(units=1, activation='relu')
])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='mean_squared_error'
)

# Función para disminuir la tasa de aprendizaje
def ajustar_tasa_de_aprendizaje(epoch):
    if epoch < 400:
        return 0.01
    elif epoch < 600:
        return 0.001  # tasa de aprendizaje inicial
    elif epoch < 900:
        return 0.0005  # tasa de aprendizaje media
    elif epoch < 1500:
        return 0.0001  # tasa de aprendizaje baja para ajustes finos
    elif epoch < 1800:
        return 0.000005
    else:
        return 0.0000005
# Aplicamos el callback
lr_scheduler = LearningRateScheduler(ajustar_tasa_de_aprendizaje)

# Compilar el modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='mean_squared_error'
)

# Entrenamiento con el ajuste de tasa de aprendizaje
historial = modelo.fit(casa_normalizada, valor_casa_normalizado, epochs=4215, verbose=False, callbacks=[lr_scheduler])


# Predicciones en el conjunto de entrenamiento
predicciones_normalizadas = modelo.predict(casa_normalizada)

# Desnormaliza las predicciones para compararlas con los valores reales
predicciones = scaler_valor.inverse_transform(predicciones_normalizadas)
valores_reales = valor_casa  # Valores originales, no normalizados

# Muestra las predicciones frente a los valores reales
for i in range(len(valores_reales)):
    print(f"Predicción: {predicciones[i][0]:.2f}, Valor Real: {valores_reales[i]:.2f}")

# Gráfica de la pérdida
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])
plt.show()

# Predicción
entrada = np.array([[1,1000000,0.6,10,1]])
entrada2=np.array([[2,3000000,0.3,20,1]])
entrada_normalizada = scaler_casa.transform(entrada)
entrada2_normalizada = scaler_casa.transform(entrada2)

# Predecir valor y desnormalizar la salida
resultado_normalizado = modelo.predict(entrada_normalizada)
resultado = scaler_valor.inverse_transform(resultado_normalizado)

resultado2_normalizado = modelo.predict(entrada2_normalizada)
resultado2 = scaler_valor.inverse_transform(resultado2_normalizado)

print(f"La prediccion es {resultado[0][0]} valor! {entrada[0][0] * entrada[0][1] * entrada[0][2] * entrada[0][3] * entrada[0][4]} ")
print(f"La prediccion es {resultado2[0][0]} valor! {entrada2[0][0] * entrada2[0][1] * entrada2[0][2] * entrada2[0][3] * entrada2[0][4]}")

# Guarda el modelo en un archivo HDF5
modelo.save('mi_modelo.h5')
