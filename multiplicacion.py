import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

casa = np.array([[2,1],
                [2,2],
                [2,3],
                [2,4],
                [2,5],
                [2,6],
                [2,7],
                [2,8],
                [2,9],
                [2,10],
                [2,11],
                [2,12],
                [2,13],           
                [2,14],
                [2,15],
                [2,16],
                [2,17],
                [2,18],
                [2,19],
                [2,20],
                ], dtype=float)

valor_casa = np.array([2, 
                    4, 
                    6, 
                    8, 
                    10,
                    12,
                    14,
                    16,
                    18,
                    20,
                    22,
                    24,
                    26,
                    28,
                    30,
                    32,
                    34,
                    36,
                    38,
                    40], dtype=float)

# Normalización de entradas y etiquetas
scaler_casa = MinMaxScaler()
scaler_valor = MinMaxScaler()

casa_normalizada = scaler_casa.fit_transform(casa)
valor_casa_normalizado = scaler_valor.fit_transform(valor_casa.reshape(-1, 1))

#modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=5, activation='linear', input_shape=(casa.shape[1],)),
    tf.keras.layers.Dense(units=32, activation = 'linear'),
    tf.keras.layers.Dense(units=32, activation = 'linear'),
    tf.keras.layers.Dense(units=32, activation = 'linear'),
    tf.keras.layers.Dense(units=32, activation = 'linear'),
    tf.keras.layers.Dense(units=1, activation='relu')
])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.00009),
    loss='mean_squared_error'
)

# Entrenamiento
historial = modelo.fit(casa_normalizada, valor_casa_normalizado, epochs=280 ,verbose=1)

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
entrada = np.array([[2,22]])
entrada_normalizada = scaler_casa.transform(entrada)

# Predecir valor y desnormalizar la salida
resultado_normalizado = modelo.predict(entrada_normalizada)
resultado = scaler_valor.inverse_transform(resultado_normalizado)

print(f"El resultado es {resultado[0][0]} valor!")
