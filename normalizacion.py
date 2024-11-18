import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Datos originales de las casas
casa = np.array([[1, 1000000, 30, 10, 1],
                [1, 1000000, 10, 10, 1],
                [1, 1000000, 0.5, 20, 1],
                [0.9, 973470, 0.6, 24, 1],
                [0.9, 907880, 0.7, 11, 1]], dtype=float)

valor_casa = np.array([300000000, 
                    100000000, 
                    1000000, 
                    12616171.2, 
                    6291608.4], dtype=float)

# Crear escaladores
scaler_casa = MinMaxScaler()
scaler_valor = MinMaxScaler()

# Normalizar las entradas y las salidas
casa_normalizada = scaler_casa.fit_transform(casa)
valor_casa_normalizado = scaler_valor.fit_transform(valor_casa.reshape(-1, 1))

# Imprimir datos normalizados
print("Datos de casa normalizados:\n", casa_normalizada)
print("Valores de casa normalizados:\n", valor_casa_normalizado)

# Supongamos que tienes una predicción (esto sería el resultado del modelo)
prediccion_normalizada = np.array([[0.5]])  # Valor de ejemplo normalizado

# Desnormalizar la predicción
prediccion_original = scaler_valor.inverse_transform(prediccion_normalizada)

# Imprimir el valor desnormalizado
print("Valor desnormalizado:\n", prediccion_original)
