import gym
import numpy as np
import tensorflow as tf
from collections import deque
import random

# Crear el entorno LunarLander
env = gym.make('LunarLander-v2')

# Parámetros de la red y del entrenamiento
estado_dim = env.observation_space.shape[0]  # Dimensiones del estado
accion_dim = env.action_space.n  # Número de acciones posibles

# Modelo de red neuronal para DQN
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(24, input_dim=estado_dim, activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(accion_dim, activation='linear')  # Salida de las Q-values para cada acción
])

# Compilación del modelo
modelo.compile(optimizer='adam', loss='mse')

# Parámetros de entrenamiento
gamma = 0.99  # Factor de descuento
epsilon = 1.0  # Tasa de exploración inicial
epsilon_min = 0.01  # Tasa de exploración mínima
epsilon_decay = 0.995  # Reducción de la exploración
batch_size = 64  # Tamaño del minibatch
memory = deque(maxlen=2000)  # Memoria de experiencias

# Función para seleccionar la acción (Explorar o Explorar)
def seleccionar_accion(estado):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()  # Explorar
    q_values = modelo.predict(estado)
    return np.argmax(q_values[0])  # Explotar

# Entrenamiento del agente
def entrenar_agente():
    global epsilon
    for episodio in range(1000):
        estado = env.reset()
        estado = np.reshape(estado, [1, estado_dim])
        total_recompensa = 0
        
        for tiempo in range(500):
            accion = seleccionar_accion(estado)
            siguiente_estado, recompensa, hecho, _ = env.step(accion)
            siguiente_estado = np.reshape(siguiente_estado, [1, estado_dim])
            
            # Almacenar la transición en la memoria
            memory.append((estado, accion, recompensa, siguiente_estado, hecho))
            
            estado = siguiente_estado
            total_recompensa += recompensa
            
            # Entrenar el modelo si hay suficientes datos en la memoria
            if len(memory) > batch_size:
                minibatch = random.sample(memory, batch_size)
                for s, a, r, s_next, d in minibatch:
                    target = r
                    if not d:
                        target += gamma * np.amax(modelo.predict(s_next)[0])
                    target_f = modelo.predict(s)
                    target_f[0][a] = target
                    modelo.fit(s, target_f, epochs=1, verbose=0)

            if hecho:
                print(f"Episodio: {episodio + 1}, Recompensa: {total_recompensa}, Epsilon: {epsilon:.2}")
                break
        
        # Reducir la tasa de exploración
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

# Entrenar el agente
entrenar_agente()

# Cerrar el entorno después del entrenamiento
env.close()
