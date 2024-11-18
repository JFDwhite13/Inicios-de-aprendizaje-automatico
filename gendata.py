import numpy as np
import pandas as pd

# Número de ejemplos que deseas generar
num_ejemplos = 1000

# Define rangos personalizados para cada número
rango1 = (1, 3)         # Rango para el primer número
rango2 = (1000000, 4000000) # Rango para el segundo número
rango3 = (0.1, 0)          # Rango para el tercer número
rango4 = (10, 100)        # Rango para el cuarto número
rango5 = (1, 3)         # Rango para el quinto número

# Genera cada columna dentro de sus rangos específicos
col1 = np.random.randint(rango1[0], rango1[1] + 1, size=num_ejemplos)
col2 = np.random.randint(rango2[0], rango2[1] + 1, size=num_ejemplos)
col3 = np.random.uniform(rango3[0], rango3[1] + 1, size=num_ejemplos)
col4 = np.random.randint(rango4[0], rango4[1] + 1, size=num_ejemplos)
col5 = np.random.randint(rango5[0], rango5[1] + 1, size=num_ejemplos)

col3 = np.round(col3, 1)
# Combina las columnas en una matriz de datos
datos = np.column_stack((col1, col2, col3, col4, col5))

# Calcula el producto de cada fila
productos = np.prod(datos, axis=1)

# Crea un DataFrame para almacenar los datos y sus productos
df = pd.DataFrame(datos, columns=['Num1', 'Num2', 'Num3', 'Num4', 'Num5'])
df['Producto'] = productos

# Guarda el conjunto de datos en un archivo CSV
df.to_csv('datos_multiplicacion_varios_rangos.csv', index=False)

print("Datos generados y guardados en 'datos_multiplicacion_varios_rangos.csv'")
