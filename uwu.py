# Abrir el archivo en modo lectura
with open('resultados.txt', 'r') as file:
    lineas = file.readlines()

# Agregar "[" al inicio y "]," al final de cada línea
lineas_modificadas = ['[' + linea.strip() + '],' + '\n' for linea in lineas]

# Escribir las líneas modificadas en un nuevo archivo
with open('resultados_modificado.txt', 'w') as file:
    file.writelines(lineas_modificadas)

print("Se ha agregado '[' al inicio y '],' al final de cada línea en 'archivo_modificado.txt'")

