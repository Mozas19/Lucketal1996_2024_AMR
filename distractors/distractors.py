import csv
import random
import string

# Función para generar una línea de 7 letras aleatorias en mayúscula, excluyendo ciertas letras y sin letras repetidas
def generar_linea_aleatoria():
    letras_permitidas = ''.join(set(string.ascii_uppercase) - set("AEIOUÑQ"))
    linea = ''.join(random.sample(letras_permitidas, 7))
    return linea

# Crear una lista de líneas aleatorias
lines = [generar_linea_aleatoria() for _ in range(1853)]

# Guardar los datos en un archivo CSV
file_path = r"C:\Users\amoza\OneDrive\Escritorio\TFM\Codigos\distractors\distractorlist.csv"
with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Escribir "dislist" en la celda A1
    writer.writerow(["dislist"])
    # Escribir líneas aleatorias en celdas A2 hasta A1854
    for line in lines:
        writer.writerow([line])

print("Archivo CSV creado exitosamente:", file_path)

