import pandas as pd

# Leer el archivo Excel
archivo_excel = r"C:\Users\amoza\OneDrive\Escritorio\TFM\PsychoPy\V5 Code\pares_palabras.xlsx"
datos_excel = pd.read_excel(archivo_excel)

# Crear una lista vacía para almacenar los datos modificados
datos_modificados = []

# Iterar sobre cada fila y modificar las palabras
for indice_fila, fila in datos_excel.iterrows():
    palabras_fila = []
    for valor_celda in fila:
        palabra = str(valor_celda)
        while len(palabra) < 7:
            if len(palabra) % 2 == 0:
                palabra = palabra + 'X'  # Añadir la 'X' al final
            else:
                palabra = 'X' * ((7 - len(palabra)) // 2) + palabra + 'X' * ((7 - len(palabra)) // 2)
        palabras_fila.append(palabra)
    datos_modificados.append(palabras_fila)

# Crear un nuevo DataFrame a partir de los datos modificados y restablecer el índice
df_nuevo = pd.DataFrame(datos_modificados)
df_nuevo.reset_index(drop=True, inplace=True)

# Guardar el nuevo DataFrame como un archivo Excel
nombre_archivo_excel = "pares_palabras_x.xlsx"
ruta_guardar = r"C:\Users\amoza\OneDrive\Escritorio\TFM\PsychoPy\V5 Code"
ruta_completa = f"{ruta_guardar}/{nombre_archivo_excel}"
df_nuevo.to_excel(ruta_completa, index=False)

print("Archivo Excel guardado exitosamente en la dirección especificada.")