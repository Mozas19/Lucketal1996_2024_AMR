import pandas as pd

# Leer el archivo Excel
archivo_excel = r"C:\Users\amoza\OneDrive\Escritorio\TFM\PsychoPy\V5 Code\pares_palabras.xlsx"
datos_excel = pd.read_excel(archivo_excel)

# Mezclar aleatoriamente las filas
datos_excel = datos_excel.sample(frac=1).reset_index(drop=True)

# Guardar el nuevo DataFrame como un archivo Excel
nombre_archivo_excel = "random_pares_palabras.xlsx"
ruta_guardar = r"C:\Users\amoza\OneDrive\Escritorio\TFM\PsychoPy\V5 Code"
ruta_completa = f"{ruta_guardar}/{nombre_archivo_excel}"
datos_excel.to_excel(ruta_completa, index=False)

print("Archivo Excel guardado exitosamente en la dirección especificada.")