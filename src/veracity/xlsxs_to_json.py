import pandas as pd
import os

# Ruta del archivo Excel
archivo_excel = 'factores_dataset/annotation_agreement/<YOUR ANNOTATED FILENAME>.xlsx'  

# Leer el archivo Excel
df = pd.read_excel(archivo_excel, engine='openpyxl')


df = df.rename(columns={'VERACITY': 'label'})

# Ruta donde quieres guardar el JSON
directorio_salida = 'factores_dataset/'  # puedes usar un path absoluto si prefieres
archivo_json = os.path.join(directorio_salida, 'dev.json')

# Crear el directorio si no existe
os.makedirs(directorio_salida, exist_ok=True)

# Guardar el DataFrame como JSON
df.to_json(archivo_json, orient='records', indent=4)

print(f"Archivo JSON guardado en: {archivo_json}")
