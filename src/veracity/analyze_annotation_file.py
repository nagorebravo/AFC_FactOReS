import pandas as pd
import os

agreement_path = "factores_dataset/annotation_agreement/"


dfs = []

# Recorre los archivos en la carpeta
for file in os.listdir(agreement_path):
    if file.startswith("<YOUR ANNOTATED FILE HERE>") and file.endswith(".xlsx"):
        file_path = os.path.join(agreement_path, file)
        df = pd.read_excel(file_path)

        # Recorta si es un archivo con "_2"
        if "_2" in file:
            print(f"\nArchivo {file} - Filas antes: {len(df)}")
            df = df.iloc[:512]
            print(f"Archivo {file} - Filas después de cortar: {len(df)}")

        # Tomar solo filas pares
        df = df.iloc[::2].reset_index(drop=True)
        print(f"Archivo {file} - Filas después de tomar pares: {len(df)}")

        dfs.append(df)

# Combina todos los DataFrames
combined_df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal de filas combinadas: {len(combined_df)}")


# Guardar el DataFrame combinado en un nuevo archivo .xlsx
combined_xlsx_path = os.path.join(agreement_path, "Nagore_annotation_total.xlsx")
combined_df.to_excel(combined_xlsx_path, index=False)
print(f"Archivo Excel combinado guardado en: {combined_xlsx_path}")


# Ruta del archivo de resultados
results_txt_path = os.path.join(agreement_path, "distribution_<YOUR FILE NAME>.txt")

# Calcula y guarda la distribución
with open(results_txt_path, 'w') as f:
    num_lines = len(combined_df)
    f.write(f"\nAmount of rows analyzed: {num_lines}\n")

    veracity_distribution = combined_df['VERACITY'].value_counts(normalize=True).sort_index()
    f.write("\nVERACITY Distribution in combined dataset:\n")
    for category, proportion in veracity_distribution.items():
        f.write(f"{category}: {proportion * 100:.2f}%\n")

print(f"Results and distributions saved in {results_txt_path}")
