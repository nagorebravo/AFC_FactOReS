import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Carga los archivos
agreement_path = "maldita_dataset/annotation_agreement/"
batch_n = "_2" # or "" for the first batch
df1 = pd.read_excel(agreement_path + "<YOUR ANNOTATED FILENAME HERE>" + batch_n + "_summarized.xlsx", nrows=200)
df2 = pd.read_excel(agreement_path + "<YOUR SECOND ANNOTATED FILENAME HERE>" + batch_n + "_summarized.xlsx", nrows=200)

# Make sure both have the same amount of lines
assert df1.shape[0] == df2.shape[0], "Los archivos deben tener el mismo número de filas"


print("BEFORE: First lines of columns 1 and 2:", df1.head(), df2.head(), len(df1), len(df2))
print("\n")

# Filtrar solo filas con índice par
df1 = df1.iloc[::2].reset_index(drop=True)
df2 = df2.iloc[::2].reset_index(drop=True)

print("AFTER: First lines of columns 1 and 2:", df1.head(), df2.head(), len(df1), len(df2))
print("\n")

# Columnas binarias
binary_columns = [
    "relevance", "critical_what", "critical_who",
    "critical_where", "critical_when", "critical_how", "objectivity"
]

# Columnas categóricas
nominal_columns = ["STANCE", "VERACITY"]

# Todas las columnas
all_columns = binary_columns + nominal_columns

# Resultados
results = []

for col in all_columns:
    print(f"Processing column: {col}")
    if col not in df1.columns or col not in df2.columns:
        continue

    col1 = df1[col]
    col2 = df2[col]
    

    if col in binary_columns:
        # Rellenar NaN con 0 para variables binarias
        col1_filled = col1.fillna(0).astype(int)
        col2_filled = col2.fillna(0).astype(int)
    else:
        # Rellenar NaN con "Unknown" para variables categóricas
        col1_filled = col1.fillna("Unknown")
        col2_filled = col2.fillna("Unknown")

    # Calcular % agreement
    percent_agreement = (col1_filled == col2_filled).mean() * 100

    # Calcular kappa
    try:
        kappa = cohen_kappa_score(col1_filled, col2_filled)
    except ValueError:
        kappa = None  # Si no hay suficientes clases

    results.append({
        "Field": col,
        "Type": "Binary" if col in binary_columns else "Nominal",
        "% Agreement": round(percent_agreement, 2),
        "Cohens Kappa": round(kappa, 3) if kappa is not None else "N/A",
        "Compared rows": len(col1_filled)
    })
    print("Results:", results)

print("Results collected for all columns.")


# Mostrar resultados
results_df = pd.DataFrame(results)
print(results_df)

results_txt_path = agreement_path + "agreement_results" + batch_n + ".txt"
with open(results_txt_path, 'w') as f:
    # Guardar los resultados previos (porcentaje de acuerdo, Kappa, etc.)
    for result in results:
        f.write(f"Field: {result['Field']}\n")
        f.write(f"Type: {result['Type']}\n")
        f.write(f"% Agreement: {result['% Agreement']}\n")
        f.write(f"Cohen's Kappa: {result['Cohens Kappa']}\n")
        f.write(f"Compared rows: {result['Compared rows']}\n")
        f.write("\n")
    
    # Añadir la distribución de VERACITY para df1
    veracity_distribution_df1 = df1['VERACITY'].value_counts(normalize=True).sort_index()
    f.write("\nVERACITY Distribution in df1:\n")
    for category, proportion in veracity_distribution_df1.items():
        f.write(f"{category}: {proportion * 100:.2f}%\n")

    # Añadir la distribución de VERACITY para df2
    veracity_distribution_df2 = df2['VERACITY'].value_counts(normalize=True).sort_index()
    f.write("\nVERACITY Distribution in df2:\n")
    for category, proportion in veracity_distribution_df2.items():
        f.write(f"{category}: {proportion * 100:.2f}%\n")


print(f"Results and distributions saved in {results_txt_path}")
