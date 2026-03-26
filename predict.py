import joblib
import os
import argparse
import numpy as np
import pandas as pd
# Importamos las funciones necesarias de tu main.py
from main import process_protein, cluster_points 

def run_prediction(pdb_path, model_path):
    # 1. Configurar carpeta de salida y nombre de archivo
    output_dir = "csv"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Carpeta creada: {output_dir}")

    protein_name = os.path.splitext(os.path.basename(pdb_path))[0]
    # Guardamos la ruta completa: csv/nombre_results.csv
    output_csv = os.path.join(output_dir, f"{protein_name}_results.csv")

    # 2. Cargar el modelo (.pkl)
    if not os.path.exists(model_path):
        print(f"Error: No se encuentra el archivo {model_path}")
        return
    
    model = joblib.load(model_path)
    print(f"--- Modelo cargado: {model_path} ---")

    # 3. Procesar la proteína (Geometría, SAS, Features)
    X, y_true, sas_points = process_protein(pdb_path)

    if X is None:
        return

    # 4. Predecir probabilidades
    probs = model.predict_proba(X)[:, 1]
    
    # 5. Agrupar en bolsillos (Clustering) con umbral 0.3
    pockets = cluster_points(sas_points, probs, threshold=0.3)

    print(f"\nResultados para: {protein_name}")
    print(f"Bolsillos encontrados: {len(pockets)}")

    # 6. Guardar resultados en CSV
    results = []
    for i, p in enumerate(pockets):
        results.append({
            "protein": protein_name,
            "pocket_id": i + 1,
            "center_x": p['center'][0],
            "center_y": p['center'][1],
            "center_z": p['center'][2],
            "size": p['size'],
            "score": p['score']
        })
        if i < 3: 
            print(f" Pocket {i+1}: Score={p['score']:.2f}, Size={p['size']}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"\n[OK] Resultados guardados en: {output_csv}")
    else:
        print("\n[!] No se detectaron bolsillos significativos con el umbral actual.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdb_file", help="Ruta al PDB a predecir")
    parser.add_argument("--model", default="rf_model.pkl", help="Ruta al modelo .pkl")
    args = parser.parse_args()

    run_prediction(args.pdb_file, args.model)