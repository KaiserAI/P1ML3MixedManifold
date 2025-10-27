import pandas as pd
import ast
import os

# --- CONFIGURACIÓN DE ARCHIVOS Y RUTAS ---

# Obtenemos la ruta del directorio que contiene este script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Nombre del archivo de resultados del Experimento A (con HPs óptimos)
RESULTS_FILENAME = "exp_A_autoencoder_vs_manifold_OPTIMAL.csv"  # Asegúrate de que este nombre coincida con tu salida
RESULTS_DIR = "results"
RESULTS_PATH = os.path.join(SCRIPT_DIR, RESULTS_DIR, RESULTS_FILENAME)

# --- SISTEMA DE PESO PARA LA PONDERACIÓN DE DATASETS ---
WEIGHTS = {
    "MNIST": 4,  # Máxima preferencia
    "FashionMNIST": 4,  # Máxima preferencia
    "Cifar10": 2,  # Preferencia media
    "GlassIdentification": 1  # Mínima preferencia
}
TOTAL_WEIGHT = sum(WEIGHTS.values())


# --- FUNCIONES DE UTILIDAD ---

def parse_params_string(param_str):
    """Convierte la cadena de parámetros (ej. "{'key': value}") en un diccionario de Python."""
    try:
        return ast.literal_eval(param_str)
    except (ValueError, SyntaxError):
        return {}


def load_and_clean_data(file_path):
    """Carga los datos y aplica la limpieza de las columnas de parámetros."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: No se encontró el archivo de resultados en la ruta: {file_path}")

    df = pd.read_csv(file_path)

    # Aplicar la conversión a diccionario para las columnas de parámetros
    df['ae_params_dict'] = df['ae_params'].apply(parse_params_string)
    df['manifold_params_dict'] = df['manifold_params'].apply(parse_params_string)

    return df


def get_combination_key(row):
    """Crea una clave única para cada combinación (AE Class + AE Params + Manifold Class + Manifold Params)."""

    # Parámetros relevantes del AE (excluyendo fijos)
    ae_params = row['ae_params_dict']
    ae_optim_params = {
        k: ae_params.get(k) for k in ['lambda_sparse', 'noise_factor'] if k in ae_params
    }

    # Parámetros relevantes del Manifold (excluyendo fijos)
    m_params = row['manifold_params_dict']
    m_optim_params = {
        k: m_params.get(k) for k in ['perplexity', 'n_neighbors'] if k in m_params
    }

    # La clave es (AE_Clase, AE_Params_Tuple, Manifold_Clase, Manifold_Params_Tuple)
    return (
        row['autoencoder'],
        tuple(sorted(ae_optim_params.items())),
        row['manifold'],
        tuple(sorted(m_optim_params.items()))
    )


def find_best_overall_model(df):
    """Calcula el score ponderado para cada combinación y devuelve el mejor modelo."""

    df['combination_key'] = df.apply(get_combination_key, axis=1)

    weighted_scores = {}

    for key, group in df.groupby('combination_key'):
        weighted_trustworthiness_sum = 0

        for dataset, data in group.groupby('dataset'):
            if dataset in WEIGHTS:
                # Usamos la Trustworthiness máxima para esa combinación y ese dataset
                max_tw = data['trustworthiness_test'].max()
                weighted_trustworthiness_sum += WEIGHTS[dataset] * max_tw

        weighted_scores[key] = weighted_trustworthiness_sum

    if not weighted_scores:
        return None, 0.0

    best_key = max(weighted_scores, key=weighted_scores.get)
    best_weighted_score = weighted_scores[best_key] / TOTAL_WEIGHT

    # Extraer y reconstruir los componentes de la mejor combinación
    best_ae_class = best_key[0]
    best_ae_params = dict(best_key[1])
    best_manifold_class = best_key[2]
    best_manifold_params = dict(best_key[3])

    return {
        'ae_class': best_ae_class,
        'ae_params': best_ae_params,
        'manifold_class': best_manifold_class,
        'manifold_params': best_manifold_params,
        'weighted_score': best_weighted_score
    }, best_weighted_score


if __name__ == '__main__':
    try:
        # --- 1. Cargar Datos ---
        df_a = load_and_clean_data(RESULTS_PATH)

        print(f"✅ Archivo de resultados {RESULTS_FILENAME} cargado correctamente.")

        # --- 2. Encontrar el Mejor Modelo Global ---
        best_combination, best_weighted_score = find_best_overall_model(df_a)

        if best_combination:

            # --- 3. Imprimir el Resultado Final ---
            output_recommendation = {
                'AE_CLASS': best_combination['ae_class'],
                'AE_PARAMS': best_combination['ae_params'],
                'MANIFOLD_CLASS': best_combination['manifold_class'],
                'MANIFOLD_PARAMS': best_combination['manifold_params']
            }

            print("\n" + "=" * 80)
            print("RECOMENDACIÓN FINAL DE MODELO (MAXIMIZANDO TRUSTWORTHINESS PONDERADO)")
            print("=" * 80)
            print(f"Puntuación Ponderada Global (Confianza): {best_weighted_score:.4f}")
            print("-" * 80)
            print(f"AE ÓPTIMO:")
            print(f"  Clase: {output_recommendation['AE_CLASS']}")
            print(f"  Parámetros: {output_recommendation['AE_PARAMS']}")
            print(f"Manifold ÓPTIMO:")
            print(f"  Clase: {output_recommendation['MANIFOLD_CLASS']}")
            print(f"  Parámetros: {output_recommendation['MANIFOLD_PARAMS']}")
            print("=" * 80)

            print("\nCÓDIGO PYTHON PARA IMPLEMENTACIÓN DEL MODELO FINAL:")
            print(f"FINAL_MODEL_AE_CLASS = \"{output_recommendation['AE_CLASS']}\"")
            print(f"FINAL_MODEL_AE_PARAMS = {output_recommendation['AE_PARAMS']}")
            print(f"FINAL_MODEL_MANIFOLD_CLASS = \"{output_recommendation['MANIFOLD_CLASS']}\"")
            print(f"FINAL_MODEL_MANIFOLD_PARAMS = {output_recommendation['MANIFOLD_PARAMS']}")

        else:
            print("\n⚠️ No se encontraron combinaciones válidas para el análisis.")

    except FileNotFoundError as e:
        print(f"\nFATAL: {e}")
        print(f"Asegúrate de que el archivo {RESULTS_FILENAME} existe en la carpeta {RESULTS_DIR}.")