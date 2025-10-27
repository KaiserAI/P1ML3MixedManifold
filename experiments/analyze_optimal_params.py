import pandas as pd
import ast
import os

# --- CONFIGURACIÓN DE ARCHIVOS Y RUTAS ---

# Obtenemos la ruta del directorio que contiene este script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define las carpetas y nombres de archivos
RESULTS_DIR = "results"
EXP_B_FILENAME = "exp_B_hyperparams_autoencoder.csv"
EXP_C_FILENAME = "exp_C_hyperparams_manifold.csv"

# Construye la ruta absoluta a los archivos
EXP_B_PATH = os.path.join(SCRIPT_DIR, RESULTS_DIR, EXP_B_FILENAME)
EXP_C_PATH = os.path.join(SCRIPT_DIR, RESULTS_DIR, EXP_C_FILENAME)

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
    df['ae_params_dict'] = df['ae_params'].apply(parse_params_string)
    df['manifold_params_dict'] = df['manifold_params'].apply(parse_params_string)
    return df


def find_best_hyperparam_per_class(df, class_name, class_key, param_key, param_names):
    """
    Encuentra la configuración de hiperparámetros óptima (valor/es) para una clase
    específica, basada en el Trustworthiness ponderado.
    """
    df_class = df[df[class_key] == class_name].copy()

    def get_param_key(row):
        """Genera una tupla de parámetros variables para la agrupación."""
        full_params = row[param_key]
        return tuple(full_params.get(name) for name in param_names if name in full_params)

    # Creamos una clave única basada solo en los valores de los hiperparámetros
    df_class['param_values'] = df_class.apply(get_param_key, axis=1)

    weighted_scores = {}

    # Agrupamos por los valores de los hiperparámetros
    for param_values, group_params in df_class.groupby('param_values'):
        weighted_trustworthiness_sum = 0

        # Calculamos el score ponderado sobre los datasets
        for dataset, group_dataset in group_params.groupby('dataset'):
            if dataset in WEIGHTS:
                # Tomamos la Trustworthiness máxima para esa combinación de params en ese dataset
                max_tw = group_dataset['trustworthiness_test'].max()
                weighted_trustworthiness_sum += WEIGHTS[dataset] * max_tw

        weighted_scores[param_values] = weighted_trustworthiness_sum

    # Encontrar la configuración con la puntuación máxima
    if not weighted_scores:
        return None, 0.0

    best_param_values = max(weighted_scores, key=weighted_scores.get)
    best_score = weighted_scores[best_param_values]

    # Reconstruimos el diccionario de parámetros óptimos
    best_params_dict = {name: val for name, val in zip(param_names, best_param_values)}

    return best_params_dict, best_score / TOTAL_WEIGHT


if __name__ == '__main__':
    try:
        # --- 1. Cargar Datos ---
        df_b = load_and_clean_data(EXP_B_PATH)
        df_c = load_and_clean_data(EXP_C_PATH)

        print(f"✅ Archivos cargados correctamente desde: {os.path.join(SCRIPT_DIR, RESULTS_DIR)}")

        # --- 2. Encontrar Mejores AE Hiperparámetros (Exp B) ---

        # LinearsSarseAutoencoder (optimiza lambda_sparse)
        best_sparse_params, sparse_score = find_best_hyperparam_per_class(
            df=df_b,
            class_name='LinearSparseAutoencoder',
            class_key='autoencoder',
            param_key='ae_params_dict',
            param_names=['lambda_sparse']
        )

        # DenoisingSparseAutoencoder (optimiza lambda_sparse y noise_factor)
        best_denoising_params, denoising_score = find_best_hyperparam_per_class(
            df=df_b,
            class_name='DenoisingSparseAutoencoder',
            class_key='autoencoder',
            param_key='ae_params_dict',
            param_names=['lambda_sparse', 'noise_factor']
        )

        # --- 3. Encontrar Mejores Manifold Hiperparámetros (Exp C) ---

        # TSNE (optimiza perplexity)
        best_tsne_params, tsne_score = find_best_hyperparam_per_class(
            df=df_c,
            class_name='TSNE',
            class_key='manifold',
            param_key='manifold_params_dict',
            param_names=['perplexity']
        )

        # LLE (optimiza n_neighbors)
        best_lle_params, lle_score = find_best_hyperparam_per_class(
            df=df_c,
            class_name='LocallyLinearEmbedding',
            class_key='manifold',
            param_key='manifold_params_dict',
            param_names=['n_neighbors']
        )

        # --- 4. Imprimir el Resultado Final para el Experimento A ---

        print("\n" + "=" * 90)
        print("RECOMENDACIÓN DE MEJORES HIPERPARÁMETROS PARA EXPERIMENTO 'A' (Ponderados)")
        print("=" * 90)

        print("MODELOS AUTOENCODER:")
        print(f"  LinearAutoencoder (Base): No tiene hiperparámetros a optimizar (se usa el default: {{}})")

        print(f"  LinearSparseAutoencoder (Score: {sparse_score:.4f}):")
        print(f"    Clase: LinearSparseAutoencoder")
        print(f"    Parámetros: {best_sparse_params}")

        print(f"  DenoisingSparseAutoencoder (Score: {denoising_score:.4f}):")
        print(f"    Clase: DenoisingSparseAutoencoder")
        print(f"    Parámetros: {best_denoising_params}")

        print("\nMODELOS MANIFOLD:")
        print(f"  TSNE (Score: {tsne_score:.4f}):")
        print(f"    Clase: TSNE")
        print(f"    Parámetros: {best_tsne_params}")

        print(f"  LLE (Score: {lle_score:.4f}):")
        print(f"    Clase: LocallyLinearEmbedding")
        print(f"    Parámetros: {best_lle_params}")

        print("\n" + "=" * 90)
        print("DICCIONARIO PYTHON FINAL PARA IMPLEMENTACIÓN DEL EXPERIMENTO A (MEJORES PARÁMETROS):")
        print("=" * 90)

        # Mapeo para el script final
        print("OPTIMAL_PARAMS_FOR_EXP_A = {")
        print(f"    'LinearSparseAutoencoder': {best_sparse_params},")
        print(f"    'DenoisingSparseAutoencoder': {best_denoising_params},")
        print(f"    'TSNE': {best_tsne_params},")
        print(f"    'LLE': {best_lle_params},")
        print("}")

    except FileNotFoundError as e:
        print(f"\nFATAL: {e}")
        print("Asegúrate de que la estructura de directorios sea: [Directorio del script]/results/*.csv")
