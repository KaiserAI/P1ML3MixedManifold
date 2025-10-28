import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any

from sklearn.manifold import TSNE, LocallyLinearEmbedding as LLE, trustworthiness
# Importamos las funciones de utilidad
from experiment_utils import load_data, load_labels, save_results_to_csv
from dataset_config import DATASETS, BASE_DATA_PATH

# Añadir el directorio raíz al path para asegurar que los imports internos funcionen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- CONFIGURACIÓN DE EJECUCIÓN ESPECÍFICA DEL EXPERIMENTO D ---
EXPERIMENT_NAME = "Exp_D_Manifold_Only"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUTPUT_FILE = os.path.join(RESULTS_DIR, f"{EXPERIMENT_NAME}.csv")
PLOTS_DIR = os.path.join(os.path.dirname(__file__),
                         f"plots/{EXPERIMENT_NAME}")  # Directorio específico de plots
EMBEDDING_DIM = 2

# Parámetros fijos para la comparación directa (sin búsqueda de hiperparámetros)
TSNE_FIXED_PARAMS = {"n_components": EMBEDDING_DIM, "perplexity": 30, "init": 'pca', "learning_rate": 'auto',
                     "random_state": 42, "n_jobs": -1}
LLE_FIXED_PARAMS = {"n_components": EMBEDDING_DIM, "n_neighbors": 30, "method": 'standard', "random_state": 42,
                    "n_jobs": -1}

# Pasar la configuración consolidada a la función load_data/load_labels
DATASETS_CONFIG = {"BASE_DATA_PATH": BASE_DATA_PATH, "DATASETS": DATASETS}


# --- FUNCIÓN DE UTILIDAD PARA PLOTEAR (Copiada y adaptada de Exp C) ---
def plot_2d_representation(embedding_2d: np.ndarray, labels: np.ndarray, dataset_name: str,
                           autoencoder_name: str, ae_params: dict, manifold_name: str,
                           manifold_params: dict, plot_dir: str):
    """Genera y guarda un gráfico de dispersión (scatter plot) 2D con el formato de Exp C."""

    # Crea el subdirectorio para el dataset si no existe: plots/exp_D/<DatasetName>/
    dataset_plot_dir = os.path.join(plot_dir, dataset_name)
    os.makedirs(dataset_plot_dir, exist_ok=True)

    # 1. Título y nombre del archivo
    manifold_param_str = ", ".join(
        [f"{k}:{v}" for k, v in manifold_params.items() if k in ['perplexity', 'n_neighbors']])

    # Adaptar título y nombre de archivo para el caso Autoencoder="None"
    title = f"{dataset_name} | AE: {autoencoder_name} | Manifold: {manifold_name}\nManifold Params: {manifold_param_str}"
    safe_ae_name = autoencoder_name.replace("Autoencoder", "AE")

    safe_manifold_name = manifold_name
    safe_params = manifold_param_str.replace(":", "_").replace(", ", "_").replace(" ", "")

    filename = f"{dataset_name}_{safe_ae_name}_{safe_manifold_name}_{safe_params}.png"
    output_path = os.path.join(dataset_plot_dir, filename)

    # 2. Generación del gráfico
    plt.figure(figsize=(10, 8))

    # Usa 'c' (color) para mapear los puntos a colores basados en la etiqueta
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                          c=labels, cmap='Spectral', s=10, alpha=0.7)

    # Añadir leyenda de colores
    unique_labels = np.unique(labels)
    # Se añade la comprobación de cantidad de etiquetas para evitar fallos si hay demasiadas
    if len(unique_labels) > 1 and len(unique_labels) <= 20:
        cbar = plt.colorbar(scatter, ticks=unique_labels)
        cbar.set_label('Clase / Etiqueta', rotation=270, labelpad=15)

    plt.title(title, fontsize=10)
    plt.xlabel('Componente 1 (x)')
    plt.ylabel('Componente 2 (y)')
    plt.grid(True, linestyle='--', alpha=0.5)

    # 3. Guardar gráfico
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> Imagen guardada en: {output_path}")


def load_csv_data(dataset_name: str, config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Función de ayuda para cargar los datos de train/test y labels,
    utilizando las funciones load_data y load_labels de experiment_utils.
    """
    datasets_config = {"DATASETS": DATASETS, "BASE_DATA_PATH": BASE_DATA_PATH}
    sample_ratio = config.get("sample_ratio", 1.0)

    # Cargar datos de entrenamiento
    X_train = load_data(dataset_name, "train", datasets_config, sample_ratio)
    y_train = load_labels(dataset_name, "train", datasets_config, sample_ratio)

    # Cargar datos de prueba (usando el 100% del split de prueba para consistencia, aunque no se usa aquí)
    X_test = load_data(dataset_name, "test", datasets_config, 1.0)
    y_test = load_labels(dataset_name, "test", datasets_config, 1.0)

    if X_train is None or y_train is None:
        raise FileNotFoundError(f"Error al cargar datos para {dataset_name}. Revise los archivos.")

    return X_train, y_train, X_test, y_test


def run_experiment_D():
    """
    Evalúa TSNE y LLE directamente sobre los datos crudos (X_train)
    para una comparación base, y guarda los plots con el formato deseado.
    """
    print(f"--- Iniciando {EXPERIMENT_NAME}: Comparación Directa Manifold ---")

    all_results = []

    # Crear el directorio base para los plots
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Definir los modelos a ejecutar
    MODELS_TO_RUN = {
        "TSNE": (TSNE, TSNE_FIXED_PARAMS),
        "LocallyLinearEmbedding": (LLE, LLE_FIXED_PARAMS)
    }

    # 1. Iteración sobre Datasets
    for dataset_name, config in DATASETS.items():
        print(
            f"\n==================== DATASET: {dataset_name} (Sampling: {config.get('sample_ratio', 1.0) * 100:.1f}%) ====================")

        try:
            # Carga de datos
            X_train, y_train, X_test, y_test = load_csv_data(dataset_name, config)
            print(f"  Datos cargados (Train): {X_train.shape[0]} muestras")
        except FileNotFoundError as e:
            print(f"  ❌ Error de carga: {e}")
            continue
        except Exception as e:
            print(f"  ❌ Error desconocido al cargar datos: {e}")
            continue

        # 2. Iteración sobre Algoritmos Manifold
        for alg_name, (manifold_cls, manifold_params) in MODELS_TO_RUN.items():

            print(f"-> Combinación: AE=None | Manifold={alg_name}")

            try:
                # Instanciar modelo Manifold
                manifold_model = manifold_cls(**manifold_params)

                # Aplicar FIT + TRANSFORM y medir tiempo
                start_time = time.time()
                # Aplicación directa a los datos de entrada X_train
                X_train_emb = manifold_model.fit_transform(X_train)
                train_time = time.time() - start_time

                # Evaluación de métricas
                tw_train = trustworthiness(X_train, X_train_emb, n_neighbors=10)

                # Guardar Plot del embedding de entrenamiento
                plot_2d_representation(
                    embedding_2d=X_train_emb,
                    labels=y_train,
                    dataset_name=dataset_name,
                    autoencoder_name="None",  # Placeholder
                    ae_params={},  # Placeholder
                    manifold_name=alg_name,
                    manifold_params=manifold_params,
                    plot_dir=PLOTS_DIR
                )

                # Almacenar Resultados
                result = {
                    "dataset": dataset_name,
                    "autoencoder": "None",
                    "manifold": alg_name,
                    "ae_params": "N/A",
                    # Convertir los parámetros relevantes a string para el CSV
                    "manifold_params": str(
                        {k: v for k, v in manifold_params.items() if k in ['perplexity', 'n_neighbors']}),
                    "time_fit_s": train_time,
                    "time_transform_s": 0.0,
                    "trustworthiness_train": tw_train,
                    "trustworthiness_test": np.nan
                }
                all_results.append(result)

            except Exception as e:
                print(f"    ❌ Error al ejecutar {alg_name} en {dataset_name}: {e}")

    # 3. Guardar Resultados
    save_results_to_csv(all_results, OUTPUT_FILE)


if __name__ == "__main__":
    run_experiment_D()
