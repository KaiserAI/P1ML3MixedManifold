import numpy as np
import os
from sklearn.manifold import TSNE, LocallyLinearEmbedding as LLE
from LinearAutoencoder import LinearAutoencoder
from experiments.experiment_utils import evaluate_combination, save_results_to_csv, load_data
from dataset_config import DATASETS, BASE_DATA_PATH # Importar la configuración de los datasets

# --- CONFIGURACIÓN DE EJECUCIÓN ---
OUTPUT_DIR = "results"
OUTPUT_FILENAME = "exp_C_hyperparams_manifold.csv"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME) # Rutas compatibles con el SO
EPOCHS = 50

# Parámetros a explorar para los Manifolds
PERPLEXITY_VALUES = [5, 30, 50]
N_NEIGHBORS_VALUES = [10, 30, 50]

# Pasar la configuración consolidada a la función load_data
DATASETS_CONFIG = {"BASE_DATA_PATH": BASE_DATA_PATH, "DATASETS": DATASETS}


def run_experiment_C():
    all_results = []

    # Autoencoder fijo (LinearAutoencoder)
    ae_cls = LinearAutoencoder

    print("\n--- Ejecutando Experimento C: Hiperparámetros del Manifold ---")

    # Bucle principal sobre la lista de datasets
    for dataset_name in DATASETS:
        print(f"\n==================== DATASET: {dataset_name} ====================")

        train_data = load_data(dataset_name, "train", DATASETS_CONFIG)
        test_data = load_data(dataset_name, "test", DATASETS_CONFIG)

        if train_data is None or test_data is None:
            continue

        # Detección automática de la dimensión de entrada
        INPUT_DIM = train_data.shape[1]
        print(f"INPUT_DIM detectado: {INPUT_DIM}")

        default_ae_params = {"input_dim": INPUT_DIM, "epochs": EPOCHS, "batch_size": 128}

        # ----------------------------------------------------
        # 1. Hiperparámetros TSNE (Perplexity)
        # ----------------------------------------------------
        print("\n-> C.1: Explorando TSNE (Perplexity)")
        for perplexity in PERPLEXITY_VALUES:
            manifold_params = {"n_components": 2, "perplexity": perplexity}

            print(f"-> Combinación: Manifold=TSNE | Perplexity={perplexity}")
            metrics = evaluate_combination(
                autoencoder_cls=ae_cls,
                manifold_cls=TSNE,
                train_data=train_data,
                test_data=test_data,
                ae_params=default_ae_params,
                manifold_params=manifold_params,
                dataset_name=dataset_name
            )
            all_results.append(metrics)

        # ----------------------------------------------------
        # 2. Hiperparámetros LLE (n_neighbors)
        # ----------------------------------------------------
        print("\n-> C.2: Explorando LLE (n_neighbors)")
        for n_neighbors in N_NEIGHBORS_VALUES:
            manifold_params = {"n_components": 2, "n_neighbors": n_neighbors}

            print(f"-> Combinación: Manifold=LLE | n_neighbors={n_neighbors}")
            metrics = evaluate_combination(
                autoencoder_cls=ae_cls,
                manifold_cls=LLE,
                train_data=train_data,
                test_data=test_data,
                ae_params=default_ae_params,
                manifold_params=manifold_params,
                dataset_name=dataset_name
            )
            all_results.append(metrics)

    # Guardar todos los resultados
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_results_to_csv(all_results, OUTPUT_PATH)

if __name__ == '__main__':
    run_experiment_C()
