import numpy as np
import os
from sklearn.manifold import TSNE, LocallyLinearEmbedding as LLE
from LinearAutoencoder import LinearAutoencoder
from LinearSparseAutoencoder import LinearSparseAutoencoder
from DenoisingSparseAutoencoder import DenoisingSparseAutoencoder
# from IdentityAutoencoder import IdentityAutoencoder # <- ELIMINADO
from experiments.experiment_utils import evaluate_combination, save_results_to_csv, load_data
# Importar la configuración de los datasets
from dataset_config import DATASETS, BASE_DATA_PATH

# --- CONFIGURACIÓN DE EJECUCIÓN ---
# Nota: La salida debería ir a una carpeta para organizarse mejor, por ejemplo, 'experiments/results'.
# Aquí se usa 'results' como en tu ejemplo.
OUTPUT_DIR = "results"
OUTPUT_FILENAME = "exp_A_autoencoder_vs_manifold.csv"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
EPOCHS = 50
N_LLE_NEIGHBORS = 30

# Pasar la configuración consolidada a la función load_data
DATASETS_CONFIG = {"BASE_DATA_PATH": BASE_DATA_PATH, "DATASETS": DATASETS}


def run_experiment_A():
    all_results = []

    # Lista de Autoencoders, SÓLO incluyendo los tres modelos desarrollados
    ae_classes = {
        "Linear": LinearAutoencoder,
        "Sparse": LinearSparseAutoencoder,
        "DenoisingSparse": DenoisingSparseAutoencoder
    }
    manifold_classes = {
        "TSNE": TSNE,
        "LLE": LLE
    }

    print("\n--- Ejecutando Experimento A: Comparativa General sobre Múltiples Datasets ---")

    # Bucle principal sobre la lista de datasets
    for dataset_name in DATASETS:
        print(f"\n==================== DATASET: {dataset_name} ====================")

        train_data = load_data(dataset_name, "train", DATASETS_CONFIG)
        test_data = load_data(dataset_name, "test", DATASETS_CONFIG)

        if train_data is None or test_data is None:
            continue

        # El input_dim se extrae directamente de los datos cargados
        INPUT_DIM = train_data.shape[1]
        print(f"Usando INPUT_DIM={INPUT_DIM}")

        default_ae_params = {"input_dim": INPUT_DIM, "epochs": EPOCHS, "batch_size": 128}

        for ae_name, ae_cls in ae_classes.items():
            ae_p = default_ae_params.copy()

            # Lógica de asignación de hiperparámetros de regularización
            if ae_name == "Sparse":
                ae_p["lambda_sparse"] = 1e-3
            elif ae_name == "DenoisingSparse":
                ae_p["lambda_sparse"] = 1e-3
                ae_p["noise_factor"] = 0.2

            # Nota: 'Linear' (básico) usa solo los parámetros por defecto (input_dim, epochs, batch_size).

            for m_name, m_cls in manifold_classes.items():

                m_p = {"n_components": 2}
                if m_name == "LLE":
                    m_p["n_neighbors"] = N_LLE_NEIGHBORS

                print(f"-> Combinación: AE={ae_name} | Manifold={m_name}")

                metrics = evaluate_combination(
                    autoencoder_cls=ae_cls,
                    manifold_cls=m_cls,
                    train_data=train_data,
                    test_data=test_data,
                    ae_params=ae_p,
                    manifold_params=m_p,
                    dataset_name=dataset_name
                )
                all_results.append(metrics)

    save_results_to_csv(all_results, OUTPUT_PATH)


if __name__ == '__main__':
    # Asegurarse de que el directorio de resultados existe
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_experiment_A()
