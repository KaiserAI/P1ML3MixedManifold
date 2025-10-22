import numpy as np
import os
from sklearn.manifold import TSNE
from LinearSparseAutoencoder import LinearSparseAutoencoder
from DenoisingSparseAutoencoder import DenoisingSparseAutoencoder
from experiments.experiment_utils import evaluate_combination, save_results_to_csv, load_data
from dataset_config import DATASETS, BASE_DATA_PATH # Importar la configuración de los datasets

# --- CONFIGURACIÓN DE EJECUCIÓN ---
OUTPUT_DIR = "results"
OUTPUT_FILENAME = "exp_B_hyperparams_autoencoder.csv"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME) # Rutas compatibles con el SO
EPOCHS = 50 
DEFAULT_PERPLEXITY = 30 # TSNE fijo

# Parámetros a explorar para los Autoencoders
LAMBDA_VALUES = [1e-4, 1e-3, 1e-2]
NOISE_FACTORS = [0.1, 0.3, 0.5]

# Pasar la configuración consolidada a la función load_data
DATASETS_CONFIG = {"BASE_DATA_PATH": BASE_DATA_PATH, "DATASETS": DATASETS}


def run_experiment_B():
    all_results = []
    
    # Manifold fijo (TSNE)
    manifold_cls = TSNE
    manifold_params = {"n_components": 2, "perplexity": DEFAULT_PERPLEXITY}

    print("\n--- Ejecutando Experimento B: Hiperparámetros del Autoencoder ---")
    
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
        # 1. Hiperparámetros de Regularización Sparse (lambda_sparse)
        # ----------------------------------------------------
        print("\n-> B.1: Explorando LinearSparse (lambda_sparse)")
        for lambda_sparse in LAMBDA_VALUES:
            ae_params = default_ae_params.copy()
            ae_params["lambda_sparse"] = lambda_sparse
            
            print(f"-> Combinación: AE=Sparse | lambda={lambda_sparse}")
            metrics = evaluate_combination(
                autoencoder_cls=LinearSparseAutoencoder, 
                manifold_cls=manifold_cls, 
                train_data=train_data, 
                test_data=test_data,
                ae_params=ae_params,
                manifold_params=manifold_params,
                dataset_name=dataset_name
            )
            all_results.append(metrics)

        # ----------------------------------------------------
        # 2. Hiperparámetros de Ruido (DenoisingSparse, noise_factor)
        # ----------------------------------------------------
        print("\n-> B.2: Explorando DenoisingSparse (noise_factor)")
        for noise_factor in NOISE_FACTORS:
            ae_params = default_ae_params.copy()
            ae_params["lambda_sparse"] = 1e-3 # Fijo
            ae_params["noise_factor"] = noise_factor
            
            print(f"-> Combinación: AE=DenoisingSparse | noise={noise_factor}")
            metrics = evaluate_combination(
                autoencoder_cls=DenoisingSparseAutoencoder, 
                manifold_cls=manifold_cls, 
                train_data=train_data, 
                test_data=test_data,
                ae_params=ae_params,
                manifold_params=manifold_params,
                dataset_name=dataset_name
            )
            all_results.append(metrics)

    # Guardar todos los resultados
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_results_to_csv(all_results, OUTPUT_PATH)

if __name__ == '__main__':
    run_experiment_B()
