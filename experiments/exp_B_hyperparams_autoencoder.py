import numpy as np
import os
from sklearn.manifold import TSNE
from LinearSparseAutoencoder import LinearSparseAutoencoder
from DenoisingSparseAutoencoder import DenoisingSparseAutoencoder
from experiments.experiment_utils import evaluate_combination, save_results_to_csv, load_data, load_labels
import matplotlib.pyplot as plt
from dataset_config import DATASETS, BASE_DATA_PATH

# --- CONFIGURACIÓN DE EJECUCIÓN ESPECÍFICA DEL EXPERIMENTO ---
OUTPUT_DIR = "results"
OUTPUT_FILENAME = "exp_B_hyperparams_autoencoder.csv"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)  # Rutas compatibles con el SO
EPOCHS = 50
DEFAULT_PERPLEXITY = 30  # TSNE fijo
PLOTS_DIR = "plots/exp_B"  # Directorio para guardar las imágenes


# Parámetros a explorar para los Autoencoders
LAMBDA_VALUES = [1e-4, 1e-3, 1e-2]
NOISE_FACTORS = [0.1, 0.3, 0.5]

# Pasar la configuración consolidada a la función load_data
DATASETS_CONFIG = {"BASE_DATA_PATH": BASE_DATA_PATH, "DATASETS": DATASETS}


# --- FUNCIÓN DE UTILIDAD PARA PLOTEAR ---
def plot_2d_representation(embedding_2d: np.ndarray, labels: np.ndarray, dataset_name: str,
                           autoencoder_name: str, ae_params: dict, manifold_name: str, plot_dir: str):
    """Genera y guarda un gráfico de dispersión (scatter plot) 2D."""

    # Crea el subdirectorio para el dataset si no existe
    dataset_plot_dir = os.path.join(plot_dir, dataset_name)
    os.makedirs(dataset_plot_dir, exist_ok=True)

    # 1. Título y nombre del archivo
    # Filtramos los parámetros por defecto para hacer el título más limpio
    ae_param_str = ", ".join(
        [f"{k}:{v}" for k, v in ae_params.items() if k not in ['input_dim', 'epochs', 'batch_size']])
    title = f"{dataset_name} | AE: {autoencoder_name} | Manifold: {manifold_name}\nAE Params: {ae_param_str}"

    # Generar un nombre de archivo limpio y único
    safe_ae_name = autoencoder_name.replace("Autoencoder", "AE")
    safe_params = ae_param_str.replace(":", "_").replace(", ", "_").replace(" ", "").replace("{", "").replace("}", "")

    filename = f"{dataset_name}_{safe_ae_name}_{manifold_name}_{safe_params}.png"
    output_path = os.path.join(dataset_plot_dir, filename)

    # 2. Generación del gráfico
    plt.figure(figsize=(10, 8))

    # Usa 'c' (color) para mapear los puntos a colores basados en la etiqueta
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                          c=labels, cmap='Spectral', s=10, alpha=0.7)

    # Añadir leyenda de colores
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1 and len(unique_labels) <= 20:
        cbar = plt.colorbar(scatter, ticks=unique_labels)
        cbar.set_label('Clase / Etiqueta', rotation=270, labelpad=15)

    plt.title(title)
    plt.xlabel('Componente 1 (x)')
    plt.ylabel('Componente 2 (y)')
    plt.grid(True, linestyle='--', alpha=0.5)

    # 3. Guardar gráfico
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> Imagen guardada en: {output_path}")


def run_experiment_B():
    all_results = []

    # Manifold fijo (TSNE)
    manifold_cls = TSNE
    manifold_name = "TSNE"
    manifold_params = {"n_components": 2, "perplexity": DEFAULT_PERPLEXITY}

    print("\n--- Ejecutando Experimento B: Hiperparámetros del Autoencoder ---")

    # Crear el directorio base para los plots
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Bucle principal sobre la lista de datasets
    for dataset_name, ds_info in DATASETS.items():

        # 1. OBTENER EL SAMPLE RATIO ESPECÍFICO DEL DATASET
        sample_ratio = ds_info.get("sample_ratio", 1.0)

        print(
            f"\n==================== DATASET: {dataset_name} (Sampling: {sample_ratio * 100:.1f}%) ====================")

        # Carga de datos con porcentaje de muestreo
        train_data = load_data(dataset_name, "train", DATASETS_CONFIG, sample_ratio=sample_ratio)
        test_data = load_data(dataset_name, "test", DATASETS_CONFIG, sample_ratio=sample_ratio)

        # Carga de etiquetas
        try:
            train_labels = load_labels(dataset_name, "train", DATASETS_CONFIG, sample_ratio=sample_ratio)
        except NameError:
            print("AVISO: No se encontró la función 'load_labels'. No se generarán gráficos con etiquetas de color.")
            if train_data is not None:
                train_labels = np.zeros(train_data.shape[0])
            else:
                continue

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
            autoencoder_cls = LinearSparseAutoencoder
            autoencoder_name = "LinearSparseAutoencoder"

            print(f"-> Combinación: AE=Sparse | lambda={lambda_sparse}")

            metrics, train_embedding_2d = evaluate_combination(
                autoencoder_cls=autoencoder_cls,
                manifold_cls=manifold_cls,
                train_data=train_data,
                test_data=test_data,
                ae_params=ae_params,
                manifold_params=manifold_params,
                dataset_name=dataset_name
            )
            all_results.append(metrics)

            # Generar y guardar la imagen
            plot_2d_representation(
                embedding_2d=train_embedding_2d,
                labels=train_labels,
                dataset_name=dataset_name,
                autoencoder_name=autoencoder_name,
                ae_params=ae_params,
                manifold_name=manifold_name,
                plot_dir=PLOTS_DIR
            )

        # ----------------------------------------------------
        # 2. Hiperparámetros de Ruido (DenoisingSparse, noise_factor)
        # ----------------------------------------------------
        print("\n-> B.2: Explorando DenoisingSparse (noise_factor)")
        for noise_factor in NOISE_FACTORS:
            ae_params = default_ae_params.copy()
            ae_params["lambda_sparse"] = 1e-3  # Fijo
            ae_params["noise_factor"] = noise_factor
            autoencoder_cls = DenoisingSparseAutoencoder
            autoencoder_name = "DenoisingSparseAutoencoder"

            print(f"-> Combinación: AE=DenoisingSparse | noise={noise_factor}")

            metrics, train_embedding_2d = evaluate_combination(
                autoencoder_cls=autoencoder_cls,
                manifold_cls=manifold_cls,
                train_data=train_data,
                test_data=test_data,
                ae_params=ae_params,
                manifold_params=manifold_params,
                dataset_name=dataset_name
            )
            all_results.append(metrics)

            # Generar y guardar la imagen
            plot_2d_representation(
                embedding_2d=train_embedding_2d,
                labels=train_labels,
                dataset_name=dataset_name,
                autoencoder_name=autoencoder_name,
                ae_params=ae_params,
                manifold_name=manifold_name,
                plot_dir=PLOTS_DIR
            )

    # Guardar todos los resultados
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_results_to_csv(all_results, OUTPUT_PATH)


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_experiment_B()
