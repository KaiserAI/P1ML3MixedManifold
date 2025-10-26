import numpy as np
import os
from sklearn.manifold import TSNE, LocallyLinearEmbedding as LLE
from LinearAutoencoder import LinearAutoencoder
# Importamos load_labels y asumimos que evaluate_combination devuelve el embedding 2D
from experiments.experiment_utils import evaluate_combination, save_results_to_csv, load_data, load_labels
from dataset_config import DATASETS, BASE_DATA_PATH
import matplotlib.pyplot as plt  # NUEVA IMPORTACIÓN PARA PLOTS

# --- CONFIGURACIÓN DE EJECUCIÓN ESPECÍFICA DEL EXPERIMENTO ---
OUTPUT_DIR = "results"
OUTPUT_FILENAME = "exp_C_hyperparams_manifold.csv"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)  # Rutas compatibles con el SO
EPOCHS = 50
PLOTS_DIR = "plots/exp_C"  # NUEVA CONSTANTE: Directorio para guardar las imágenes

# NUEVO: Porcentaje de datos a usar para este experimento (0.0 a 1.0)
SAMPLE_PERCENTAGE = 0.5

# Parámetros a explorar para los Manifolds
PERPLEXITY_VALUES = [5, 30, 50]
N_NEIGHBORS_VALUES = [10, 30, 50]

# Pasar la configuración consolidada a la función load_data
DATASETS_CONFIG = {"BASE_DATA_PATH": BASE_DATA_PATH, "DATASETS": DATASETS}


# --- NUEVA FUNCIÓN DE UTILIDAD PARA PLOTEAR ---
def plot_2d_representation(embedding_2d: np.ndarray, labels: np.ndarray, dataset_name: str,
                           autoencoder_name: str, ae_params: dict, manifold_name: str,
                           manifold_params: dict, plot_dir: str):
    """Genera y guarda un gráfico de dispersión (scatter plot) 2D."""

    # Crea el subdirectorio para el dataset si no existe
    dataset_plot_dir = os.path.join(plot_dir, dataset_name)
    os.makedirs(dataset_plot_dir, exist_ok=True)

    # 1. Título y nombre del archivo
    # Usamos los parámetros clave del manifold, ya que el AE es fijo en este experimento
    manifold_param_str = ", ".join([f"{k}:{v}" for k, v in manifold_params.items()])
    title = f"{dataset_name} | AE: {autoencoder_name} | Manifold: {manifold_name}\nManifold Params: {manifold_param_str}"

    # Generar un nombre de archivo limpio y único
    safe_ae_name = autoencoder_name.replace("Autoencoder", "AE")
    safe_manifold_name = manifold_name
    safe_params = manifold_param_str.replace(":", "_").replace(", ", "_").replace(" ", "").replace("{", "").replace("}",
                                                                                                                    "")

    filename = f"{dataset_name}_{safe_ae_name}_{safe_manifold_name}_{safe_params}.png"
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

    plt.title(title, fontsize=10)
    plt.xlabel('Componente 1 (x)')
    plt.ylabel('Componente 2 (y)')
    plt.grid(True, linestyle='--', alpha=0.5)

    # 3. Guardar gráfico
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> Imagen guardada en: {output_path}")


def run_experiment_C():
    all_results = []

    # Autoencoder fijo (LinearAutoencoder)
    ae_cls = LinearAutoencoder
    ae_name = ae_cls.__name__

    print("\n--- Ejecutando Experimento C: Hiperparámetros del Manifold ---")

    # Crear el directorio base para los plots
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Bucle principal sobre la lista de datasets
    for dataset_name in DATASETS:
        print(
            f"\n==================== DATASET: {dataset_name} (Sampling: {SAMPLE_PERCENTAGE * 100:.1f}%) ====================")

        # Carga de datos con porcentaje de muestreo
        train_data = load_data(dataset_name, "train", DATASETS_CONFIG, sample_ratio=SAMPLE_PERCENTAGE)
        test_data = load_data(dataset_name, "test", DATASETS_CONFIG, sample_ratio=SAMPLE_PERCENTAGE)

        # Carga de etiquetas
        try:
            train_labels = load_labels(dataset_name, "train", DATASETS_CONFIG, sample_ratio=SAMPLE_PERCENTAGE)
        except NameError:
            print("AVISO: No se encontró la función 'load_labels'. No se generarán gráficos con etiquetas de color.")
            train_labels = np.zeros(train_data.shape[0])

        if train_data is None or test_data is None:
            continue

        # Detección automática de la dimensión de entrada
        INPUT_DIM = train_data.shape[1]
        print(f"INPUT_DIM detectado: {INPUT_DIM}")

        # Parámetros del AE fijo para este experimento
        default_ae_params = {"input_dim": INPUT_DIM, "epochs": EPOCHS, "batch_size": 128}

        # ----------------------------------------------------
        # 1. Hiperparámetros TSNE (Perplexity)
        # ----------------------------------------------------
        print("\n-> C.1: Explorando TSNE (Perplexity)")
        for perplexity in PERPLEXITY_VALUES:
            manifold_name = "TSNE"
            manifold_params = {"n_components": 2, "perplexity": perplexity}

            print(f"-> Combinación: AE={ae_name} | Manifold={manifold_name} | Perplexity={perplexity}")

            # ATENCIÓN: Se asume que evaluate_combination ahora devuelve una tupla de (metrics, train_embedding_2d)
            metrics, train_embedding_2d = evaluate_combination(
                autoencoder_cls=ae_cls,
                manifold_cls=TSNE,
                train_data=train_data,
                test_data=test_data,
                ae_params=default_ae_params,
                manifold_params=manifold_params,
                dataset_name=dataset_name
            )
            all_results.append(metrics)

            # NUEVO: Generar y guardar la imagen
            plot_2d_representation(
                embedding_2d=train_embedding_2d,
                labels=train_labels,
                dataset_name=dataset_name,
                autoencoder_name=ae_name,
                ae_params=default_ae_params,
                manifold_name=manifold_name,
                manifold_params=manifold_params,
                plot_dir=PLOTS_DIR
            )

        # ----------------------------------------------------
        # 2. Hiperparámetros LLE (n_neighbors)
        # ----------------------------------------------------
        print("\n-> C.2: Explorando LLE (n_neighbors)")
        for n_neighbors in N_NEIGHBORS_VALUES:
            manifold_name = "LocallyLinearEmbedding"
            manifold_params = {"n_components": 2, "n_neighbors": n_neighbors}

            print(f"-> Combinación: AE={ae_name} | Manifold={manifold_name} | n_neighbors={n_neighbors}")

            # ATENCIÓN: Se asume que evaluate_combination ahora devuelve una tupla de (metrics, train_embedding_2d)
            metrics, train_embedding_2d = evaluate_combination(
                autoencoder_cls=ae_cls,
                manifold_cls=LLE,
                train_data=train_data,
                test_data=test_data,
                ae_params=default_ae_params,
                manifold_params=manifold_params,
                dataset_name=dataset_name
            )
            all_results.append(metrics)

            # NUEVO: Generar y guardar la imagen
            plot_2d_representation(
                embedding_2d=train_embedding_2d,
                labels=train_labels,
                dataset_name=dataset_name,
                autoencoder_name=ae_name,
                ae_params=default_ae_params,
                manifold_name=manifold_name,
                manifold_params=manifold_params,
                plot_dir=PLOTS_DIR
            )

    # Guardar todos los resultados
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_results_to_csv(all_results, OUTPUT_PATH)


if __name__ == '__main__':
    run_experiment_C()
