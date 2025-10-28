import numpy as np
import os
from sklearn.manifold import TSNE, LocallyLinearEmbedding as LLE
from LinearAutoencoder import LinearAutoencoder
from LinearSparseAutoencoder import LinearSparseAutoencoder
from DenoisingSparseAutoencoder import DenoisingSparseAutoencoder
from experiments.experiment_utils import evaluate_combination, save_results_to_csv, load_data, load_labels
from dataset_config import DATASETS, BASE_DATA_PATH
import matplotlib.pyplot as plt

# --- PARÁMETROS GLOBALES ÓPTIMOS (Derivados del Análisis Ponderado) ---
OPTIMAL_PARAMS_FOR_EXP_A = {
    'LinearSparseAutoencoder': {'lambda_sparse': 0.0001},
    'DenoisingSparseAutoencoder': {'lambda_sparse': 0.001, 'noise_factor': 0.3},
    'TSNE': {'perplexity': 30},
    'LLE': {'n_neighbors': 10},
}
# --- CONFIGURACIÓN DE EJECUCIÓN ESPECÍFICA DEL EXPERIMENTO ---
OUTPUT_DIR = "results"
OUTPUT_FILENAME = "exp_A_autoencoder_vs_manifold_OPTIMAL.csv" # Nuevo nombre para diferenciar
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
EPOCHS = 50
PLOTS_DIR = "plots/exp_A_optimal_vs_manifold" # Nuevo directorio de plots

# REMOVIDA LA CONSTANTE N_LLE_NEIGHBORS (ahora se usa el valor óptimo 10)

# Pasar la configuración consolidada a la función load_data
DATASETS_CONFIG = {"BASE_DATA_PATH": BASE_DATA_PATH, "DATASETS": DATASETS}


# --- FUNCIÓN DE UTILIDAD PARA PLOTEAR ---
def plot_2d_representation(embedding_2d: np.ndarray, labels: np.ndarray, dataset_name: str,
                           autoencoder_name: str, ae_params: dict, manifold_name: str,
                           manifold_params: dict, plot_dir: str):
    """Genera y guarda un gráfico de dispersión (scatter plot) 2D."""

    # Crea el subdirectorio para el dataset si no existe
    dataset_plot_dir = os.path.join(plot_dir, dataset_name)
    os.makedirs(dataset_plot_dir, exist_ok=True)

    # 1. Título y nombre del archivo
    ae_param_str = ", ".join(
        [f"{k}:{v}" for k, v in ae_params.items() if k not in ['input_dim', 'epochs', 'batch_size']])
    m_p_str = ", ".join([f"{k}:{v}" for k, v in manifold_params.items() if k not in ['n_components']])
    title = f"{dataset_name} | AE: {autoencoder_name} ({ae_param_str})\nManifold: {manifold_name} ({m_p_str})"

    # Generar un nombre de archivo limpio y único
    safe_ae_name = autoencoder_name.replace("Autoencoder", "AE")
    safe_manifold_name = manifold_name
    safe_ae_params = ae_param_str.replace(":", "_").replace(", ", "_").replace(" ", "").replace("{", "").replace("}", "")
    safe_m_params = m_p_str.replace(":", "_").replace(", ", "_").replace(" ", "").replace("{", "").replace("}", "")


    filename = f"{dataset_name}_{safe_ae_name}_{safe_ae_params}_{safe_manifold_name}_{safe_m_params}.png"
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


def run_experiment_A():
    all_results = []

    ae_classes = {
        "Linear": LinearAutoencoder,
        "Sparse": LinearSparseAutoencoder,
        "DenoisingSparse": DenoisingSparseAutoencoder
    }
    manifold_classes = {
        "TSNE": TSNE,
        "LLE": LLE
    }

    print("\n--- Ejecutando Experimento A: Comparativa General con Hiperparámetros Óptimos Ponderados ---")
    os.makedirs(PLOTS_DIR, exist_ok=True)

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
        except Exception:
            print("AVISO: Fallo al cargar etiquetas. Usando etiquetas dummy.")
            if train_data is not None:
                train_labels = np.zeros(train_data.shape[0])
            else:
                continue

        if train_data is None or test_data is None:
            continue

        # El input_dim se extrae directamente del array cargado
        INPUT_DIM = train_data.shape[1]
        print(f"Usando INPUT_DIM={INPUT_DIM}")

        default_ae_params = {"input_dim": INPUT_DIM, "epochs": EPOCHS, "batch_size": 128}

        for ae_name, ae_cls in ae_classes.items():
            ae_p = default_ae_params.copy()

            # --- APLICAR HIPERPARÁMETROS ÓPTIMOS DEL AUTOENCODER ---
            if ae_name == "Sparse":
                ae_p.update(OPTIMAL_PARAMS_FOR_EXP_A['LinearSparseAutoencoder'])
            elif ae_name == "DenoisingSparse":
                ae_p.update(OPTIMAL_PARAMS_FOR_EXP_A['DenoisingSparseAutoencoder'])

            for m_name, m_cls in manifold_classes.items():

                # --- APLICAR HIPERPARÁMETROS ÓPTIMOS DEL MANIFOLD ---
                m_p = {"n_components": 2}
                if m_name == "TSNE":
                    m_p.update(OPTIMAL_PARAMS_FOR_EXP_A['TSNE'])
                elif m_name == "LLE":
                    m_p.update(OPTIMAL_PARAMS_FOR_EXP_A['LLE'])


                print(f"-> Combinación: AE={ae_name} | Manifold={m_name}")
                print(f"   AE Params: {ae_p} | Manifold Params: {m_p}")

                # ATENCIÓN: Capturamos el embedding 2D
                metrics, train_embedding_2d = evaluate_combination(
                    autoencoder_cls=ae_cls,
                    manifold_cls=m_cls,
                    train_data=train_data,
                    test_data=test_data,
                    ae_params=ae_p,
                    manifold_params=m_p,
                    dataset_name=dataset_name
                )
                all_results.append(metrics)

                # NUEVO: Generar y guardar la imagen
                plot_2d_representation(
                    embedding_2d=train_embedding_2d,
                    labels=train_labels,
                    dataset_name=dataset_name,
                    autoencoder_name=ae_name,
                    ae_params=ae_p,
                    manifold_name=m_name,
                    manifold_params=m_p,
                    plot_dir=PLOTS_DIR
                )

    save_results_to_csv(all_results, OUTPUT_PATH)


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_experiment_A()
