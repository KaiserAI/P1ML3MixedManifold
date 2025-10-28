import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os  # Necesario para la manipulación de archivos
from MixedManifoldDetector import MixedManifoldDetector
from LinearAutoencoder import LinearAutoencoder
from sklearn.manifold import TSNE, LocallyLinearEmbedding

# --- CONSTANTES DE ARCHIVO ---
PLOTS_OUTPUT_DIR = "final_plots"


def load_csv(path: str, sample_fraction: float = None, random_state: int = 42):
    """
    Carga un CSV con formato: label, feature1, feature2, ...
    Devuelve una tupla (labels, features).
    """
    df = pd.read_csv(path, header=None, skiprows=0)

    if sample_fraction is not None and 0 < sample_fraction < 1:
        # Muestreo aleatorio si se especifica una fracción
        df = df.sample(frac=sample_fraction, random_state=random_state)

    # FIX: Conversión robusta de etiquetas y manejo del encabezado
    labels_series = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    valid_mask = labels_series.notna()

    # Filtramos las filas donde la etiqueta no es un número (elimina el encabezado)
    labels = labels_series.loc[valid_mask].values.astype(int)
    X = df.loc[valid_mask].iloc[:, 1:].values.astype(np.float32)

    # Normalización min-max a [0, 1]
    X = (X - X.min()) / (X.max() - X.min() + 1e-8)

    return labels, X


def main():
    # Comprobación de argumentos de línea de comandos
    if len(sys.argv) < 2:
        print("Uso: python mixed_manifold_detector.py <train.csv> [test.csv]")
        sys.exit(1)

    train_path = sys.argv[1]
    test_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Crear la carpeta de salida para los plots
    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)

    # Nota: Se mantiene el sample_fraction=0.1 del código original.
    print(f"🔹 Cargando datos de entrenamiento desde {train_path} (Muestreo: 10%)")
    train_labels, train_data = load_csv(train_path, sample_fraction=0.1)
    if test_path:
        print(f"🔹 Cargando datos de test desde {test_path} (Muestreo: 10%)")
        test_labels, test_data = load_csv(test_path, sample_fraction=0.1)

    # =========================================================================
    # CONFIGURACIÓN DEL MODELO FINAL ÓPTIMO (LinearAE + TSNE Perplexity 50)
    # =========================================================================

    FINAL_AE_CLASS = LinearAutoencoder
    FINAL_MANIFOLD_CLASS = TSNE

    autoencoder = FINAL_AE_CLASS(
        input_dim=train_data.shape[1],
        epochs=50,
        batch_size=64
    )

    manifold = FINAL_MANIFOLD_CLASS(
        n_components=2,
        random_state=42,
        perplexity=30
    )

    detector = MixedManifoldDetector(autoencoder=autoencoder, manifold_alg=manifold)

    # Entrenamiento
    print("🔹 Entrenando el modelo mixto...")
    train_embedding = detector.fit_transform(train_data)
    print("✅ Entrenamiento completado.")

    # --- Visualización y Guardado TRAIN ---
    train_title = f"Representación 2D del conjunto de entrenamiento\nAE: {FINAL_AE_CLASS.__name__}, Manifold: {FINAL_MANIFOLD_CLASS.__name__} (Perplexity=50)"

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(train_embedding[:, 0], train_embedding[:, 1], c=train_labels, cmap="tab10", s=10)
    plt.colorbar(scatter, label="Etiqueta")
    plt.title(train_title)
    plt.xlabel("Dimensión 1")
    plt.ylabel("Dimensión 2")
    plt.tight_layout()

    # GUARDAR EL GRÁFICO DE ENTRENAMIENTO
    train_filename = os.path.join(PLOTS_OUTPUT_DIR, "final_train_embedding.png")
    plt.savefig(train_filename)
    print(f"🖼️ Gráfico de entrenamiento guardado en: {train_filename}")
    plt.show()  # Intentará mostrar (puede dar error gráfico, pero no detendrá el proceso si el siguiente savefig es exitoso)

    # Test (si se pasa como argumento)
    if test_path:
        print("🔹 Transformando datos de test...")
        test_embedding = detector.transform(test_data)

        # --- Visualización y Guardado TEST ---
        test_title = f"Representación 2D del conjunto de test\nAE: {FINAL_AE_CLASS.__name__}, Manifold: {FINAL_MANIFOLD_CLASS.__name__} (Perplexity=50)"

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(test_embedding[:, 0], test_embedding[:, 1], c=test_labels, cmap="tab10", s=10)
        plt.colorbar(scatter, label="Etiqueta")
        plt.title(test_title)
        plt.xlabel("Dimensión 1")
        plt.ylabel("Dimensión 2")
        plt.tight_layout()

        # GUARDAR EL GRÁFICO DE TEST
        test_filename = os.path.join(PLOTS_OUTPUT_DIR, "final_test_embedding.png")
        plt.savefig(test_filename)
        print(f"🖼️ Gráfico de test guardado en: {test_filename}")
        plt.show()  # Intentará mostrar (puede dar error gráfico, pero ya está guardado)


if __name__ == "__main__":
    main()
