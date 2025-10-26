import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MixedManifoldDetector import MixedManifoldDetector
from LinearAutoencoder import LinearAutoencoder
from sklearn.manifold import TSNE, LocallyLinearEmbedding


def load_csv(path: str, sample_fraction: float = None, random_state: int = 42):
    """
    Carga un CSV con formato: label, feature1, feature2, ...

    Devuelve una tupla (labels, features).

    Parámetros:
    - path: Ruta al archivo CSV.
    - sample_fraction: Fracción del dataset a usar (0 < f <= 1).
      Ejemplo: 0.1 usa el 10 % del dataset.
    - random_state: Semilla para la muestreo reproducible.
    """
    df = pd.read_csv(path)

    if sample_fraction is not None and 0 < sample_fraction < 1:
        # Muestreo aleatorio si se especifica una fracción
        df = df.sample(frac=sample_fraction, random_state=random_state)

    # La primera columna es la etiqueta (label)
    labels = df.iloc[:, 0].values
    # El resto son las características (features), convertidas a float32
    X = df.iloc[:, 1:].values.astype(np.float32)

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

    print(f"🔹 Cargando datos de entrenamiento desde {train_path}")
    # Carga de datos, usando un 10% del conjunto para rapidez
    train_labels, train_data = load_csv(train_path, sample_fraction=0.1)
    if test_path:
        test_labels, test_data = load_csv(test_path, sample_fraction=0.1)


    # Configuración del sistema
    autoencoder = LinearAutoencoder(input_dim=train_data.shape[1], epochs=50, batch_size=64)
    manifold = TSNE(n_components=2, random_state=42)

    detector = MixedManifoldDetector(autoencoder=autoencoder, manifold_alg=manifold)

    # Entrenamiento
    print("🔹 Entrenando el modelo mixto...")
    # El método fit_transform entrena el autoencoder y luego calcula la proyección TSNE
    train_embedding = detector.fit_transform(train_data)
    print("✅ Entrenamiento completado.")

    # Visualización TRAIN
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(train_embedding[:, 0], train_embedding[:, 1], c=train_labels, cmap="tab10", s=10)
    plt.colorbar(scatter, label="Etiqueta")
    plt.title("Representación 2D del conjunto de entrenamiento")
    plt.xlabel("Dimensión 1")
    plt.ylabel("Dimensión 2")
    plt.tight_layout()
    plt.show()

    # Test (si se pasa como argumento)
    if test_path:
        print("🔹 Transformando datos de test...")
        # Aplica el encoder entrenado y la proyección manifold
        test_embedding = detector.transform(test_data)

        # Visualización TEST
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(test_embedding[:, 0], test_embedding[:, 1], c=test_labels, cmap="tab10", s=10)
        plt.colorbar(scatter, label="Etiqueta")
        plt.title("Representación 2D del conjunto de test")
        plt.xlabel("Dimensión 1")
        plt.ylabel("Dimensión 2")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
