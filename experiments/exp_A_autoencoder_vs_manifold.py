"""
EXPERIMENTO A:
Comparativa de combinaciones Autoencoder + Manifold.
"""

import os
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from LinearAutoencoder import LinearAutoencoder
from LinearSparseAutoencoder import LinearSparseAutoencoder
from DenoisingSparseAutoencoder import DenoisingSparseAutoencoder
from mixed_manifold_detector import load_csv
from experiment_utils import evaluate_combination, save_results_to_csv

# Carga de datos reducidos
labels, train_data = load_csv("../data/MNIST/mnist_train.csv", sample_fraction=0.05)
_, test_data = load_csv("../data/MNIST/mnist_test.csv", sample_fraction=0.05)

# --- INICIO DE LA SOLUCIÓN ---

# 1. Inferir la dimensión de entrada (input_dim) desde los datos de entrenamiento
input_dim = train_data.shape[1]
print(f"Detectada dimensión de entrada (input_dim): {input_dim}")

# 2. Definir los parámetros base del Autoencoder, INCLUYENDO input_dim
base_ae_params = {
    "epochs": 30,
    "batch_size": 64,
    "input_dim": input_dim  # <--- AÑADIDO
}

# --- FIN DE LA SOLUCIÓN ---


autoencoders = [LinearAutoencoder, LinearSparseAutoencoder, DenoisingSparseAutoencoder]
manifolds = [TSNE, LocallyLinearEmbedding]

results = []

for ae_cls in autoencoders:
    for m_cls in manifolds:
        print(f"🔹 Probando combinación: {ae_cls.__name__} + {m_cls.__name__}")
        metrics = evaluate_combination(
            ae_cls, m_cls,
            train_data=train_data,
            test_data=test_data,
            ae_params=base_ae_params  # <--- USAMOS LOS PARÁMETROS BASE COMPLETOS
        )
        results.append(metrics)

# Añadimos comparación con manifold puro (sin autoencoder)
for m_cls in manifolds:
    print(f"🔹 Probando {m_cls.__name__} sin autoencoder")

    # --- INICIO DE LA SOLUCIÓN (CASO PURO) ---

    # 3. Corregir el 'dummy' autoencoder para que acepte input_dim
    #    El lambda debe aceptar kwargs y pasarlos al constructor
    #    Necesitamos instanciarlo (aunque con epochs=0) para que la interfaz funcione.
    dummy_ae_cls = lambda **kwargs: LinearAutoencoder(
        input_dim=kwargs.get('input_dim'),  # <--- Acepta input_dim
        epochs=0
    )

    # 4. Los parámetros para este caso también deben incluir input_dim
    dummy_ae_params = {"epochs": 0, "input_dim": input_dim}

    metrics = evaluate_combination(
        autoencoder_cls=dummy_ae_cls,  # <--- lambda corregido
        manifold_cls=m_cls,
        train_data=train_data,
        test_data=test_data,
        ae_params=dummy_ae_params  # <--- params corregidos
    )
    # --- FIN DE LA SOLUCIÓN (CASO PURO) ---

    metrics["autoencoder"] = "None"
    results.append(metrics)

os.makedirs("results", exist_ok=True)
save_results_to_csv(results, "results/exp_A_autoencoder_vs_manifold.csv")