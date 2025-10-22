"""
EXPERIMENTO C:
Evaluación de hiperparámetros de los algoritmos de Manifold Learning
usando un autoencoder fijo (LinearAutoencoder).
"""

import os
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from LinearAutoencoder import LinearAutoencoder
from mixed_manifold_detector import load_csv
from experiment_utils import evaluate_combination, save_results_to_csv

labels, train_data = load_csv("../data/MNIST/mnist_train.csv", sample_fraction=0.05)
_, test_data = load_csv("../data/MNIST/mnist_test.csv", sample_fraction=0.05)

# --- INICIO DE LA SOLUCIÓN ---
# 1. Inferir la dimensión de entrada (input_dim)
input_dim = train_data.shape[1]
print(f"Detectada dimensión de entrada (input_dim): {input_dim}")

# 2. Definir los parámetros fijos del Autoencoder (incluyendo input_dim)
base_ae_params = {
    "epochs": 30,
    "batch_size": 64,
    "input_dim": input_dim  # <--- ARREGLO AÑADIDO AQUÍ
}
# --- FIN DE LA SOLUCIÓN ---

results = []

# TSNE: variando perplexity y learning_rate
for perplexity in [10, 30, 50]:
    for lr in [100, 200, 500]:
        print(f"🔹 TSNE(perplexity={perplexity}, lr={lr})")
        manifold_params = {"n_components": 2, "perplexity": perplexity, "learning_rate": lr, "random_state": 42}
        metrics = evaluate_combination(
            LinearAutoencoder, TSNE,
            train_data=train_data,
            test_data=test_data,
            ae_params=base_ae_params,  # <--- Usamos los params corregidos
            manifold_params=manifold_params
        )
        metrics.update(manifold_params)
        results.append(metrics)

# LLE: variando n_neighbors
for n in [5, 10, 20, 50]:
    print(f"🔹 LLE(n_neighbors={n})")
    manifold_params = {"n_components": 2, "n_neighbors": n}
    metrics = evaluate_combination(
        LinearAutoencoder, LocallyLinearEmbedding,
        train_data=train_data,
        test_data=test_data,
        ae_params=base_ae_params,  # <--- Usamos los params corregidos
        manifold_params=manifold_params
    )
    metrics.update(manifold_params)
    results.append(metrics)

os.makedirs("results", exist_ok=True)
save_results_to_csv(results, "results/exp_C_hyperparams_manifold.csv")
