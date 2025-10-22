"""
EXPERIMENTO B:
Evaluación de hiperparámetros del Autoencoder (épocas, lambda_sparse, embedding_dim)
usando un manifold fijo (TSNE).
"""

import os
from sklearn.manifold import TSNE
from LinearSparseAutoencoder import LinearSparseAutoencoder
from mixed_manifold_detector import load_csv
from experiment_utils import evaluate_combination, save_results_to_csv

labels, train_data = load_csv("../data/MNIST/mnist_train.csv", sample_fraction=0.05)
_, test_data = load_csv("../data/MNIST/mnist_test.csv", sample_fraction=0.05)

# --- INICIO DE LA SOLUCIÓN ---
# 1. Inferir la dimensión de entrada (input_dim)
input_dim = train_data.shape[1]
print(f"Detectada dimensión de entrada (input_dim): {input_dim}")
# --- FIN DE LA SOLUCIÓN ---

epochs_list = [30, 60]
lambda_list = [1e-4, 1e-3, 1e-2]
embedding_dims = [16, 32, 64]

results = []

for e in epochs_list:
    for lam in lambda_list:
        for dim in embedding_dims:
            print(f"🔹 epochs={e}, lambda={lam}, embedding_dim={dim}")
            ae_params = {
                "epochs": e,
                "lambda_sparse": lam,
                "embedding_dim": dim,
                "batch_size": 64,
                "input_dim": input_dim  # <--- ARREGLO AÑADIDO AQUÍ
            }
            metrics = evaluate_combination(
                LinearSparseAutoencoder, TSNE,
                train_data=train_data,
                test_data=test_data,
                ae_params=ae_params,
                manifold_params={"n_components": 2, "random_state": 42}
            )
            metrics.update(ae_params)
            results.append(metrics)

os.makedirs("results", exist_ok=True)
save_results_to_csv(results, "results/exp_B_hyperparams_autoencoder.csv")
