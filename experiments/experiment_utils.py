import time
import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness
from MixedManifoldDetector import MixedManifoldDetector


def evaluate_combination(autoencoder_cls, manifold_cls, train_data, test_data=None,
                         ae_params=None, manifold_params=None):
    """
    Entrena una combinación Autoencoder + Manifold y devuelve métricas básicas.
    """
    ae_params = ae_params or {}
    manifold_params = manifold_params or {}

    ae = autoencoder_cls(**ae_params)
    manifold = manifold_cls(**manifold_params)

    detector = MixedManifoldDetector(autoencoder=ae, manifold_alg=manifold)

    start_time = time.time()
    embedding_train = detector.fit_transform(train_data)
    train_time = time.time() - start_time

    tw_train = trustworthiness(train_data, embedding_train, n_neighbors=10)

    metrics = {
        "autoencoder": ae.__class__.__name__,
        "manifold": manifold.__class__.__name__,
        "time_fit": train_time,
        "trustworthiness_train": tw_train
    }

    if test_data is not None:
        embedding_test = detector.transform(test_data)
        tw_test = trustworthiness(test_data, embedding_test, n_neighbors=10)
        metrics["trustworthiness_test"] = tw_test

    return metrics


def save_results_to_csv(results: list, path: str):
    """
    Guarda una lista de diccionarios con métricas en un archivo CSV.
    """
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    print(f"✅ Resultados guardados en {path}")
    return df
