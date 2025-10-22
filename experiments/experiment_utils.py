import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness
from MixedManifoldDetector import MixedManifoldDetector
import os
from torch.utils.data import DataLoader  # Se usa en Autoencoder, solo para mostrar las dependencias si se usa aquí


# Función de Carga de Datos (implementada aquí para ser usada por todos los scripts)
def load_data(dataset_name: str, file_type: str, datasets_config: dict) -> Optional[np.ndarray]:
    """
    Carga los datos desde un archivo CSV y detecta automáticamente la dimensión de entrada.
    :param dataset_name: Nombre del dataset (e.g., 'MNIST').
    :param file_type: 'train' o 'test'.
    :param datasets_config: Diccionario con la configuración del dataset (BASE_DATA_PATH y DATASETS).
    :return: Matriz de numpy con los datos normalizados, o None si el archivo no existe.
    """
    ds_info = datasets_config["DATASETS"].get(dataset_name)
    base_path = datasets_config["BASE_DATA_PATH"]

    if not ds_info:
        print(f"⚠️ ERROR: Dataset '{dataset_name}' no encontrado en la configuración.")
        return None

    file_name = ds_info[f"{file_type}_file"]

    # Construcción de la ruta independiente del SO: BASE_DATA_PATH/DATASET_NAME/FILE_NAME
    full_path = os.path.join(base_path, dataset_name, file_name)

    print(f"Cargando {file_type} data desde: {full_path}...")
    try:
        # Cargamos el CSV. Asumimos que la primera columna es la etiqueta (y se excluye).
        df = pd.read_csv(full_path, header=None)
        # Excluir la primera columna (etiqueta) y obtener los valores
        data = df.iloc[:, 1:].values.astype(np.float32)

        # Normalización a [0, 1] si los datos están en el rango 0-255
        if data.max() > 1.0:
            data /= 255.0

        # El input_dim es data.shape[1]
        print(f"Datos cargados (Shape: {data.shape}, Input_dim detectado: {data.shape[1]})")
        return data

    except FileNotFoundError:
        print(f"⚠️ ERROR: Archivo no encontrado en {full_path}. Saltando {dataset_name}.")
        return None


def evaluate_combination(autoencoder_cls, manifold_cls, train_data, test_data=None,
                         ae_params=None, manifold_params=None, dataset_name: str = "Unknown"):
    ae_params = ae_params or {}
    manifold_params = manifold_params or {}
    input_dim = train_data.shape[1] if train_data is not None else 0

    # LÍNEA CORREGIDA: Se usa 'autoencoder_cls' en lugar de 'ae_cls'
    if 'input_dim' not in ae_params and autoencoder_cls.__name__ not in ["IdentityAutoencoder"]:
        ae_params['input_dim'] = input_dim

    ae = autoencoder_cls(**ae_params)
    manifold = manifold_cls(**manifold_params)

    detector = MixedManifoldDetector(autoencoder=ae, manifold_alg=manifold)

    # 2. Fit y Time Fit
    start_time = time.time()
    embedding_train = detector.fit_transform(train_data)
    train_time = time.time() - start_time

    # 3. Métrica de Trustworthiness (Train)
    tw_train = trustworthiness(train_data, embedding_train, n_neighbors=10)

    metrics = {
        "dataset": dataset_name,
        "autoencoder": ae.__class__.__name__,
        "manifold": manifold.__class__.__name__,
        "ae_params": str(ae_params),
        "manifold_params": str(manifold_params),
        "time_fit_s": train_time,
        "trustworthiness_train": tw_train
    }

    # 4. Transform y Time Transform
    if test_data is not None:
        start_time_transform = time.time()
        embedding_test = detector.transform(test_data)
        time_transform = time.time() - start_time_transform

        # Métrica de Trustworthiness (Test - Opcional)
        tw_test = trustworthiness(test_data, embedding_test, n_neighbors=10)

        metrics["time_transform_s"] = time_transform
        metrics["trustworthiness_test"] = tw_test

    return metrics


def save_results_to_csv(results: list, path: str):
    df = pd.DataFrame(results)
    order = ["dataset", "autoencoder", "manifold", "ae_params", "manifold_params",
             "time_fit_s", "time_transform_s", "trustworthiness_train", "trustworthiness_test"]
    cols = [c for c in order if c in df.columns]
    df = df[cols]
    df.to_csv(path, index=False)
    print(f"✅ Resultados guardados en {path}")
    return df