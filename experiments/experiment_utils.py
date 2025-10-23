import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness
from MixedManifoldDetector import MixedManifoldDetector
import os

# from torch.utils.data import DataLoader # Se mantiene comentado para limpieza


# --- MÍNIMO REQUERIDO: 21 muestras (para que Trustworthiness con k=10 no falle) ---
MIN_SAMPLES_REQUIRED = 21


def load_data(dataset_name: str, file_type: str, datasets_config: dict, sample_ratio: float = 1.0) -> Optional[
    np.ndarray]:
    """
    Carga los datos, aplicando muestreo. Si el muestreo genera menos de 21 instancias,
    se carga el 100% de los datos para evitar fallos de TSNE/Trustworthiness.
    """
    ds_info = datasets_config["DATASETS"].get(dataset_name)
    base_path = datasets_config["BASE_DATA_PATH"]

    if not ds_info:
        print(f"⚠️ ERROR: Dataset '{dataset_name}' no encontrado en la configuración.")
        return None

    file_name = ds_info[f"{file_type}_file"]

    header_present = ds_info.get("has_header", False)
    skip_rows = 1 if header_present else 0
    print(f"   -> has_header: {header_present}. Saltando {skip_rows} fila(s).")

    full_path = os.path.join(base_path, dataset_name, file_name)

    print(f"Cargando {file_type} data desde: {full_path}...")
    try:
        # Carga del CSV con manejo de encabezados
        df = pd.read_csv(full_path, header=None, skiprows=skip_rows)
        data_full = df.iloc[:, 1:].values.astype(np.float32)

        # Normalización a [0, 1] si los datos están en el rango 0-255
        if data_full.max() > 1.0:
            data_full /= 255.0

        total_samples = data_full.shape[0]
        desired_samples = int(total_samples * sample_ratio)

        # --- SEGURO DE TAMAÑO (Implementando la lógica solicitada) ---
        current_sample_ratio = sample_ratio

        if desired_samples < MIN_SAMPLES_REQUIRED and sample_ratio < 1.0:
            current_sample_ratio = 1.0
            print(
                f"⚠️ Alerta de muestreo: El sample ratio ({sample_ratio}) produciría menos de {MIN_SAMPLES_REQUIRED} instancias.")
            print(f"   Sobrescribiendo sample_ratio a 1.0. Se usará la BBDD completa ({total_samples} muestras).")
        elif sample_ratio == 1.0 and total_samples < MIN_SAMPLES_REQUIRED:
            # Caso donde la BBDD completa es ya muy pequeña (ej. Glass sin split)
            print(f"⚠️ Alerta de tamaño: La BBDD completa solo tiene {total_samples} muestras.")

        # --- Lógica de MUESTREO final ---
        if current_sample_ratio < 1.0:
            num_samples = int(total_samples * current_sample_ratio)
            indices = np.random.choice(total_samples, size=num_samples, replace=False)
            data = data_full[indices]
            print(f"Muestreo aplicado: {current_sample_ratio * 100:.1f}%. Usando {num_samples} muestras.")
        else:
            data = data_full

        print(f"Datos cargados (Shape: {data.shape})")
        return data

    except FileNotFoundError:
        print(f"⚠️ ERROR: Archivo no encontrado en {full_path}. Saltando {dataset_name}.")
        return None


def evaluate_combination(autoencoder_cls, manifold_cls, train_data, test_data=None,
                         ae_params=None, manifold_params=None, dataset_name: str = "Unknown"):
    ae_params = ae_params or {}
    manifold_params = manifold_params or {}

    input_dim = train_data.shape[1] if train_data is not None else 0
    n_train_samples = train_data.shape[0]

    # --- NO SE APLICAN AJUSTES DINÁMICOS DE HIPERPARÁMETROS ---
    # Los hiperparámetros como perplexity (TSNE) y n_neighbors (Trustworthiness)
    # se mantendrán fijos (30 y 10, respectivamente) o los definidos por el usuario.
    # El fallo se evita asegurando que n_train_samples >= 21 en load_data.

    # 1. Asignar INPUT_DIM al AE
    if 'input_dim' not in ae_params and autoencoder_cls.__name__ not in ["IdentityAutoencoder"]:
        ae_params['input_dim'] = input_dim

    # Inicialización
    ae = autoencoder_cls(**ae_params)
    manifold = manifold_cls(**manifold_params)

    detector = MixedManifoldDetector(autoencoder=ae, manifold_alg=manifold)

    # 2. Fit y Time Fit
    start_time = time.time()
    embedding_train = detector.fit_transform(train_data)
    train_time = time.time() - start_time

    # 3. Métrica de Trustworthiness (Train)
    # Usamos n_neighbors=10 (el valor por defecto que causó problemas)
    # Si load_data hizo bien su trabajo, n_train_samples >= 21, y esto funcionará.
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

    # 4. Transform y Time Transform (Test)
    if test_data is not None:
        # El conjunto de test también debe tener al menos 21 muestras para el trustworthiness.
        # Asumimos que si el train set es grande, el test set también lo será,
        # o que el usuario ha configurado un SAMPLE_PERCENTAGE adecuado para test.

        start_time_transform = time.time()
        embedding_test = detector.transform(test_data)
        time_transform = time.time() - start_time_transform

        # Métrica de Trustworthiness (Test - usando n_neighbors=10)
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
