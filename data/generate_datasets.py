import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from typing import Dict, Any

# --- CONFIGURACIÓN DE DATASETS ---

# La ruta base asume que el script está dentro de la carpeta 'data'
BASE_DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Mapeo y nombres de archivos finales (basados en tu dataset_config.py)
DATASETS_CONFIG: Dict[str, Dict[str, Any]] = {
    "MNIST": {
        "openml_id": "mnist_784",
        "folder": "MNIST",
        "train_filename": "mnist_train.csv",  # Archivo de entrenamiento
        "test_filename": "mnist_test.csv",  # Archivo de prueba
        "has_header": True
    },
    "FashionMNIST": {
        "openml_id": "Fashion-MNIST",
        "folder": "FashionMNIST",
        "train_filename": "fashion_mnist_train.csv",
        "test_filename": "fashion_mnist_test.csv",
        "has_header": True
    },
    "Cifar10": {
        "openml_id": "CIFAR_10",
        "folder": "Cifar10",
        "train_filename": "cifar10_train.csv",
        "test_filename": "cifar10_test.csv",
        "has_header": True
    },
    "GlassIdentification": {
        "openml_id": "glass",
        "folder": "GlassIdentification",
        "train_filename": "glass_train.csv",
        "test_filename": "glass_test.csv",
        "has_header": False
    }
}


# --- FUNCIÓN PRINCIPAL ---
def load_split_and_save_dataset(config: Dict[str, Any]):
    """
    Descarga el dataset de OpenML, lo preprocesa, lo divide en 80/20 (Train/Test)
    y guarda los dos archivos CSV por separado.
    """
    name = config["openml_id"]
    folder = config["folder"]
    train_filename = config["train_filename"]
    test_filename = config["test_filename"]
    has_header = config["has_header"]

    output_dir = os.path.join(BASE_DATA_DIR, folder)
    train_path = os.path.join(output_dir, train_filename)
    test_path = os.path.join(output_dir, test_filename)

    print(f"🔹 Procesando dataset: {folder} (ID: {name})...")

    # 1. Cargar datos
    try:
        data = fetch_openml(name=str(name), version="active", as_frame=True, return_X_y=False)
    except Exception as e:
        print(f"Error al cargar {name} de OpenML: {e}")
        return

    X = data.data.values.astype(np.float32)
    y = data.target.values

    # 2. Conversión de tipos y Normalización

    # Normalización Min-Max a [0, 1] por característica
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    denominador = max_vals - min_vals
    denominador[denominador == 0] = 1.0
    X = (X - min_vals) / denominador

    # Aseguramos que las etiquetas sean enteras
    if y.dtype.kind not in ('i', 'f'):
        y_encoded, _ = pd.factorize(y)
        y = y_encoded.astype(int)
    else:
        y = y.astype(int)

    # 3. División Train (80%) / Test (20%) con estratificación
    # La estratificación asegura que la proporción de clases se mantenga en ambos splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Combinar X e y y preparar para guardar
    # np.insert inserta y (etiquetas) como la primera columna (label, f1, f2, ...)
    train_data_to_save = np.insert(X_train, 0, y_train, axis=1)
    test_data_to_save = np.insert(X_test, 0, y_test, axis=1)

    # Definir la cabecera si se requiere
    header_list = None
    if has_header:
        header_list = ['label'] + [f'f{i}' for i in range(X.shape[1])]

    # Asegurar que el directorio exista
    os.makedirs(output_dir, exist_ok=True)

    # 5. Guardar archivos
    # Guardar TRAIN
    pd.DataFrame(train_data_to_save).to_csv(train_path, index=False, header=header_list)
    print(f"  ✅ TRAIN (80%) guardado: {train_filename} ({X_train.shape[0]} muestras)")

    # Guardar TEST
    pd.DataFrame(test_data_to_save).to_csv(test_path, index=False, header=header_list)
    print(f"  ✅ TEST (20%) guardado: {test_filename} ({X_test.shape[0]} muestras)")


# --- EJECUCIÓN ---
if __name__ == "__main__":
    print(f"La ruta base de guardado será: {os.path.abspath(BASE_DATA_DIR)}\n")

    for config in DATASETS_CONFIG.values():
        load_split_and_save_dataset(config)

    print("\nProceso de generación de datasets completado. Los archivos train/test están listos.")
