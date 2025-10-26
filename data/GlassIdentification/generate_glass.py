import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# --- CONFIGURACIÓN DE ARCHIVOS ---
DATA_FILE = "raw_glass/glass.data"
OUTPUT_TRAIN_FILE = "glass_train.csv"
OUTPUT_TEST_FILE = "glass_test.csv"
TEST_SIZE_RATIO = 0.20  # 20% para prueba
RANDOM_STATE = 42  # Semilla para split reproducible


def prepare_glass_dataset():
    print(f"Iniciando preparación del dataset Glass Identification...")

    # 1. Cargar los datos brutos. El archivo no tiene encabezados y usa comas.
    try:
        # El archivo 'glass.data' tiene 11 columnas:
        # [ID, RI, Na, Mg, Al, Si, K, Ca, Ba, Fe, Type]
        df = pd.read_csv(DATA_FILE, header=None)
    except FileNotFoundError:
        print(f"🛑 ERROR: Archivo '{DATA_FILE}' no encontrado. Asegúrate de que esté en el directorio correcto.")
        return

    print(f"Datos originales cargados (Filas: {df.shape[0]}, Columnas: {df.shape[1]})")

    # 2. Limpieza y Separación de Características (X) y Etiquetas (y)

    # Columna de Etiqueta (y): Es la última columna (índice 10)
    # NOTA: Los tipos de clase están numerados del 1 al 7, pero las clases 4 (vehículos no procesados)
    # no existen en el dataset. Las etiquetas no son consecutivas (1, 2, 3, 5, 6, 7).
    y = df.iloc[:, 10].values  # La columna 11 (índice 10) es el Tipo de vidrio

    # Características (X): Son las columnas del índice 1 al 9 (9 características)
    # Columna 0 (ID) debe ser ELIMINADA.
    X = df.iloc[:, 1:10].values

    # 3. Conversión de tipos y aseguramiento de float
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # 4. Dividir el dataset en entrenamiento y prueba (80/20)
    # Usamos stratify=y para asegurar que la proporción de cada tipo de vidrio se mantenga en ambos splits.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE_RATIO, random_state=RANDOM_STATE, stratify=y
    )

    # 5. Formatear y guardar los CSV (y | X)

    # Conjunto de Entrenamiento
    y_train_reshaped = y_train.reshape(-1, 1)
    train_csv_data = np.hstack((y_train_reshaped, X_train))

    # Conjunto de Prueba
    y_test_reshaped = y_test.reshape(-1, 1)
    test_csv_data = np.hstack((y_test_reshaped, X_test))

    # 6. Guardar archivos con rutas absolutas
    output_path_train = os.path.join(OUTPUT_TRAIN_FILE)
    output_path_test = os.path.join(OUTPUT_TEST_FILE)

    np.savetxt(output_path_train, train_csv_data, delimiter=',', fmt='%.8f')
    np.savetxt(output_path_test, test_csv_data, delimiter=',', fmt='%.8f')

    print(f"✅ Datos de entrenamiento guardados en: {output_path_train} (Shape: {train_csv_data.shape})")
    print(f"✅ Datos de prueba guardados en: {output_path_test} (Shape: {test_csv_data.shape})")
    print("El dataset Glass ya está listo para ser usado en los experimentos.")


if __name__ == '__main__':
    prepare_glass_dataset()
