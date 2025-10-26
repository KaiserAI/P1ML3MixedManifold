import numpy as np
import os
import pickle
import sys
import os  # Importamos os de nuevo por si acaso

# --- CONFIGURACIÓN DE ARCHIVOS ---
TRAIN_BATCH_FILES = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
TEST_BATCH_FILE = "raw_cifar10/test_batch"
OUTPUT_TRAIN_FILENAME = "cifar10_train.csv"  # Solo nombre
OUTPUT_TEST_FILENAME = "cifar10_test.csv"  # Solo nombre

# Determinamos la ruta absoluta del directorio del script (e.g., .../data/Cifar10/)
# Esto garantiza que las operaciones de guardado y lectura siempre usan la misma ruta.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def unpickle_batch(file_path):
    """Carga el archivo binario de un lote de CIFAR-10."""
    print(f"-> Cargando lote: {file_path}")
    # ... (restante del código de unpickle_batch es el mismo) ...
    try:
        with open(file_path, 'rb') as f:
            if sys.version_info[0] == 2:
                dict_data = pickle.load(f)
            else:
                dict_data = pickle.load(f, encoding='bytes')
        return dict_data
    except FileNotFoundError:
        print(f"   ⚠️ Advertencia: Archivo '{file_path}' no encontrado. Saltando.")
        return None
    except Exception as e:
        print(f"   🛑 Error al cargar el archivo {file_path}: {e}")
        return None


def verify_file_integrity(file_path: str, expected_rows: int):
    """Lee el archivo recién guardado y verifica si tiene el número de filas esperado."""
    try:
        # Se usa file_path (ruta absoluta) para leer el archivo.
        verif_data = np.genfromtxt(file_path, delimiter=',', dtype=np.float32)

        rows_written = 1 if verif_data.ndim == 1 else verif_data.shape[0]

        if rows_written == expected_rows:
            print(f"✅ VERIFICACIÓN EXITOSA: '{file_path}' tiene {rows_written} filas.")
        else:
            print(f"\n🛑 🛑 🛑 VERIFICACIÓN FALLIDA 🛑 🛑 🛑")
            print(f"   El archivo '{file_path}' solo tiene {rows_written} filas (Esperado {expected_rows}).")
            print("   La escritura fue interrumpida prematuramente.")

    except Exception as e:
        # Si falla la lectura, el archivo no existe en la ruta proporcionada.
        print(f"🛑 ERROR DE VERIFICACIÓN: No se pudo leer el archivo guardado.")
        # Usamos os.path.basename para dar un mensaje más claro
        print(f"   Detalle: {os.path.basename(file_path)} no se encontró o el acceso falló.")


def combine_and_save_train_data(batch_files):
    """Combina todos los lotes de entrenamiento y guarda el CSV final."""
    all_data = []
    all_labels = []

    for file_name in batch_files:
        batch_path = os.path.join(file_name)
        batch_dict = unpickle_batch(batch_path)

        if batch_dict is not None:
            data = batch_dict[b'data']
            labels = batch_dict[b'labels']
            all_data.append(data)
            all_labels.extend(labels)

    if not all_data:
        print("🛑 Error: No se pudo cargar ningún lote de entrenamiento. Abortando.")
        return

    # 1. Crear matriz final [y | X]
    X_train = np.vstack(all_data).astype(np.float32)
    y_train = np.array(all_labels, dtype=np.float32).reshape(-1, 1)
    train_csv_data = np.hstack((y_train, X_train))
    expected_rows = train_csv_data.shape[0]

    # 2. CALCULAR RUTA ABSOLUTA Y GUARDAR
    output_path_train = os.path.join(SCRIPT_DIR, OUTPUT_TRAIN_FILENAME)  # <-- RUTA ABSOLUTA
    np.savetxt(output_path_train, train_csv_data, delimiter=',', fmt='%.8f')

    print(f"\n✅ Datos de entrenamiento guardados en: {OUTPUT_TRAIN_FILENAME}")
    print(f"   Instancias totales: {expected_rows} (Esperado 50000).")

    # 3. VERIFICAR INTEGRIDAD USANDO RUTA ABSOLUTA
    verify_file_integrity(output_path_train, expected_rows)


def save_test_data(file_name):
    """Carga el lote de prueba y guarda el CSV."""
    batch_dict = unpickle_batch(file_name)

    if batch_dict is None:
        print("🛑 Error: No se pudo cargar el lote de prueba. Abortando.")
        return

    X_test = batch_dict[b'data'].astype(np.float32)
    y_test = np.array(batch_dict[b'labels'], dtype=np.float32).reshape(-1, 1)

    test_csv_data = np.hstack((y_test, X_test))
    expected_rows = test_csv_data.shape[0]

    # CALCULAR RUTA ABSOLUTA Y GUARDAR
    output_path_test = os.path.join(SCRIPT_DIR, OUTPUT_TEST_FILENAME)  # <-- RUTA ABSOLUTA
    np.savetxt(output_path_test, test_csv_data, delimiter=',', fmt='%.8f')

    print(f"\n✅ Datos de prueba guardados en: {OUTPUT_TEST_FILENAME}")
    print(f"   Instancias totales: {expected_rows} (Esperado 10000).")

    # VERIFICAR INTEGRIDAD
    verify_file_integrity(output_path_test, expected_rows)


if __name__ == '__main__':
    print("---------------------------------------------------------")
    print("PROCESADOR DE BATCHES CIFAR-10 CON VERIFICACIÓN DE I/O")
    print("---------------------------------------------------------")

    # Proceso de Entrenamiento
    combine_and_save_train_data(TRAIN_BATCH_FILES)

    # Proceso de Prueba
    save_test_data(TEST_BATCH_FILE)
