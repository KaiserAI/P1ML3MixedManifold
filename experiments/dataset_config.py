# dataset_config.py (dentro de experiments/)
import os

# Obtener la ruta absoluta del directorio actual (experiments/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# La carpeta de datos está un nivel arriba (..) desde la carpeta actual (experiments/)
# Construcción de la ruta absoluta: /ProyectoRaíz/data
BASE_DATA_PATH = os.path.join(CURRENT_DIR, "..", "data")

# El resto de la configuración permanece igual
DATASETS = {
    "MNIST": {
        "train_file": "mnist_train.csv",
        "test_file": "mnist_test.csv",
        "has_header": True
    },
    "FashionMNIST": {
        "train_file": "fashion_mnist_train.csv",
        "test_file": "fashion_mnist_test.csv",
        "has_header": True
    },
    "Cifar10": {
        "train_file": "cifar10_train.csv",
        "test_file": "cifar10_test.csv",
        "has_header": True
    },
    "GlassIdentification": {
        "train_file": "glass_train.csv",
        "test_file": "glass_test.csv",
        "has_header": False
    }
}