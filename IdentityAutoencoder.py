import numpy as np

class IdentityAutoencoder(object):
    """
    Clase para usar como baseline. Cumple la interfaz Autoencoder pero no modifica los datos.
    Esto permite probar los algoritmos Manifold (TSNE, LLE) en solitario,
    ya que el algoritmo Manifold se aplica directamente sobre los datos de entrada originales.
    """
    def __init__(self, **kwargs):
        pass

    def fit(self, data: np.ndarray):
        """No realiza entrenamiento, por lo que el tiempo de fit será ~0."""
        pass

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Devuelve los datos de entrada sin modificar."""
        return data

    @property
    def __class____name__(self):
        # Nombre representativo para la tabla de resultados
        return "Identity"