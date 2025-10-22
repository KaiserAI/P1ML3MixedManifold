import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.base import TransformerMixin
from LinearAutoencoder import LinearAutoencoder


class MixedManifoldDetector:
    def __init__(self, autoencoder=None, manifold_alg: TransformerMixin = None):
        """
        Inicializa el detector mixto con un autoencoder y un algoritmo de manifold.

        Parámetros:
        - autoencoder: objeto con interfaz fit(data) y transform(data).
          Si no se especifica, se creará una instancia por defecto de LinearAutoencoder.
        - manifold_alg: algoritmo de manifold learning de sklearn.
          Si no se especifica, se usará TSNE.
        """
        # Importación perezosa para evitar dependencias circulares
        if autoencoder is None:
            autoencoder = LinearAutoencoder()

        self.autoencoder = autoencoder
        self.manifold_alg = manifold_alg if manifold_alg is not None else TSNE(n_components=2)
        self.train_embeddings = None
        self.train_data = None
        self.train_latent = None

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Entrena el autoencoder y el manifold, devolviendo la representación 2D.
        """
        # Entrenar autoencoder
        self.autoencoder.fit(data)
        latent = self.autoencoder.transform(data)

        # Aplicar manifold
        embedding_2d = self.manifold_alg.fit_transform(latent)

        # Guardar datos de entrenamiento
        self.train_data = data
        self.train_embeddings = embedding_2d
        self.train_latent = latent

        return embedding_2d

    def fit(self, train_data: np.ndarray):
        """
        Entrena el modelo (autoencoder + manifold).
        """
        self.fit_transform(train_data)

    def transform(self, data: np.ndarray, k: int = 5) -> np.ndarray:
        """
        Transforma nuevos datos a 2D. Si el dato es nuevo, interpola según los k vecinos más cercanos.
        """
        if self.train_data is None or self.train_embeddings is None:
            raise RuntimeError("El modelo debe entrenarse primero con fit().")

        latent_new = self.autoencoder.transform(data)

        embeddings_out = []
        for sample in latent_new:
            distances = np.linalg.norm(self.train_latent - sample, axis=1)
            if np.min(distances) == 0:
                # El dato ya existe en el entrenamiento
                idx = np.argmin(distances)
                embeddings_out.append(self.train_embeddings[idx])
            else:
                # Interpolación por vecinos más cercanos
                nearest_idx = np.argsort(distances)[:k]
                nearest_embeds = self.train_embeddings[nearest_idx]
                nearest_dist = distances[nearest_idx]
                weights = 1 / (nearest_dist + 1e-8)
                weights /= np.sum(weights)
                weighted_avg = np.average(nearest_embeds, axis=0, weights=weights)
                embeddings_out.append(weighted_avg)

        return np.array(embeddings_out)
