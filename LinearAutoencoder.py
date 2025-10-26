import torch.nn as nn
import numpy as np
from Autoencoder import Autoencoder


class LinearAutoencoder(Autoencoder):
    """
    Implementa la interfaz Autoencoder con un modelo PyTorch
    de 3 capas lineales en el encoder, embedding de 32 neuronas, y 3 capas en el decoder.
    """

    def __init__(self, input_dim: int, **kwargs):
        """
        Constructor del autoencoder lineal.

        Parámetros:
        - input_dim: Dimensión de la entrada (número de características).
        """
        # Llama al constructor de la clase base, pasando los argumentos adicionales
        super().__init__(**kwargs)

        # Definición de la dimensión latente
        self.embedding_dim = 32

        # Dimensiones intermedias para las 3 capas lineales
        h1 = max(128, input_dim // 4)
        h2 = max(64, input_dim // 8)

        # Encoder (3 capas lineales → embedding)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, self.embedding_dim)  # Capa final de 32 neuronas
        )

        # Decoder (simétrico)
        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, input_dim)  # Reconstrucción a la dimensión original
        )

        # Modelo Completo (nn.Module)
        # La clase Autoencoder encapsula torch.nn.Module
        self.model = self._LinearAutoencoderModule(self.encoder, self.decoder)

    # Definición interna del módulo PyTorch
    class _LinearAutoencoderModule(nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, x):
            z = self.encoder(x)
            x_reconstructed = self.decoder(z)
            return x_reconstructed

    def fit(self, data: np.ndarray):
        """
        Entrena el autoencoder.

        Parámetros:
        - data: Datos de entrenamiento (numpy array).
        """
        # La dimensión de entrada (input_dim) debe coincidir con la de los datos
        if data.shape[1] != self.model.decoder[-1].out_features:
            raise ValueError(
                f"La dimensión de entrada de los datos ({data.shape[1]}) no coincide con la salida del decoder ({self.model.decoder[-1].out_features}).")

        # Llama al método fit de la clase base (Autoencoder)
        super().fit(data)
