import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
# Importamos la clase base y el dataset personalizado
from Autoencoder import Autoencoder, NumpyDataset


# La clase ahora hereda de Autoencoder
class DenoisingSparseAutoencoder(Autoencoder):
    """
    Autoencoder lineal con regularización Denoising + Sparse.
    Cumple la interfaz Autoencoder exigida en la práctica:
        - __init__(epochs, error_threshold, batch_size)
        - fit(data)
        - transform(data)
    """

    def __init__(self, input_dim: int, epochs: int = 100, error_threshold: float = 0.0, batch_size: int = 64,
                 lr: float = 1e-3, lambda_sparse: float = 1e-3, noise_factor: float = 0.2,
                 embedding_dim: int = 32, **kwargs):
        """
        Constructor del autoencoder con regularizaciones Denoising y Sparse.

        Parámetros:
        - input_dim: Dimensión de la entrada (número de características).
        - epochs: número máximo de épocas de entrenamiento.
        - error_threshold: umbral de pérdida para parada anticipada.
        - batch_size: tamaño del mini-lote.
        - lr: tasa de aprendizaje del optimizador Adam.
        - lambda_sparse: peso del término de regularización L1 sobre el embedding.
        - noise_factor: intensidad del ruido gaussiano aplicado a la entrada (Denoising).
        - embedding_dim: tamaño de la capa latente (por defecto 32).
        """
        # Llamamos al constructor de la clase base
        # Nota: Pasamos 'lr' como 'learning_rate' a la clase base
        super().__init__(epochs=epochs, error_threshold=error_threshold, batch_size=batch_size,
                         learning_rate=lr, embedding_dim=embedding_dim, **kwargs)

        # Parámetros específicos de esta subclase
        self.lambda_sparse = lambda_sparse
        self.noise_factor = noise_factor

        # --- INICIO DE LA SOLUCIÓN ---
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # <-- ESTA LÍNEA SE ELIMINA (se hereda de Autoencoder)
        # --- FIN DE LA SOLUCIÓN ---

        # La lógica de _build_model se mueve aquí, al constructor
        # para definir self.model y self.encoder como espera la clase base.

        class DenoisingSparseModel(nn.Module):
            def __init__(self, input_dim, embedding_dim):
                super().__init__()
                # Encoder: 3 capas lineales → embedding
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, embedding_dim)
                )
                # Decoder: simétrico
                self.decoder = nn.Sequential(
                    nn.Linear(embedding_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, input_dim)
                )

            def forward(self, x):
                z = self.encoder(x)
                x_rec = self.decoder(z)
                return x_rec, z

        # Definimos self.model y self.encoder
        # Usamos self.device (heredado) para mover el modelo
        self.model = DenoisingSparseModel(input_dim, self.embedding_dim).to(self.device)
        self.encoder = self.model.encoder
        # self.is_fitted será establecido por fit()

    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Añade ruido gaussiano a la entrada (Denoising).
        """
        noise = torch.randn_like(x) * self.noise_factor
        # Aseguramos que el ruido se añade en el dispositivo correcto
        x_noisy = x.to(self.device) + noise.to(self.device)
        # Clampeamos a [0, 1] para evitar valores fuera de rango si los datos están normalizados
        return torch.clamp(x_noisy, 0., 1.)

    def fit(self, data: np.ndarray):
        """
        Entrena el autoencoder usando MSE + regularización L1 sobre el embedding
        y añade ruido gaussiano a las entradas (Denoising).
        (Este método SOBRESCRIBE el 'fit' base para añadir la lógica de ruido y sparse).
        """
        # Comprobación de dimensiones
        if data.shape[1] != self.model.decoder[-1].out_features:
            raise ValueError(
                f"La dimensión de entrada de los datos ({data.shape[1]}) no coincide con la del modelo ({self.model.decoder[-1].out_features}).")

        # Usamos el NumpyDataset de la clase base
        dataset = NumpyDataset(data)
        # Usamos el self.batch_size de la clase base
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Usamos self.learning_rate de la clase base (que se asignó desde 'lr')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        # Mover el modelo al dispositivo (ya se hizo en __init__, pero no hace daño)
        self.model.to(self.device)
        self.model.train()
        print(f"Iniciando entrenamiento (DenoisingSparse) en {self.device} por {self.epochs} épocas...")

        # Usamos self.epochs de la clase base
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            # NumpyDataset devuelve (inputs, targets)
            for inputs, targets in loader:
                # Mover datos al dispositivo
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()

                # Denoising: entrada ruidosa
                noisy_batch = self._add_noise(inputs)

                # Forward pass
                recon, z = self.model(noisy_batch)

                # Pérdida de reconstrucción (comparando con la entrada original limpia)
                recon_loss = loss_fn(recon, targets)
                # Regularización Sparse (L1 sobre el embedding)
                sparse_loss = torch.mean(torch.abs(z))

                # Pérdida total
                loss = recon_loss + self.lambda_sparse * sparse_loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)

            mean_loss = epoch_loss / len(dataset)
            print(f"Epoch {epoch:03d}/{self.epochs} - Loss: {mean_loss:.6f}")

            # Usamos self.error_threshold de la clase base
            if self.error_threshold > 0 and mean_loss <= self.error_threshold:
                print(f"✅ Early stopping: error {mean_loss:.6f} <= {self.error_threshold}")
                break

        # Establecemos los flags de la clase base al finalizar
        self.is_fitted = True
        self.model.eval()

    # El método 'transform' ya no es necesario.
    # Se utilizará el de la clase base 'Autoencoder',
    # que funciona gracias a que 'self.encoder' y 'self.is_fitted'
    # se han definido correctamente.