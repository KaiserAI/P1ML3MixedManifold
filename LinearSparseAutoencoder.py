import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from Autoencoder import Autoencoder, NumpyDataset


class LinearSparseAutoencoder(Autoencoder):
    """
    Autoencoder lineal con regularización Sparse.
    """

    def __init__(self, input_dim: int, epochs: int = 100, error_threshold: float = 0.0, batch_size: int = 64,
                 lr: float = 1e-3, lambda_sparse: float = 1e-3, embedding_dim: int = 32, **kwargs):
        """
        Constructor del autoencoder con regularización sparse.

        Parámetros:
        - input_dim: Dimensión de la entrada (número de características).
        - epochs: número máximo de épocas de entrenamiento.
        - error_threshold: umbral de pérdida para parada anticipada.
        - batch_size: tamaño del mini-lote.
        - lr: tasa de aprendizaje del optimizador Adam.
        - lambda_sparse: peso del término de regularización L1 sobre el embedding.
        - embedding_dim: tamaño de la capa latente (por defecto 32).
        """
        super().__init__(epochs=epochs, error_threshold=error_threshold, batch_size=batch_size,
                         learning_rate=lr, embedding_dim=embedding_dim, **kwargs)

        # Parámetros específicos de esta subclase
        self.lambda_sparse = lambda_sparse


        class SparseAutoencoderModel(nn.Module):
            def __init__(self, input_dim, embedding_dim):
                super().__init__()
                # Encoder: 3 capas lineales → embedding (32 neuronas)
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, embedding_dim)
                )
                # Decoder: 3 capas lineales simétricas
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
        self.model = SparseAutoencoderModel(input_dim, self.embedding_dim).to(self.device)
        self.encoder = self.model.encoder

    def fit(self, data: np.ndarray):
        """
        Entrena el autoencoder usando MSE + regularización L1 sobre el embedding.
        (Este método SOBRESCRIBE el 'fit' base para añadir la lógica de sparse).
        """
        # Comprobación de dimensiones
        if data.shape[1] != self.model.decoder[-1].out_features:
            raise ValueError(
                f"La dimensión de entrada de los datos ({data.shape[1]}) no coincide con la del modelo ({self.model.decoder[-1].out_features}).")

        dataset = NumpyDataset(data)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        # Mover el modelo al dispositivo
        self.model.to(self.device)
        self.model.train()
        print(f"Iniciando entrenamiento (LinearSparse) en {self.device} por {self.epochs} épocas...")

        # Usamos self.epochs de la clase base
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            # NumpyDataset devuelve (inputs, targets)
            for inputs, targets in loader:
                # Mover datos al dispositivo
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                recon, z = self.model(inputs)

                # Pérdida de reconstrucción
                recon_loss = loss_fn(recon, targets)
                # Regularización Sparse: L1 sobre las activaciones del embedding
                sparse_loss = torch.mean(torch.abs(z))
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
