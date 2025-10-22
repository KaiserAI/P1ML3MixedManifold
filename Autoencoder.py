import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List


# Definición de un Dataset simple para PyTorch
class NumpyDataset(Dataset):
    def __init__(self, data: np.ndarray):
        # Convertir datos a tensor de PyTorch (float32 es estándar)
        self.data = torch.from_numpy(data).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Para autoencoders, la entrada es la misma que la salida
        return self.data[idx], self.data[idx]


class Autoencoder(object):
    """
    Clase base que encapsula un modelo torch.nn.Module para el Autoencoder.
    Debe ser implementada por clases concretas como LinearAutoencoder.
    """

    def __init__(self,
                 epochs: int = 100,
                 error_threshold: float = 0.0,
                 batch_size: int = 64,  # Valor por defecto común para batch_size
                 learning_rate: float = 1e-3,  # Parámetro opcional añadido para la optimización
                 embedding_dim: int = 32,  # Dimensión del espacio latente.
                 **kwargs):
        """
        Constructor del Autoencoder.
        :param epochs: Número de épocas de entrenamiento (Defecto: 100).
        :param error_threshold: Umbral de error para detener el entrenamiento (Defecto: 0.0).
        :param batch_size: Tamaño del batch para el entrenamiento.
        :param learning_rate: Tasa de aprendizaje para el optimizador.
        :param embedding_dim: Dimensión del embedding latente.
        """
        self.epochs = epochs
        self.error_threshold = error_threshold
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim

        # --- INICIO DE LA SOLUCIÓN (1/3) ---
        # Definir el dispositivo (CPU o GPU) en la clase base
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # --- FIN DE LA SOLUCIÓN (1/3) ---

        self.model: Optional[nn.Module] = None  # Almacena el objeto torch.nn.Module
        self.encoder: Optional[nn.Sequential] = None  # Almacena solo el encoder para 'transform'
        self.is_fitted = False

    def fit(self, data: np.ndarray):
        """
        Entrena el autoencoder con los datos de entrada.
        Internamente, configura la función de pérdida y el optimizador.
        :param data: Matriz de numpy con patrones de entrenamiento.
        """
        if self.model is None:
            raise NotImplementedError("El modelo (self.model) debe ser definido en la subclase.")

        # --- INICIO DE LA SOLUCIÓN (2/3) ---
        # Mover el modelo al dispositivo correcto ANTES de crear el optimizador
        self.model.to(self.device)
        # --- FIN DE LA SOLUCIÓN (2/3) ---

        # Configuración estándar: MSELoss y Adam (ambos son opcionales pero se usan comúnmente)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Preparación de datos
        dataset = NumpyDataset(data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        # (Opcional) Informar al usuario qué dispositivo se está usando
        print(f"Iniciando entrenamiento en {self.device} por {self.epochs} épocas...")

        for epoch in range(self.epochs):
            total_loss = 0
            for inputs, targets in dataloader:
                # --- INICIO DE LA SOLUCIÓN (2/3) ---
                # Mover los datos del batch al dispositivo correcto
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # --- FIN DE LA SOLUCIÓN (2/3) ---

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass y optimización
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")

            # Condición de parada por umbral de error
            if self.error_threshold > 0 and avg_loss < self.error_threshold:
                print(
                    f"Entrenamiento detenido: Pérdida ({avg_loss:.6f}) por debajo del umbral ({self.error_threshold})")
                break

        self.is_fitted = True
        self.model.eval()  # Poner el modelo en modo evaluación al finalizar

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Devuelve los embeddings (representación latente) de los patrones de entrada.
        :param data: Matriz de numpy con patrones.
        :return: Embeddings (matriz de numpy d-dimensional).
        """
        if not self.is_fitted:
            raise RuntimeError("Transform solo se puede invocar después de fit.")
        if self.encoder is None:
            raise NotImplementedError("El encoder (self.encoder) debe ser definido en la subclase.")

        # --- INICIO DE LA SOLUCIÓN (3/3) ---
        # Asegurarse de que el encoder está en el dispositivo correcto
        self.encoder.to(self.device)
        # --- FIN DE LA SOLUCIÓN (3/3) ---
        self.encoder.eval()  # Asegurar que el encoder está en modo evaluación

        # Preparación de datos para la inferencia
        with torch.no_grad():
            # --- INICIO DE LA SOLUCIÓN (3/3) ---
            # Crear el tensor y MOVERLO al dispositivo correcto
            tensor_data = torch.from_numpy(data).float().to(self.device)
            # --- FIN DE LA SOLUCIÓN (3/3) ---

            # Asegurar que el encoder solo procesa un batch a la vez, si es necesario.
            # Aquí se procesa todo el tensor directamente para simplicidad.
            embeddings_tensor = self.encoder(tensor_data)

        # Convertir el tensor de embeddings a numpy
        return embeddings_tensor.cpu().numpy()
