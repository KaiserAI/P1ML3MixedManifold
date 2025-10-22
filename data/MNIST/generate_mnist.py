from torchvision import datasets, transforms
import pandas as pd
import numpy as np

# Transformación a tensores normalizados
transform = transforms.Compose([transforms.ToTensor()])

# Descargar datasets
train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

def save_to_csv(dataset, path):
    labels = np.array(dataset.targets)
    images = dataset.data.numpy().reshape(len(dataset), -1)  # (N, 28*28)
    df = pd.DataFrame(images)
    df.insert(0, "label", labels)
    df.to_csv(path, index=False)
    print(f"✅ Guardado: {path}")

save_to_csv(train_set, "mnist_train.csv")
save_to_csv(test_set, "mnist_test.csv")
