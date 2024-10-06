import torch
import time
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

batch_size = 64  # Example batch size
dataloader = DataLoader(datasets, batch_size=batch_size)

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

dataloader = DataLoader(training_data, batch_size=batch_size)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using ", device)

zero_time = time.time()
model = NeuralNetwork().to(device)
model.eval()

start_time = time.time()
total_samples = 0
with torch.no_grad():
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        total_samples += X.size(0)

if device.type == 'cuda':
    torch.cuda.synchronize() # Make sure all operations are finished

end_time = time.time()
elapsed_time = end_time - start_time
throughput = total_samples / elapsed_time

print(f"Total samples : {total_samples}")
print(f"Throughput: {throughput:.2f} samples/second")
print(f"Latency: {end_time - zero_time}")
