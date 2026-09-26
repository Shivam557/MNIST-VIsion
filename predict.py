import torch
from torchvision.datasets import MNIST
from torchvision import transforms

from train import MNISTModel


# Load the trained model
model = MNISTModel()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()


# Load test data
convertTensor = transforms.ToTensor()

test_dataset = MNIST(
    root="./data",
    train=False,
    download=True,
    transform=convertTensor
)


# Get one image
image, label = test_dataset[0]

# Add batch dimension
image = image.unsqueeze(0)


# Make prediction
with torch.no_grad():
    output = model(image)
    prediction = output.argmax(dim=1).item()


print(f"Predicted: {prediction}")
print(f"Actual: {label}")