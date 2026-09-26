
import torch
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn


# -------------------------
# 1. Dataset
# -------------------------

convertTensor = transforms.ToTensor()

train_dataset = MNIST(
    root="./data",
    train=True,
    download=True,
    transform=convertTensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)


# -------------------------
# 2. Model
# -------------------------

class MNISTModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x


# -------------------------
# 3. Model, Loss, Optimizer
# -------------------------

if __name__ == "__main__":

    model = MNISTModel()

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )


    # -------------------------
    # 4. Training
    # -------------------------

    for epoch in range(5):

        total_loss = 0

        for images, labels in train_loader:

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(images)

            # Calculate loss
            loss = loss_fn(output, labels)

            # Add this batch's loss
            total_loss += loss.item()

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

        # Average loss for this epoch
        average_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch + 1}, Loss: {average_loss:.4f}")


    # -------------------------
    # 5. Test Dataset
    # -------------------------

    test_dataset = MNIST(
        root="./data",
        train=False,
        download=True,
        transform=convertTensor
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )


    # -------------------------
    # 6. Testing / Accuracy
    # -------------------------

    correct = 0

    with torch.no_grad():

        for images, labels in test_loader:

            output = model(images)

            predictions = output.argmax(dim=1)

            correct += (predictions == labels).sum().item()


    total = len(test_dataset)

    accuracy = correct / total * 100

    print(f"Test Accuracy: {accuracy:.2f}%")


    # -------------------------
    # 7. Test 5 Different Images
    # -------------------------

    for i in range(5):

        image, label = test_dataset[i]

        image = image.unsqueeze(0)

        with torch.no_grad():

            output = model(image)

        prediction = output.argmax(dim=1).item()

        print(
            f"Image {i + 1}: "
            f"Predicted = {prediction}, "
            f"Actual = {label}"
        )


    # -------------------------
    # 8. Save Model
    # -------------------------

    torch.save(model.state_dict(), "mnist_model.pth")


# print(output.shape);
# print(images.shape);
# print(flattened_images.shape);
# print(convertTensor)


#some concepts we need
# loss
# ↓
# tensor(0.8472)
# ↓
# loss.item()
# ↓
# 0.8472
