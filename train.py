import torch
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from torch import nn
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


# print(len(train_dataset))
image, label = train_dataset[5]

# plt.imshow(image.squeeze(), cmap="gray")
# plt.savefig("mnist_sample.png")    // we can see the image through the matplotlib
images, labels = next(iter(train_loader))



flatten = nn.Flatten();
flattened_images = flatten(images);

linear1 = nn.Linear(784, 128);

relu = nn.ReLU();



class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__();
        self.flatten = nn.Flatten();
        self.linear1 = nn.Linear(784, 128);
        self.linear2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        return x

model = MNISTModel();
output = model(images);

loss_fn = nn.CrossEntropyLoss();

optimizer.zero_grad();
loss = loss_fn(output, labels);

loss.backward();

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3);


for epoch in range(5):
    for images, labels in train_loader:
        total_loss = 0;
        total_loss = total_loss + loss;
    #batch
    #zero grad
        optimizer.zero_grad(); 
    #prediction
        output = model(images);
      #loss
        loss = loss_fn(output, labels);
      #backward() -> gradients
        loss.backward(); 
        optimizer.step();  #step() - weights update



print(output.shape);
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