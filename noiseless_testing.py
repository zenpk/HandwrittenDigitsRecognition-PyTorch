import torch
from numpy import zeros
from numpy import savetxt
from torch import nn  # Neural Network
from torchvision import datasets, transforms  # Testing datasets
from torch.utils.data import DataLoader  # To input data more conveniently

# * Hyperparameters
num_classes = 10  # 0~9
input_size = 784  # 28*28
hidden_layers = 64
torch.set_printoptions(precision=4)

# * Set up the datasets
# Get the datasets from MNIST
test_data = datasets.MNIST(
    "data", train=False, download=False, transform=transforms.ToTensor())
# Set up the DataLoader
test_loader = DataLoader(
    dataset=test_data, batch_size=1, shuffle=False)
# samples = numOfTestingData = 10000
samples = len(test_loader)

# * Define the model


class Recognizer(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes):
        super(Recognizer, self).__init__()
        self.input = nn.Linear(in_features=input_size,
                               out_features=hidden_layers)
        self.relu_1 = nn.ReLU()
        self.hidden = nn.Linear(
            in_features=hidden_layers, out_features=hidden_layers)
        self.relu_2 = nn.ReLU()
        self.output = nn.Linear(
            in_features=hidden_layers, out_features=num_classes)

    def forward(self, x):
        model = self.input(x)
        model = self.relu_1(model)
        model = self.hidden(model)
        model = self.relu_2(model)
        model = self.output(model)
        return model


model = torch.load("trained_model.pth")  # Load the trained model
accuracy_rec = zeros(samples)  # Record the accuracy
right_cnt = 0
for step, (image, label) in enumerate(test_loader):
    print(f"Testing the {step+1} image", end=", ")
    # Reshape to 2D tensor (image[batch_size][input_size])
    image = image.reshape(-1, input_size)
    output = model(image)  # Get the output tensor
    res = output.argmax()  # Get the maximum's index
    print(f"the number is {label.item()}, recognized as {res}", end=", ")
    if res == label.item():
        right_cnt += 1
        print("RIGHT", end=", ")
    else:
        print("WRONG", end=", ")
    accuracy = right_cnt/(step+1)  # Calculate the accuracy
    accuracy_rec[step] = accuracy
    print(f"Accuracy = {accuracy:.2f}")
savetxt("accuracy_noiseless.csv", accuracy_rec, delimiter=",")
