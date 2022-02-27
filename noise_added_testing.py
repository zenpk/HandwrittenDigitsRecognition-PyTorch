import torch
from numpy import zeros
from numpy import savetxt
from numpy import arange
from torch import nn  # Neural Network
from torchvision import datasets, transforms  # Testing datasets
from torch.utils.data import DataLoader  # To input data more conveniently

# * Hyperparameters
num_classes = 10  # 0~9
input_size = 784  # 28*28
hidden_layers = 64
noise_level = 0.5  # Maximum Gaussian noise level (~Variance)
noise_step = 0.02
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
cnt = int(noise_level / noise_step) + 1
accuracy_rec = zeros(cnt)  # Record the accuracy
noise_rec = zeros(cnt)  # Record the noise level
for noise in arange(0, noise_level+noise_step, noise_step):
    right_cnt = 0
    for step, (image, label) in enumerate(test_loader):
        # Reshape to 2D tensor (image[batch_size][input_size])
        image = image.reshape(-1, input_size)
        gauss = (noise**0.5)*torch.randn(28*28)  # Get Gaussian noise
        gauss = gauss.clamp(-1., 1.)  # Limit Gaussian noise to [-1,1]
        image += gauss  # Add Gaussian noise
        image = image.clamp(0., 1.)  # Limit image tensor to [0,1]
        output = model(image)  # Get the output tensor
        res = output.argmax()  # Get the maximum's index
        if res == label.item():
            right_cnt += 1
    accuracy = right_cnt/samples  # Calculate the accuracy
    index = int(noise/noise_step)
    accuracy_rec[index] = accuracy
    noise_rec[index] = noise
    print(f"Noise Level = {noise:.2f}, Accuracy = {accuracy:.2f}")
savetxt("accuracy_noise_added.csv", accuracy_rec, delimiter=",")
savetxt("noise_level.csv", noise_rec, delimiter=",")
