import torch
from numpy import zeros
from numpy import savetxt
from torch import nn  # Neural Network
from torch import optim  # Optimizer
from torchvision import datasets, transforms  # Training datasets
from torch.utils.data import DataLoader  # To input data more conveniently

# * Hyperparameters
num_epochs = 10  # Training loop times
num_classes = 10  # 0~9
learning_rate = 0.01  # Determine how much W will change during each loop
batch_size = 32
input_size = 784  # 28*28
hidden_layers = 64  # Determine how manny nodes in the hidden network
torch.set_printoptions(precision=4)

# * Setup the datasets
# Get the datasets from MNIST
if_download = False
train_data = datasets.MNIST(
    'data', train=True, download=if_download, transform=transforms.ToTensor())
# Setup the DataLoader
train_loader = DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True)
# samples = numOfTrainingData/batch_size = 60000/32 = 1875
samples = len(train_loader)

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


model = Recognizer(input_size, hidden_layers, num_classes)

# * Define the optimiser
criterion = nn.CrossEntropyLoss()
params = model.parameters()
optimizer = optim.SGD(params, lr=learning_rate)  # Stochastic Gradient Descent

# * Training loops
loss_rec = zeros(num_epochs*samples)
loss_rec_epoch = zeros(num_epochs)
for epoch in range(num_epochs):
    loss_list = list()
    for step, (image, label) in enumerate(train_loader):
        # Reshape to 2D tensor (image[batch_size][input_size])
        image = image.reshape(-1, 28*28)
        output = model(image)  # Forward
        # loss: the average of the variance between output and label
        loss = criterion(output, label)
        loss_rec[epoch*samples+step] = loss.item()
        loss_list.append(loss.item())
        optimizer.zero_grad()  # Empty the gradients
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        print(f'Epoch {epoch+1}/{num_epochs}', end=', ')
        print(f'step: {step+1}/{samples}', end=', ')
        print(f'training loss: {loss.item():.4f}')
    loss_rec_epoch[epoch] = torch.tensor(
        loss_list).mean()  # The average loss of the epoch
torch.save(model, 'trained_model.pth')  # Save the model
# Save the loss record (by batch)
savetxt('loss_rec.csv', loss_rec, delimiter=',')
# Save the loss record (by epoch)
savetxt('loss_rec_epoch.csv', loss_rec_epoch, delimiter=',')
