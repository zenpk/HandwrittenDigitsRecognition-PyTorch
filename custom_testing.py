import torch
import matplotlib.pyplot as plt
from torch import nn  # Neural Network
from torchvision import datasets, transforms  # Testing datasets
from torch.utils.data import DataLoader  # To input data more conveniently

# * Hyperparameters
num_classes = 10  # 0~9
input_size = 784  # 28*28
hidden_layers = 64
noise = 0.1  # ~Variance
torch.set_printoptions(precision=4)

# * Setup the datasets
# Get the datasets from 'test' folder
# Transform from the RGB to Garyscale
data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                     transforms.ToTensor()])
test_data = datasets.ImageFolder('test', transform=data_transform)
# Setup the DataLoader
test_loader = DataLoader(
    dataset=test_data, batch_size=1, shuffle=False)
samples = len(test_loader)  # 10

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


model = torch.load('trained_model.pth')  # Load the trained model
right_cnt = 0
it_loader = iter(test_loader)  # Iterator of the test_loader
for image, label in test_loader:
    print(f'Testing the image {label.item()}', end=', ')
    plt_image, plt_label = next(it_loader)  # Next sets
    plt.subplot(1, 2, 1)
    plt.imshow(plt_image[0][0], cmap='gray')
    plt.axis('off')
    plt.title('Original Image')
    plt_gauss = (noise**0.5)*torch.randn(28, 28)  # Get Gaussian noise
    plt_gauss = plt_gauss.clamp(-1., 1.)  # Limit Gaussian noise to [-1,1]
    plt_image[0][0] += plt_gauss  # Add Gaussian noise to the image
    plt_image[0][0] = plt_image[0][0].clamp(
        0., 1.)  # Limit image tensor to [0,1]
    plt.subplot(1, 2, 2)
    plt.imshow(plt_image[0][0], cmap='gray')
    plt.axis('off')
    plt.title('Noise Added Image')
    plt.show()
    # Reshape to 2D tensor (image[batch_size][input_size])
    image = image.reshape(-1, input_size)
    gauss = (noise**0.5)*torch.randn(28*28)  # Get Gaussian noise
    gauss = gauss.clamp(-1., 1.)  # Limit Gaussian noise to [-1,1]
    image += gauss  # Add Gaussian noise to the tensor
    mage = image.clamp(0., 1.)  # Limit image tensor to [0,1]
    output = model(image)  # Get the output tensor
    res = output.argmax()  # Get the maximum's index
    if(res == label.item()):
        right_cnt += 1
    print(f'recognized as {res}')
accuracy = right_cnt/10  # Calculate the accuracy
print(f'Accuracy: {accuracy:.2f}')
