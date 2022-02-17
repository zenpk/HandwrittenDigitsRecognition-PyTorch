import matplotlib.pyplot as plt
from torchvision import datasets, transforms  # Training datasets
from torch.utils.data import DataLoader  # To input data more conveniently

# * Set up the datasets
batch_size = 32
train_data = datasets.MNIST(
    'data', train=True, download=False, transform=transforms.ToTensor())
train_loader = DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True)

# ! image[batch_size][1][y][x]

# * Show datasets (5*5)
checkdata = iter(train_loader)  # Iterator of the train_loader
image, label = next(checkdata)  # Next sets
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(image[i][0], cmap='gray')
    plt.axis('off')
plt.show()
