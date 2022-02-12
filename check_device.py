# * Check if current device support CUDA
from torch import device, cuda
this_device = device('cuda' if cuda.is_available() else 'cpu')
print(this_device)
