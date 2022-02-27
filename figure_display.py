import numpy as np
import matplotlib.pyplot as plt

# Loss graph (by batch)
loss_rec = np.genfromtxt("loss_rec.csv", delimiter=",")
fig1 = plt.figure()
plt.plot(loss_rec)
plt.title("Loss (by batch)")
plt.xlabel("Steps")
plt.ylabel("Value")
plt.show()

# Loss graph (by epoch)
loss_rec_epoch = np.genfromtxt("loss_rec_epoch.csv", delimiter=",")
fig2 = plt.figure()
plt.plot(loss_rec_epoch)
plt.title("Loss (by epoch)")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.show()

# Noiseless Accuracy graph
accuracy_noiseless = np.genfromtxt("accuracy_noiseless.csv", delimiter=",")
fig3 = plt.figure()
plt.plot(accuracy_noiseless)
plt.title("Noiseless Accuracy")
plt.xlabel("Steps")
plt.ylabel("Percentage")
plt.show()

# Noise Added Accuracy graph
accuracy_noise_added = np.genfromtxt("accuracy_noise_added.csv", delimiter=",")
noise_level = np.genfromtxt("noise_level.csv", delimiter=",")
fig4 = plt.figure()
plt.plot(noise_level, accuracy_noise_added)
x_step = 0.05
plt.xticks(np.arange(min(noise_level), max(
    noise_level)+x_step, x_step))
plt.title("Gaussian Noise Added Accuracy")
plt.xlabel("Noise Level (Variance)")
plt.ylabel("Percentage")
plt.show()
