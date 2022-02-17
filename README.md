# HandwrittenDigitsRecognition-PyTorch
**A basic handwritten-digits recognition project based on PyTorch using datasets from MNIST**

> Data resource: [**MNIST**](http://yann.lecun.com/exdb/mnist/)
>
> Referenced code: 
>
> - [Pytorch Tutorials - 9 -Handwritten Digit Recognition using MNIST Digit Recognition for Deep Learning](https://www.youtube.com/watch?v=uGNel1qPrxo&list=LLTkz85dtpbZf_S4eaIQYrUQ) (**Author: Akash Bhiwgade**)
>
> - [Episode 1: Training a classification model on MNIST with PyTorch](https://www.youtube.com/watch?v=OMDn66kM9Qc&list=LLTkz85dtpbZf_S4eaIQYrUQ&index=2) (**Author: PyTorch Lightning**) 

Sincerely express my gratitude to the authors above. If this repository infringed anyone's copyright, please let me know so that I will delete and apologize at once. Thank you!

## Brief Introduction

This project trained a shallow neural network model using PyTorch. It has some basic functions, including "output the test result", "print the accuracy graph", "add noise to the source image", etc.

You can customize the parameters of the model, such as the number of epochs, nodes, layers, batch_size and learning_rate. The hyperparameters used in this project are learned from the referenced code linked above.

## Python Files Explanation

`check_device.py`: If your device support CUDA, it will print 'cuda', otherwise it will print 'cpu'. You can add CUDA integration to this project to improve the speed. By default, this project is based on CPU.

`check_datasets.py`: Show 25 random images from the datasets. You can change the number if you want.

`model_training.py`: Build the training model. A trained model has already been included in the repository, but you can rebuild it if you want. The resource datasets are in the 'data' folder.
After execution, you will get a trained model (trainedModel.pth), a record of loss by batch (loss_rec.csv) and a record of loss by epoch (loss_rec_epoch.csv).

`noiseless_testing.py`: Test the model using the test_data without noise. It will show the output result and accuracy, and you will get a record of the accuracy (accuracy_noiseless.csv).

`noise_added_testing.py`: Test the model using the test_data with Gaussian noise added. You can change the level of the noise by editing the 'noise_level'. It will show the accuracy of each level of the noise, and you will get a record of it (accuracy_noise_added.csv & noise_level.csv)

`figure_display.py`: Print the graphs from the data generated above. fig1: Loss (by batch), fig2: Loss (by epoch), fig3: noiseless_accuracy, fig4: noise_added_accuracy.

`custom_testing.py`: Test the handwritten digits created by yourself to see the result (put the images in the 'test' folder).

## Problems to Be Solved

- [x] 1. The value of the tensor will exceed [0,1] after adding the Gaussian noise, which can cause some unwanted influence. (Use torch.clamp() to solve it)
- [x] 2. The accuracy of 'custom test' is worse than the accuracy of the MNIST test datasets, why? (Solved after the 1. was solved)
- [ ] 3. Unwanted zeros appear in the 'noise_level.csv' and 'accuracy_noise_added.csv' when the noise_level set above 0.58. Bug? 

If you have any questions, please feel free to ask me, it's my pleasure to help.

***Thanks for reading!***
