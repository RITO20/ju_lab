# Experiment 1:
#     Implement MLP and CNN for handwritten digit classification.
#     Compare the results in terms of accuracy, F1 score.
#     Plot train vs validation curve. Report the test accuracy and F1 score
#     Use MNIST dataset. Split into train, val and test sets in the ratio 60: 20:20. Before split sufffle the data.
#     Tune hyperparameters : learning rate, epoch, batch size.



import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import torch
import torch.nn as nn

epochs = 100
learning_rate = 0.00001
batch_size = 10

#Download data 
train_ds = datasets.MNIST(root="./data", train=True, download=True)
test_ds  = datasets.MNIST(root="./data", train=False, download=True)

X_data = np.concatenate([train_ds.data.numpy(), test_ds.data.numpy()], axis=0)
Y_data = np.concatenate([train_ds.targets.numpy(), test_ds.targets.numpy()], axis=0)

#split data

# First split: 60% train, 40% temporary
X_train, X_temp, Y_train, Y_temp = train_test_split(
    X_data,
    Y_data,
    test_size=0.4,
    random_state=42,
    shuffle=True
)

# Second split: divide the remaining 40% equally
X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp,
    Y_temp,
    test_size=0.5,
    random_state=45,
    shuffle=True
)

# Check sizes
print("Train:", X_train.shape, Y_train.shape)
print("Validation:", X_val.shape, Y_val.shape)
print("Test:", X_test.shape, Y_test.shape)

#MLP Model
mlp_model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

#CNN Model
cnn_model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),

    nn.Linear(64 * 7 * 7, 128),
    nn.ReLU(),

    nn.Linear(128, 10)
)