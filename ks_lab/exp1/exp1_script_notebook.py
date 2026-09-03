# %% [markdown]
# Experiment 1: Implement MLP and CNN for handwritten digit classification. Compare the results in terms of accuracy, F1 score. Plot train vs validation curve. Report the test accuracy and F1 score

# %% [markdown]
# Use MNIST dataset. Split into train, val and test sets in the ratio 60: 20:20. Before split sufffle the data.

# %% [markdown]
# Tune hyperparameters : learning rate, epoch, batch size.

# %%
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import numpy as np

train_ds = datasets.MNIST(root="./data", train=True, download=True)
test_ds  = datasets.MNIST(root="./data", train=False, download=True)

X_data = np.concatenate([train_ds.data.numpy(), test_ds.data.numpy()], axis=0)
Y_data = np.concatenate([train_ds.targets.numpy(), test_ds.targets.numpy()], axis=0)

# %%
epochs = 100
learning_rate = 0.00001
batch_size = 10

# %%

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

# %%
# Check sizes
print("Train:", X_train.shape, Y_train.shape)
print("Validation:", X_val.shape, Y_val.shape)
print("Test:", X_test.shape, Y_test.shape)

# %%
x_train_norm = X_train.astype("float32")/255
x_test_norm = X_test.astype("float32")/255
x_val_norm = X_val.astype("float32")/255

# %%
import matplotlib.pyplot as plt

# %%
fig, ax = plt.subplots(10, 10)
k = 0
for i in range(10):
    for j in range(10):
        ax[i][j].imshow(x_train_norm[k].reshape(28, 28), aspect='auto')
        k += 1
plt.show()

# %%
import torch
import torch.nn as nn

print(torch.__version__)
print(torch.cuda.is_available())

# %%
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

print(model)

# %%
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

model = model.to(device)

# %%
# Convert NumPy arrays to PyTorch tensors

X_train_tensor = torch.tensor(x_train_norm, dtype=torch.float32)
X_val_tensor   = torch.tensor(x_val_norm, dtype=torch.float32)
X_test_tensor  = torch.tensor(x_test_norm, dtype=torch.float32)

Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
Y_val_tensor   = torch.tensor(Y_val, dtype=torch.long)
Y_test_tensor  = torch.tensor(Y_test, dtype=torch.long)

# %%
X_train_tensor = X_train_tensor.reshape(-1, 784)
X_val_tensor   = X_val_tensor.reshape(-1, 784)
X_test_tensor  = X_test_tensor.reshape(-1, 784)

# %%
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset   = TensorDataset(X_val_tensor, Y_val_tensor)
test_dataset  = TensorDataset(X_test_tensor, Y_test_tensor)

# %%
# batch_size = 10

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# %%
criterion = nn.CrossEntropyLoss()

# %%
# learning_rate = 0.00001

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)

# %%
# epochs = 100

train_losses_mlp = []
val_losses_mlp = []

train_accuracies_mlp = []
val_accuracies_mlp = []

for epoch in range(epochs):

    # -------------------------
    # Training
    # -------------------------
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, Y_batch in train_loader:

        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        # Forward pass
        outputs = model(X_batch)

        # Calculate loss
        loss = criterion(outputs, Y_batch)

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

        # Predictions
        _, predicted = torch.max(outputs, 1)

        total += Y_batch.size(0)
        correct += (predicted == Y_batch).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total

    # -------------------------
    # Validation
    # -------------------------
    model.eval()

    running_val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for X_batch, Y_batch in val_loader:

            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            outputs = model(X_batch)

            loss = criterion(outputs, Y_batch)

            running_val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += Y_batch.size(0)
            correct += (predicted == Y_batch).sum().item()

    val_loss = running_val_loss / len(val_loader)
    val_accuracy = correct / total

    # Store results
    train_losses_mlp.append(train_loss)
    val_losses_mlp.append(val_loss)

    train_accuracies_mlp.append(train_accuracy)
    val_accuracies_mlp.append(val_accuracy)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.plot(train_losses_mlp, label="Train Loss")
plt.plot(val_losses_mlp, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MLP: Training vs Validation Loss")

plt.legend()
plt.grid()
plt.show()

# %%
plt.figure(figsize=(8, 5))

plt.plot(train_accuracies_mlp, label="Train Accuracy")
plt.plot(val_accuracies_mlp, label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MLP: Training vs Validation Accuracy")

plt.legend()
plt.grid()
plt.show()

# %%
from sklearn.metrics import accuracy_score, f1_score

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for X_batch, Y_batch in test_loader:

        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        outputs = model(X_batch)

        _, predictions = torch.max(outputs, 1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(Y_batch.cpu().numpy())

# %%
test_accuracy_mlp = accuracy_score(
    all_labels,
    all_predictions
)

test_f1_mlp = f1_score(
    all_labels,
    all_predictions,
    average="weighted"
)

print(f"MLP Test Accuracy: {test_accuracy_mlp:.4f}")
print(f"MLP Test F1 Score: {test_f1_mlp:.4f}")

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(all_labels, all_predictions)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=range(10),
    yticklabels=range(10)
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("MLP Confusion Matrix")

plt.show()

# %%
model = nn.Sequential(
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

# %%
# Convert NumPy arrays to PyTorch tensors

X_train_tensor = torch.tensor(x_train_norm, dtype=torch.float32)
X_val_tensor   = torch.tensor(x_val_norm, dtype=torch.float32)
X_test_tensor  = torch.tensor(x_test_norm, dtype=torch.float32)

Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
Y_val_tensor   = torch.tensor(Y_val, dtype=torch.long)
Y_test_tensor  = torch.tensor(Y_test, dtype=torch.long)

# %%
# Add channel dimension for CNN

X_train_tensor = X_train_tensor.unsqueeze(1)
X_val_tensor   = X_val_tensor.unsqueeze(1)
X_test_tensor  = X_test_tensor.unsqueeze(1)

# %%
print(X_train_tensor.shape)
print(X_val_tensor.shape)
print(X_test_tensor.shape)

# %%
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset   = TensorDataset(X_val_tensor, Y_val_tensor)
test_dataset  = TensorDataset(X_test_tensor, Y_test_tensor)

# %%
# batch_size = 10

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# %%
criterion = nn.CrossEntropyLoss()

# %%
# learning_rate = 0.0001

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)

# %%
# epochs = 100

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

for epoch in range(epochs):

    # -------------------------
    # Training
    # -------------------------
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, Y_batch in train_loader:

        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        # Forward pass
        outputs = model(X_batch)

        # Calculate loss
        loss = criterion(outputs, Y_batch)

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

        # Predictions
        _, predicted = torch.max(outputs, 1)

        total += Y_batch.size(0)
        correct += (predicted == Y_batch).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total

    # -------------------------
    # Validation
    # -------------------------
    model.eval()

    running_val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for X_batch, Y_batch in val_loader:

            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            outputs = model(X_batch)

            loss = criterion(outputs, Y_batch)

            running_val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += Y_batch.size(0)
            correct += (predicted == Y_batch).sum().item()

    val_loss = running_val_loss / len(val_loader)
    val_accuracy = correct / total

    # Store results
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MLP: Training vs Validation Loss")

plt.legend()
plt.grid()
plt.show()

# %%
plt.figure(figsize=(8, 5))

plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MLP: Training vs Validation Accuracy")

plt.legend()
plt.grid()
plt.show()

# %%
from sklearn.metrics import accuracy_score, f1_score

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for X_batch, Y_batch in test_loader:

        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        outputs = model(X_batch)

        _, predictions = torch.max(outputs, 1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(Y_batch.cpu().numpy())

# %%
test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)

test_f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted"
)

print(f"MLP Test Accuracy: {test_accuracy:.4f}")
print(f"MLP Test F1 Score: {test_f1:.4f}")

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(all_labels, all_predictions)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=range(10),
    yticklabels=range(10)
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("MLP Confusion Matrix")

plt.show()

# %%
# ============================================
# MLP vs CNN - Final Comparison
# ============================================

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --------------------------------------------
# Store CNN results using clear names
# --------------------------------------------

test_accuracy_cnn = test_accuracy
test_f1_cnn = test_f1

# --------------------------------------------
# Comparison table
# --------------------------------------------

print("=" * 50)
print("        MLP vs CNN TEST RESULTS")
print("=" * 50)

print(f"{'Metric':<20} {'MLP':>12} {'CNN':>12}")
print("-" * 50)

print(
    f"{'Test Accuracy':<20} "
    f"{test_accuracy_mlp:>11.4f} "
    f"{test_accuracy_cnn:>11.4f}"
)

print(
    f"{'Test F1 Score':<20} "
    f"{test_f1_mlp:>11.4f} "
    f"{test_f1_cnn:>11.4f}"
)

print("=" * 50)

# Percentage form

print("\nPercentage Results:")
print(f"MLP Test Accuracy : {test_accuracy_mlp * 100:.2f}%")
print(f"MLP Test F1 Score : {test_f1_mlp * 100:.2f}%")
print(f"CNN Test Accuracy : {test_accuracy_cnn * 100:.2f}%")
print(f"CNN Test F1 Score : {test_f1_cnn * 100:.2f}%")

# %%
# ============================================
# Accuracy and F1 Score Comparison
# ============================================

models = ["MLP", "CNN"]

accuracy_values = [
    test_accuracy_mlp,
    test_accuracy_cnn
]

f1_values = [
    test_f1_mlp,
    test_f1_cnn
]

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(
    x - width/2,
    accuracy_values,
    width,
    label="Accuracy"
)

plt.bar(
    x + width/2,
    f1_values,
    width,
    label="F1 Score"
)

plt.xticks(x, models)
plt.ylabel("Score")
plt.title("MLP vs CNN: Test Accuracy and F1 Score")
plt.ylim(0.95, 1.05)

plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.show()

# %%
# ============================================
# Training vs Validation Accuracy
# MLP vs CNN
# ============================================

plt.figure(figsize=(9, 6))

plt.plot(
    range(1, len(train_accuracies_mlp) + 1),
    train_accuracies_mlp,
    label="MLP Train Accuracy"
)

plt.plot(
    range(1, len(val_accuracies_mlp) + 1),
    val_accuracies_mlp,
    label="MLP Validation Accuracy"
)

plt.plot(
    range(1, len(train_accuracies) + 1),
    train_accuracies,
    label="CNN Train Accuracy"
)

plt.plot(
    range(1, len(val_accuracies) + 1),
    val_accuracies,
    label="CNN Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MLP vs CNN: Training and Validation Accuracy")

plt.legend()
plt.grid()

plt.show()

# %%
# ============================================
# Training vs Validation Loss
# MLP vs CNN
# ============================================

plt.figure(figsize=(9, 6))

plt.plot(
    range(1, len(train_losses_mlp) + 1),
    train_losses_mlp,
    label="MLP Train Loss"
)

plt.plot(
    range(1, len(val_losses_mlp) + 1),
    val_losses_mlp,
    label="MLP Validation Loss"
)

plt.plot(
    range(1, len(train_losses) + 1),
    train_losses,
    label="CNN Train Loss"
)

plt.plot(
    range(1, len(val_losses) + 1),
    val_losses,
    label="CNN Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MLP vs CNN: Training and Validation Loss")

plt.legend()
plt.grid()

plt.show()


