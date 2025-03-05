import torch
import matplotlib.pyplot as plt

def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_

def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize=(4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()

def plot_loss(loss_values, filename="topological_loss.png"):
    """Plots and saves the Topological Loss graph."""
    plt.figure(figsize=(6, 4))
    for i, loss_curve in enumerate(loss_values):
        plt.plot(loss_curve, label=f'Layer {i+1} TL', linewidth=2)
    plt.axhline(y=0.1, color='gray', linestyle='dotted')
    plt.xlabel("Epochs")
    plt.ylabel("Topological Loss (TL)")
    plt.title("Topological Loss Over Training")
    plt.legend()
    plt.savefig(filename)
    plt.show()

def plot_accuracy(accuracy_history, filename="accuracy.png"):
    """Plots and saves the Accuracy graph."""
    plt.figure(figsize=(6, 4))
    plt.plot(accuracy_history, label="Accuracy", linewidth=2)
    plt.axhline(y=90, color='gray', linestyle='dotted')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Training")
    plt.legend()
    plt.savefig(filename)
    plt.show()
