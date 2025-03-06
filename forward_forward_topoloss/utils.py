import torch
import matplotlib.pyplot as plt
import os

def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_

def visualize_sample(data, name='', idx=0):
    """Visualizes and saves a sample image."""
    
    os.makedirs("plots", exist_ok=True)
    filename = f"plots/{name}.png"
    if os.path.exists(filename):
        print(f"Skipping {filename}, already exists")
        return

    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize=(4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.savefig(filename)  
    print(f"Saved: {filename}")

    plt.show() 
def plot_loss(loss_values, name="topological_loss.png"):
    """Plots and saves the Topological Loss graph."""
    os.makedirs("plots", exist_ok=True)
    if not loss_values:
        print("Warning: No loss data to plot!")
        return
    
    plt.figure(figsize=(6, 4))
    for i, loss_curve in enumerate(loss_values):
        plt.plot(loss_curve, label=f'Layer {i+1} TL', linewidth=2)
    plt.axhline(y=0.1, color='gray', linestyle='dotted')
    plt.xlabel("Epochs")
    plt.ylabel("Topological Loss (TL)")
    plt.title("Topological Loss Over Training")
    plt.legend()
    filename = f"plots/{name}.png"
    plt.savefig(filename)
    plt.show()

def plot_accuracy(accuracy_history, name="accuracy.png"):
    """Plots and saves the Accuracy graph."""
    if not accuracy_history:  
        print("Warning: No accuracy data to plot!")
        return
    
    os.makedirs("plots", exist_ok=True)
    epochs = list(range(1, len(accuracy_history) + 1))  # Correct x-axis range
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, accuracy_history, label="Accuracy", linewidth=2)
    plt.axhline(y=90, color='gray', linestyle='dotted')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Training")
    plt.legend()
    filename = f"plots/{name}.png"
    plt.savefig(filename)
    plt.show()