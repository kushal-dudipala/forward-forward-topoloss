import matplotlib.pyplot as plt
import os

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
    
def plot_loss(loss_values, name="topological_loss"):
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

def plot_accuracy(train_acc_hist, val_acc_hist, name="accuracy"):
    """Plots training and validation accuracy over epochs."""
    plt.figure(figsize=(6, 4))
    epochs = list(range(len(train_acc_hist)))

    plt.plot(epochs, train_acc_hist, label="Train Accuracy", linewidth=2)
    plt.plot(epochs, val_acc_hist, label="Validation Accuracy", linestyle="dashed", linewidth=2)

    plt.axhline(y=90, color='gray', linestyle='dotted')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training & Validation Accuracy Over Training")
    plt.legend()
    filename = f"plots/{name}.png"
    plt.savefig(filename)
    plt.show()