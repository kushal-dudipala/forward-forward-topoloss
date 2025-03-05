from utils import plot_loss, plot_accuracy

def train_and_evaluate(device, net, x, y, x_pos, x_neg, test_loader):
    """Train the model, compute accuracy, and generate plots."""
    # Train and collect loss values
    net.train(x_pos, x_neg)
    loss_values = [layer.loss_history for layer in net.layers]  # Get loss history

    # Compute train accuracy
    train_accuracy = net.predict(x).eq(y).float().mean().item() * 100
    print(f'Train Accuracy: {train_accuracy:.2f}%')

    # Compute test accuracy
    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.to(device), y_te.to(device)
    test_accuracy = net.predict(x_te).eq(y_te).float().mean().item() * 100
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    # Generate plots
    plot_loss(loss_values, filename="topological_loss.png")
    plot_accuracy(net.accuracy_history, filename="accuracy.png")