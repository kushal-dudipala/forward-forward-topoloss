import torch
from data import prepare_data
from model import Net
from visualizations import visualize_sample, plot_loss, plot_accuracy
from train import train_and_evaluate

'''
Main execution function.

'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)
train_loader, test_loader, x, y, x_pos, x_neg = prepare_data()

net = Net([784, 500, 500, 10])  

for data, name in zip([x, x_pos, x_neg], ['orig', 'pos', 'neg']):
    visualize_sample(data, name)

# Train and collect loss & accuracy (saves checkpoints)
loss_values, net = train_and_evaluate(net, x, y, x_pos, x_neg, test_loader)

# Collect accuracy history per layer
accuracy_histories = net.get_accuracy_history()

# Plot results
plot_loss([layer.loss_history for layer in net.layers], name="topological_loss")
plot_accuracy(accuracy_histories, name="accuracy")



