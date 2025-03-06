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
train_loader, test_loader, x_train, y_train, x_pos_train, x_neg_train, x_val, y_val = prepare_data()

net = Net([784, 500, 500, 10])  

for data, name in zip([x_train, x_pos_train, x_neg_train], ['orig', 'pos', 'neg']):
    visualize_sample(data, name)

# Train with validation tracking
net.train(x_pos_train, x_neg_train, y_train, x_val, y_val)

# Collect accuracy history
train_acc_hist, val_acc_hist = net.get_accuracy_history()

# Generate plots
plot_loss([layer.loss_history for layer in net.layers], name="topological_loss")
plot_accuracy(train_acc_hist, val_acc_hist, name="accuracy")



