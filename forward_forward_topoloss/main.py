import torch
from data import prepare_data
from model import Net
from utils import visualize_sample
from train import train_and_evaluate

"""Main execution function."""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)

train_loader, test_loader, x, y, x_pos, x_neg = prepare_data(device)

# Initialize model
net = Net([784, 500, 500])

# Visualize samples
for data, name in zip([x, x_pos, x_neg], ['orig', 'pos', 'neg']):
    visualize_sample(data, name)

# Train and evaluate
train_and_evaluate(device, net, x, y, x_pos, x_neg, test_loader)



