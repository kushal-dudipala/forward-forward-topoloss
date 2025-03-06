import torch
import torch.nn as nn
from torch.optim import Adam
from loss import TopoLoss, LaplacianPyramid
from utils import overlay_y_on_x
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, dims, checkpoint_path="checkpoints/model_checkpoint.pth"):
        super().__init__()
        self.layers = []
        self.layers = nn.ModuleList([
            Layer(dims[d], dims[d + 1], last_layer=(d == len(dims) - 2)).to(device)
            for d in range(len(dims) - 1)
        ])
        self.checkpoint_path = checkpoint_path
        self.accuracy_history = []
        
    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)] 
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, x_pos, x_neg):
        """Trains all layers and saves the model after training."""
        h_pos, h_neg = x_pos, x_neg
        acc = self.predict(x_pos).eq(torch.argmax(x_pos, dim=1)).float().mean().item() * 100
        self.accuracy_history.append(acc)
        
        for i, layer in enumerate(self.layers):
            print(f'Training layer {i}...')
            h_pos, h_neg = layer.train(h_pos, h_neg)
        
        self.save_checkpoint()
    
    def _generate_checkpoint_path(self, dims, lr):
        """Generates a filename based on the model hyperparameters."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        layer_sizes = "_".join(map(str, dims))  
        return f"{self.checkpoint_dir}/model_{layer_sizes}_lr{lr}.pth"
        
    def save_checkpoint(self):
        """Saves model state after training with a descriptive filename."""
        checkpoint = {
            "model_state": self.state_dict(),
            "accuracy_history": self.accuracy_history,
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Model saved: {self.checkpoint_path}")
    
    def load_checkpoint(self):
        """Loads model state if a checkpoint exists."""
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.load_state_dict(checkpoint["model_state"])
            self.accuracy_history = checkpoint["accuracy_history"]
            print(f"Loaded saved model: {self.checkpoint_path}")

    def get_accuracy_history(self):
        return self.accuracy_history

class Layer(nn.Linear):
    def __init__(self, in_features, out_features,last_layer=False, num_epochs = 1000,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = num_epochs
        self.loss_history = []  # Store loss per epoch
        self.last_layer = last_layer 
        
        if not last_layer:
            self.topo_loss_fn = TopoLoss(
                        losses=[LaplacianPyramid.from_layer(
                            model=self,
                            layer=self,  
                            factor_h=5.0, 
                            factor_w=5.0, 
                            scale=1.0 # This is the scale factor for the Laplacian Pyramid
                        )]
                    )
    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        """Trains this layer for `num_epochs` iterations."""
        for _ in tqdm(range(self.num_epochs), desc="Training Layer"):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)

            base_loss = torch.log(1 + torch.exp(torch.cat([-g_pos + self.threshold, g_neg - self.threshold]))).mean()

            if not self.last_layer:
                topo_loss_value = self.topo_loss_fn.compute(model=self)
                loss = base_loss + topo_loss_value
            else:
                loss = base_loss

            self.loss_history.append(loss.item())  # Save loss

            self.opt.zero_grad()
            # not a real backward pass
            loss.backward()
            self.opt.step()

        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


