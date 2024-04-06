import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, user_size, item_size, hidden_layers, output_size):
        super(MLP, self).__init__()
        self.user_embedding = nn.Embedding(user_size, hidden_layers[0] // 2)
        self.item_embedding = nn.Embedding(item_size, hidden_layers[0] // 2)
        
        self.mlp_layers = nn.ModuleList()
        for i in range(1, len(hidden_layers)):
            self.mlp_layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        
        self.output_layer = nn.Linear(hidden_layers[-1], output_size)
        
    def forward(self, user_indices, item_indices):
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)
        x = torch.cat([user_embedding, item_embedding], dim=1)
        
        for layer in self.mlp_layers:
            x = F.relu(layer(x))
        
        x = torch.sigmoid(self.output_layer(x))
        return x.squeeze()