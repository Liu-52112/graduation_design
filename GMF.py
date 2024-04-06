import torch
import torch.nn as nn

class GMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(GMF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.affine_layer = nn.Linear(embedding_dim, out_features=1)
        self.activation = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)
        interaction = torch.mul(user_embedding , item_embedding)
        logs = self.affine_layer(interaction)
        prediction = self.activation(logs)
        return prediction.squeeze()
    
    def _init_weight_(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.kaiming_uniform_(self.affine_layer.weight, a=1, nonlinearity='sigmoid')
        
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()