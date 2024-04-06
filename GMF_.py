import torch
import torch.nn as nn

class GMF(nn.Module):
    def __init__(self, num_users, num_items, factor_num):
        super(GMF, self).__init__()
        self.embed_user_GMF = nn.Embedding(num_users, factor_num)
        self.embed_item_GMF = nn.Embedding(num_items, factor_num)
        self.predict_layer = nn.Linear(factor_num, 1)
        self._init_weight_()

    def _init_weight_(self):
        """Initialize weights"""
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')
        
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user, item):
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        output_GMF = embed_user_GMF * embed_item_GMF
        prediction = self.predict_layer(output_GMF)
        return prediction.view(-1)