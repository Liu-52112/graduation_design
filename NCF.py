import torch
import torch.nn as nn
import torch.nn.functional as F

class NCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers, dropout):
        super(NCF, self).__init__()
        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(user_num, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(item_num, factor_num * (2 ** (num_layers - 1)))

        MLP_layers = []
        input_size = factor_num * 2  # 用户和物品嵌入的拼接大小
        for i in range(num_layers):
            MLP_layers.append(nn.Dropout(p=dropout))
            MLP_layers.append(nn.Linear(input_size, input_size // 2))
            MLP_layers.append(nn.ReLU())
            input_size = input_size // 2
        self.MLP_layers = nn.Sequential(*MLP_layers)

        # 融合后的向量大小为GMF的factor_num加上MLP的最后一层输出大小
        self.predict_layer = nn.Linear(factor_num + input_size, 1)
        
    def forward(self, user, item):
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        output_GMF = embed_user_GMF * embed_item_GMF

        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        input_MLP = torch.cat((embed_user_MLP, embed_item_MLP), dim=-1)
        output_MLP = self.MLP_layers(input_MLP)

        concat = torch.cat((output_GMF, output_MLP), dim=-1)
        
        prediction = torch.sigmoid(self.predict_layer(concat))
        return prediction.squeeze()