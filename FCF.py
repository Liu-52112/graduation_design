import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class FCF(nn.Module):
    def __init__(self, num_users, num_items, factor_num, drop_ratio=0.01):
        super(FCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.users_embeddings = torch.nn.Embedding(num_users, factor_num)
        self.items_embeddings = torch.nn.Embedding(num_items, factor_num)
        self.dropout = nn.Dropout(p=drop_ratio)
        self.affine_layer = nn.Linear(factor_num, out_features=1)

        nn.init.normal_(self.users_embeddings.weight, std=0.01)
        nn.init.normal_(self.items_embeddings.weight, std=0.01)
        nn.init.kaiming_uniform_(self.affine_layer.weight, a=1, nonlinearity='sigmoid')
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
        
        
    def forward(self, user, item):
        
        user_embd = self.users_embeddings(user)
        item_embd = self.items_embeddings(item)
        out =  user_embd*item_embd
        #out = self.dropout(out)
        prediction = self.affine_layer(out)
        #out = torch.sigmoid(torch.sum(out, dim=1))
        
        return prediction.view(-1)
    
    def bce_loss(self, users, items,labels, layer = False):
        user_emd = self.users_embeddings(users)
        item_emb = self.items_embeddings(items)
        if layer:
            outputs = user_emd*item_emb
            outputs = self.dropout(outputs)
            outputs = self.affine_layer(outputs)
        else:
            outputs = torch.sum(user_emd * item_emb, dim=1)
            outputs = torch.sigmoid(outputs)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(outputs.view(-1), labels)
        return loss, loss.item()
        
    
    def train_one_epoch(self, optimizer, training_loader, device):
        users = np.random.permutation(range(self.num_users))
        training_loader.dataset.generate_ngs()

        for i, user_id in tqdm(enumerate(users)):
            users_ids, items_ids, labels = training_loader.dataset.get_users_all(user_id)
            users_ids, items_ids, labels = users_ids.to(device),  items_ids.to(device), labels.to(device)
            for _ in range(1): 
                loss, loss_value = self.bce_loss(users_ids, items_ids, labels, False)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
    def train_one_epoch_fedavg(self, optimizer, training_loader, device,  client_fraction ,loss_f = None, epoch = 0):
        training_loader.dataset.generate_ngs()
        
        selected_clients = np.random.choice(range(self.num_users), size=int(self.num_users * client_fraction), replace=False)
        
        original_item_embedding = self.items_embeddings.weight
        original_affine_layer_weight = self.affine_layer.weight
        original_affine_layer_bias = self.affine_layer.bias
        list_item_embedding = []
        list_affine_layer_weight = []
        list_affine_layer_bias = []
        sum_n = 0
        total_loss = 0
        for client_id in selected_clients:
            users_ids, items_ids, labels = training_loader.dataset.get_users_all(client_id)
            users_ids, items_ids, labels = users_ids.to(device), items_ids.to(device), labels.to(device)
            for _ in range(1): 
                optimizer.zero_grad()
                loss, loss_value = self.bce_loss(users_ids, items_ids, labels, True)
                loss.backward()
                
                optimizer.step()
                total_loss += loss.item()
            with torch.no_grad():
                sum_n += len(users_ids)
                list_item_embedding.append([self.items_embeddings.weight, len(users_ids)])
                self.items_embeddings.weight.data = original_item_embedding
                
                list_affine_layer_weight.append([self.affine_layer.weight, len(users_ids)])
                self.affine_layer.weight.data = original_affine_layer_weight
                
                list_affine_layer_bias.append([self.affine_layer.bias, len(users_ids)])
                self.affine_layer.bias.data = original_affine_layer_bias
                
        with torch.no_grad():
            tmp = torch.zeros_like(self.items_embeddings.weight)
            for item_embedding, n_k in list_item_embedding:
                tmp += (n_k / sum_n) * item_embedding
            self.items_embeddings.weight.data = tmp
            
            tmp = torch.zeros_like(self.affine_layer.weight)
            for affine_layer_weight, n_k in list_affine_layer_weight:
                tmp += (n_k / sum_n) * affine_layer_weight
            self.affine_layer.weight.data = tmp
            
            tmp = torch.zeros_like(self.affine_layer.bias)
            for affine_layer_bias, n_k in list_affine_layer_bias:
                tmp += (n_k / sum_n) * affine_layer_bias
            self.affine_layer.bias.data = tmp
            
        return total_loss
        
        
        
        
