import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import copy



class FCF(nn.Module):
    def __init__(self, num_users, num_items, factor_num):
        super(FCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.users_embeddings = torch.nn.Embedding(num_users, factor_num)
        self.items_embeddings = torch.nn.Embedding(num_items, factor_num)
        self.cen_users_embeddings = torch.nn.Embedding(num_users, factor_num)
        self.cen_items_embeddings = torch.nn.Embedding(num_items, factor_num)
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
        out = torch.sum(user_embd*item_embd, dim=1)
        return torch.sigmoid(out)
    
    def bce_loss(self, users, items,labels, cen):
        if not cen:
            user_emd = self.users_embeddings(users)
            item_emb = self.items_embeddings(items)
        else:
            user_emd = self.cen_users_embeddings(users)
            item_emb = self.cen_items_embeddings(items)
        outputs = torch.sum(user_emd*item_emb, dim=1)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(outputs, labels)
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
                #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.)
                optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10)
            # optimizer.zero_grad()
        
        # for user_ids, item_ids, labels in tqdm(training_loader, desc= "Item "):
        #     loss, loss_value = self.bce_loss(user_ids, item_ids, labels, True)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10)
            
    def train_one_epoch_fedavg(self, optimizer, training_loader, device,  client_fraction):
        training_loader.dataset.generate_ngs()

        selected_clients = np.random.choice(range(self.num_users), size=int(self.num_users * client_fraction), replace=False)
        
        original_item_embedding = self.items_embeddings.weight
        list_item_embedding = []
        sum_n = 0
        for client_id in selected_clients:

            users_ids, items_ids, labels = training_loader.dataset.get_users_all(client_id)
            users_ids, items_ids, labels = users_ids.to(device), items_ids.to(device), labels.to(device)

            for _ in range(1): 
                loss, loss_value = self.bce_loss(users_ids, items_ids, labels, False)
                loss.backward()
                optimizer.step()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.)
                optimizer.zero_grad()

            with torch.no_grad():
                sum_n += len(users_ids)
                list_item_embedding.append([self.items_embeddings.weight, len(users_ids)])
                self.items_embeddings.weight.data = original_item_embedding
            
        
        with torch.no_grad():
            tmp = torch.zeros_like(self.items_embeddings.weight)
            for item_embedding, n_k in list_item_embedding:
                tmp += (n_k / sum_n) * item_embedding
            self.items_embeddings.weight.data = tmp
        return loss
        
        
        
        
