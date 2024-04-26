import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import random
import pandas as pd

def add_laplace_noise(data, scale, device):
    """
    向数据添加拉普拉斯噪声。
    :param data: 要添加噪声的数据。
    :param scale: 拉普拉斯噪声的规模参数。
    """
    noise = torch.distributions.Laplace(0, scale).sample(data.shape).to(device)
    return data + noise

class FedFast(nn.Module):
    def __init__(self, num_users, num_items, factor_num, num_cluster =20, drop_ratio=0.1, user_dict=None):
        super(FedFast, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_cluster = num_cluster 
        self.clusters = {i: [] for i in range(num_cluster)}
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
        
        # self.init_cluster(user_dict)
    
    def init_cluster(self, user_dict = None):
        if user_dict is None:
            self.cluster_user()
        else:
            data = pd.read_csv('ml-100k/u.user', sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
            le = LabelEncoder()
            data['occupation'] = le.fit_transform(data['occupation'])
            data['zip_code'] = data['zip_code'].str.extract('(\d+)', expand=False).astype(float)
            data['gender'] = (data['gender']== 'M').astype(int)
            
            scaler = StandardScaler()
            x = scaler.fit_transform(data[['age', 'gender', 'occupation', 'zip_code']])
            kmeans = KMeans(n_clusters=self.num_cluster, random_state=0,n_init=10).fit(x)
            
            
            for user_id, cluster_id  in enumerate(kmeans.labels_):
                self.clusters[cluster_id].append(user_dict[str(user_id+1)])
            
    def init_weights(self, optimizer, training_loader, device = None):
        users = np.random.permutation(range(self.num_users))
        training_loader.dataset.generate_ngs()
        
        #original_user_emb = self.users_embeddings.weight.data.clone()
        original_item_emb = self.items_embeddings.weight.data.clone()
        original_aff_weights = self.affine_layer.weight.data.clone()
        original_aff_bias = self.affine_layer.bias.data.clone()
        
        list_item_embedding = []
        list_predict_layer_weight = []
        list_predict_layer_bias = []
        
        sum_n = 0
        # 本地训练。
        for i, user_id in tqdm(enumerate(users)):
            users_ids, items_ids, labels = training_loader.dataset.get_users_all(user_id)
            users_ids, items_ids, labels = users_ids.to(device),  items_ids.to(device), labels.to(device)
            for _ in range(1): 
                optimizer.zero_grad()
                loss, loss_value = self.bce_loss(users_ids, items_ids, labels, False)
                loss.backward()
                optimizer.step()
                
            with torch.no_grad():
                # 保存每个用户的embedding
                list_item_embedding.append([self.items_embeddings.weight.data.clone(), len(users_ids)])
                list_predict_layer_weight.append([self.affine_layer.weight.data.clone(), len(users_ids)])
                list_predict_layer_bias.append([self.affine_layer.bias.data.clone(), len(users_ids)])
                sum_n += len(users_ids)  
                              
                # 用户上传加密的user_embeddings
                #self.users_embeddings.weight.data[user_id] = add_laplace_noise(self.weighted_k[user_id]['user_embeddings'][user_id], 0.1)
                
                # 还原原始的模型。
                self.items_embeddings.weight.data = original_item_emb.data.clone()
                self.affine_layer.weight.data = original_aff_weights.data.clone()
                self.affine_layer.bias.data = original_aff_bias.data.clone()
                
        # 用fedavg 去更新全局的self.items_embeddings.weight.data, self.predict_layer.weight.data, self.predict_layer.bias.data
        with torch.no_grad():
            
            tmp = torch.zeros_like(self.items_embeddings.weight)
            for item_embedding, n_k in list_item_embedding:
                tmp += (n_k / sum_n) * item_embedding.data
            self.items_embeddings.weight.data = tmp
            
            tmp = torch.zeros_like(self.affine_layer.weight)
            for affine_layer_weight, n_k in list_predict_layer_weight:
                tmp += (n_k / sum_n) * affine_layer_weight.data
            self.affine_layer.weight.data = tmp
            
            tmp = torch.zeros_like(self.affine_layer.bias)
            for affine_layer_bias, n_k in list_predict_layer_bias:
                tmp += (n_k / sum_n) * affine_layer_bias.data  
            self.affine_layer.bias.data = tmp
        self.cluster_user()
    
    def cluster_user(self):
        self.clusters = {i:[] for i in range(self.num_cluster)}
        user_embeddings = self.users_embeddings.weight.data.cpu()        
        kmeans = KMeans(n_clusters=self.num_cluster, random_state=0,n_init=10)
        cluster_labels =kmeans.fit(user_embeddings)
        for user_id, cluster_id  in enumerate(cluster_labels.labels_):
            self.clusters[cluster_id].append(user_id)
            
    def get_cluster(self, client):
        cluster_c = None
        for cluster in self.clusters.values():
            if client in cluster:
                cluster_c = cluster
        return cluster_c
        
    
        
    def forward(self, user, item):
        
        user_embd = self.users_embeddings(user)
        item_embd = self.items_embeddings(item)
        output = user_embd * item_embd
        out = self.affine_layer(output)
        return out.view(-1)
    
    def bce_loss(self, users, items,labels, cen):
        user_emd = self.users_embeddings(users)
        item_emb = self.items_embeddings(items)
        output = user_emd * item_emb
        output = self.dropout(output)
        out = self.affine_layer(output)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(out.view(-1), labels)
        return loss, loss.item()
    
    
    @torch.no_grad()
    def ActvSamp(self, m):
        num_users = m/self.num_cluster
        selected_clients = []
        for cluster in self.clusters.values():
            num_samples = min(int(num_users+1), len(cluster))
            clients = np.random.choice(cluster, num_samples, replace=False)
            for c in clients:
                selected_clients.append(int(c))
        # print(f"Selected clients: {selected_clients}")
        return selected_clients
    
    @torch.no_grad()
    def ActvAGG(self, selected_clients,original_user_embedding ,weight_k, num_k, gama, device='cpu'):
        """ This is the implementation of the Active Aggregation algorithm in the paper.
        Data:
            S: sampled clients ∀k ∈ S,    ====> selected_clients
            w k : model weights for clients  ====>  weight_k
            w 0 : aggregated weights from previous round  =====> self.user_embeddings, self.items_embeddings, self.affine_layer
            n k : number of training instances on client k  =====> num_k
            p: number of client clusters  =====> self.num_cluster
            γ: averaging discount (for subordinate updates)  =====> gama
        Result: 
            w: aggregated weights from w k   ====> self.user_embeddings, self.items_embeddings, self.affine_layer
            G: client cluster ====> self.clusters

        """
            
        ## Updating the non-embedding components 用fedavg的方式去做。
        affine_layer_weights = torch.zeros_like(self.affine_layer.weight)
        affine_layer_bias = torch.zeros_like(self.affine_layer.bias)
        for client in selected_clients:
            affine_layer_weights += num_k[client]/sum(num_k.values()) * weight_k[client]['aff_weights']
            affine_layer_bias += num_k[client]/sum(num_k.values()) * weight_k[client]['aff_bias']
        self.affine_layer.weight.data= affine_layer_weights
        self.affine_layer.bias.data = affine_layer_bias
        
        
        # 论文中的  更新方式
        q = {k: torch.zeros(self.num_items) for k in selected_clients}
        for client in selected_clients:
            q[client] = q[client].to(device)
            Ik = torch.where(torch.norm(weight_k[client]['item_embeddings'] - self.items_embeddings.weight.data, p =1,dim=1) > 0)[0]
            q[client][Ik] = torch.norm(weight_k[client]['item_embeddings'][Ik] - self.items_embeddings.weight[Ik].data, p=1, dim=1)
        
        for i in range(self.num_items):
            ks = [k for k in selected_clients if q[k][i] > 0]
            if len(ks) > 0:
                self.items_embeddings.weight[i].data = torch.sum(torch.stack([q[k][i] * weight_k[k]['item_embeddings'][i] for k in ks])) / torch.sum(torch.stack([q[k][i] for k in ks]))
        
        ## Updating the user delegate embeddings
        # 已经在本地训练的时候更新过了。
        for client in selected_clients:
            self.users_embeddings.weight[client].data = weight_k[client]['user_embeddings'][client].data

        self.cluster_user()
        
        delta = torch.zeros_like(self.users_embeddings.weight)
        
        for k in selected_clients:
            ck = self.get_cluster(k)
            for s in ck:
                if s not in selected_clients:
                    delta[s] += (weight_k[k]['user_embeddings'][k].data - original_user_embedding[k].data) 
                    
        for c in self.clusters:
            selected_clients_in_c = [s for s in self.clusters[c] if s in selected_clients]
            if(len(selected_clients_in_c) == 0):
                continue
            for s in self.clusters[c]:
                if s not in selected_clients:
                    # print("Update")
                    self.users_embeddings.weight[s].data  = original_user_embedding[s].data +  gama * delta[s]/len(selected_clients_in_c)
        ## 返回本地的 w， G
            
    
    def train_one_epoch(self, optimizer, training_loader, device, client_fraction, epoch):
    
        self.users_embeddings.weight.data = add_laplace_noise(self.users_embeddings.weight.data.clone(), 0.01, device)
    
        training_loader.dataset.generate_ngs()
        
        
        m = max(1, int(client_fraction*self.num_users +1))
        
        selected_clients = self.ActvSamp(m)
        
        #selected_clients = np.random.choice(range(self.num_users), size=int(self.num_users * client_fraction), replace=False)
        # epoch = torch.tensor(epoch).to(device)

        original_item_embedding = self.items_embeddings.weight.data.clone()
        original_affine_layer_weight = self.affine_layer.weight.data.clone()
        original_affine_layer_bias = self.affine_layer.bias.data.clone()
        original_user_embedding = self.users_embeddings.weight.data.clone()
        # list_item_embedding = []
        # list_affine_layer_weight = []
        # list_affine_layer_bias = []
        weighted = {k: {'user_embeddings': torch.zeros_like(self.users_embeddings.weight),
                        'item_embeddings': torch.zeros_like(self.items_embeddings.weight),
                        'aff_weights': torch.zeros_like(self.affine_layer.weight),
                        'aff_bias': torch.zeros_like(self.affine_layer.bias)} for k in selected_clients}
        
        num_k = {k: 0 for k in selected_clients}
        sum_n = 0
        for client in tqdm(selected_clients, desc='Training clients'):
            user, item, label = training_loader.dataset.get_users_all(client)
            user, item, label = torch.LongTensor(user).to(device), torch.LongTensor(item).to(device), torch.FloatTensor(label).to(device)
            for i in range(1):
                optimizer.zero_grad()
                loss, loss_value = self.bce_loss(user, item, label, False)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                # list_item_embedding.append([self.items_embeddings.weight, len(user)])
                # list_affine_layer_weight.append([self.affine_layer.weight, len(user)])
                # list_affine_layer_bias.append([self.affine_layer.bias, len(user)])
                sum_n += len(user)
                weighted[client]['user_embeddings'] = self.users_embeddings.weight.data.clone()
                weighted[client]['item_embeddings'] = self.items_embeddings.weight.data.clone()
                weighted[client]['aff_weights'] = self.affine_layer.weight.data.clone()
                weighted[client]['aff_bias'] = self.affine_layer.bias.data.clone()
                num_k[client] = len(user)
                
                self.users_embeddings.weight.data = original_user_embedding
                self.items_embeddings.weight.data = original_item_embedding
                self.affine_layer.weight.data = original_affine_layer_weight
                self.affine_layer.bias.data = original_affine_layer_bias
        
        gama = np.exp(-epoch)
        self.ActvAGG(selected_clients, original_user_embedding ,weighted, num_k, gama, device = device)
        
        
    
            