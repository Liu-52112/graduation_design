import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import random
import pandas as pd
from evaluate import *

def add_laplace_noise(data, scale):
    """
    向数据添加拉普拉斯噪声。
    :param data: 要添加噪声的数据。
    :param scale: 拉普拉斯噪声的规模参数。
    """
    noise = torch.distributions.Laplace(0, scale).sample(data.shape)
    return data + noise

class FedBSO(nn.Module):
    def __init__(self, num_users, num_items, factor_num, num_cluster=20,drop_ratio=0.1, user_dict=None, p5 = 0.2, p6=0.2, if_fedavg = False):
        
        super(FedBSO, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        self.users_embeddings = nn.Embedding(num_users, factor_num)
        self.items_embeddings = nn.Embedding(num_items, factor_num)
        
        self.dropout = nn.Dropout(p=drop_ratio)
        self.predict_layer = nn.Linear(factor_num, 1)
        
        nn.init.normal_(self.users_embeddings.weight, std=0.01)
        nn.init.normal_(self.items_embeddings.weight, std=0.01)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
        
        
        self.num_cluster = num_cluster
        self.clusters = {i:[] for i in range(num_cluster)}
        self.best_client = {i:0 for i in range(num_cluster)}
        self.best_client_cluster = {i:torch.zeros_like(self.users_embeddings.weight.data[0]) for i in range(num_cluster)}

        self.p_5= p5
        self.p_6= p6
        self.is_fedavg = if_fedavg
        
        
    def forward(self, user, item):
        
        embed_user_GMF = self.users_embeddings(user)
        embed_item_GMF = self.items_embeddings(item)
        output_GMF = embed_user_GMF * embed_item_GMF
        prediction = self.predict_layer(output_GMF)
        return prediction.view(-1)
    
    # 每一个用户先训练一下。 fedavg 初始化。
    def init_weights(self, optimizer, training_loader, device = None):
        users = np.random.permutation(range(self.num_users))
        training_loader.dataset.generate_ngs()
        
        #original_user_emb = self.users_embeddings.weight.data.clone()
        original_item_emb = self.items_embeddings.weight.data.clone()
        original_aff_weights = self.predict_layer.weight.data.clone()
        original_aff_bias = self.predict_layer.bias.data.clone()
        
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
                list_predict_layer_weight.append([self.predict_layer.weight.data.clone(), len(users_ids)])
                list_predict_layer_bias.append([self.predict_layer.bias.data.clone(), len(users_ids)])
                sum_n += len(users_ids)  
                              
                # 用户上传加密的user_embeddings
                #self.users_embeddings.weight.data[user_id] = add_laplace_noise(self.weighted_k[user_id]['user_embeddings'][user_id], 0.1)
                
                # 还原原始的模型。
                self.items_embeddings.weight.data = original_item_emb.clone()
                self.predict_layer.weight.data = original_aff_weights.clone()
                self.predict_layer.bias.data = original_aff_bias.clone()
                
        #用fedavg 去更新全局的self.items_embeddings.weight.data, self.predict_layer.weight.data, self.predict_layer.bias.data
        with torch.no_grad():
            
            tmp = torch.zeros_like(self.items_embeddings.weight)
            for item_embedding, n_k in list_item_embedding:
                tmp += (n_k / sum_n) * item_embedding.data
            self.items_embeddings.weight.data = tmp
            
            tmp = torch.zeros_like(self.predict_layer.weight)
            for predict_layer_weight, n_k in list_predict_layer_weight:
                tmp += (n_k / sum_n) * predict_layer_weight.data
            self.predict_layer.weight.data = tmp
            
            tmp = torch.zeros_like(self.predict_layer.bias)
            for predict_layer_bias, n_k in list_predict_layer_bias:
                tmp += (n_k / sum_n) * predict_layer_bias.data  
            self.predict_layer.bias.data = tmp
        self.cluster_user(False)
    
    # def init_clusters(self, user_dict):
    #     if user_dict is None:
    #         self.cluster_user()
    #     else:
    #         data = pd.read_csv('ml-100k/u.user', sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
    #         le = LabelEncoder()
    #         data['occupation'] = le.fit_transform(data['occupation'])
    #         data['zip_code'] = data['zip_code'].str.extract('(\d+)', expand=False).astype(float)
    #         data['gender'] = (data['gender']== 'M').astype(int)
            
    #         scaler = StandardScaler()
    #         x = scaler.fit_transform(data[['age', 'gender', 'occupation', 'zip_code']])
    #         kmeans = KMeans(n_clusters=self.num_cluster, random_state=0,n_init=10).fit(x)
    #         for user_id, cluster_id  in enumerate(kmeans.labels_):
    #             self.clusters[cluster_id].append(user_dict[str(user_id+1)])
    
    def cluster_user(self, has_center=True):
        self.clusters = {i:[] for i in range(self.num_cluster)}
        user_embeddings = self.users_embeddings.weight.data.cpu()
        if has_center:
            center_=  list(self.best_client_cluster.values())        
            kmeans = KMeans(n_clusters=self.num_cluster, init=center_,random_state=0)
        else:
            kmeans = KMeans(n_clusters=self.num_cluster, random_state=0,n_init=10)
        cluster_labels =kmeans.fit(user_embeddings)
        for user_id, cluster_id  in enumerate(cluster_labels.labels_):
            self.clusters[cluster_id].append(user_id)
        
    
    def get_cluster(self, client):
        cluster_c = []
        for cluster in self.clusters.values():
            if client in cluster:
                cluster_c = cluster
        return cluster_c
    
    def bce_loss(self, users, items,labels, cen):
        user_emd = self.users_embeddings(users)
        item_emb = self.items_embeddings(items)
        output = user_emd * item_emb
        output = self.dropout(output)
        out = self.predict_layer(output)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(out.view(-1), labels)
        return loss, loss.item()
    
    
    # 从每个类中挑选出最好的用户。
    def get_best_client(self, val_user_item, val_data, device):
        self.eval()
        self.best_client = {i:-1 for i in range(self.num_cluster)}
        for i   in self.clusters:
            if len(self.clusters[i]) == 0:
                continue
            best_client = -1
            best_ndcg = -1
            for client in self.clusters[i]:
                items = val_user_item[client]
                users = [client for _ in range(len(items))]
                users, items = torch.LongTensor(users).to(device), torch.LongTensor(items).to(device)
                prediction = predict_all_items(self, users, items)[client]
                sorted_data = sorted(prediction, key=lambda x: x[1], reverse=True)[:10]
                item_ids = [item_id for item_id, _ in sorted_data]
                try:
                    rank = item_ids.index(val_data[client]) + 1
                    dcg  = 1/np.log2(rank+1)
                    idcg = 1/np.log2(2)
                    ndcg = dcg/idcg
                    if(ndcg > best_ndcg):
                        best_ndcg = ndcg
                        best_client = client
                except ValueError:
                    ndcg = 0
                
            self.best_client[i] = best_client
            self.best_client_cluster[i] = self.users_embeddings.weight.data[best_client].clone()
    
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
            w 0 : aggregated weights from previous round  =====> self.user_embeddings, self.items_embeddings, self.predict_layer
            n k : number of training instances on client k  =====> num_k
            p: number of client clusters  =====> self.num_cluster
            γ: averaging discount (for subordinate updates)  =====> gama
        Result: 
            w: aggregated weights from w k   ====> self.user_embeddings, self.items_embeddings, self.predict_layer
            G: client cluster ====> self.clusters

        """
            
        ## Updating the non-embedding components 用fedavg的方式去做。
        predict_layer_weights = torch.zeros_like(self.predict_layer.weight)
        predict_layer_bias = torch.zeros_like(self.predict_layer.bias)
        for client in selected_clients:
            predict_layer_weights += num_k[client]/sum(num_k.values()) * weight_k[client]['aff_weights']
            predict_layer_bias += num_k[client]/sum(num_k.values()) * weight_k[client]['aff_bias']
        self.predict_layer.weight.data= predict_layer_weights
        self.predict_layer.bias.data = predict_layer_bias
        
        
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

        # self.cluster_user(False)
        
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
    
    def train_one_epoch(self, optimizer, training_loader, val_user_item, val_data ,device = None, client_fraction =0.01, epoch=0):
        
        training_loader.dataset.generate_ngs()
        
        self.users_embeddings.weight.data = add_laplace_noise(self.users_embeddings.weight.data.clone(), 0.01)
        
        if epoch == 0:
            self.cluster_user(False) #  聚类用户。 更新 self.clusters 初始化的时候没有中心点
        else:
            self.cluster_user(True) #  聚类用户。 更新 self.clusters 之后就有中心点了
            
        self.get_best_client(val_user_item, val_data, device) # 从每个类中挑选出最好的用户。更新self.best_client_cluster
        
        self.train()
        
        if random.random() < self.p_5 * (1- np.exp(-epoch/200)):
            # 替换掉中心点。emm， 没有排除本身。 ## 为的是在聚类的时候发生变化。 第五步，替换中心点,随机替换掉某个类的中心点
            while True:
                random_client = random.choice(range(self.num_users))
                random_best_client = random.choice(range(self.num_cluster)) # 这个是index
                if random_client not in self.best_client.values():
                    self.best_client_cluster[random_best_client] = self.users_embeddings.weight.data[random_client].clone()
                    # 随机替换最好的用户。
                    self.clusters[random_best_client].remove(self.best_client[random_best_client])
                    self.clusters[random_best_client].append(random_client)
                    # 把最好的用户放入到random_client的类中。
                    for i in self.clusters:
                        if random_client in self.clusters[i]:
                            self.clusters[i].remove(random_client)
                            self.clusters[i].append(self.best_client[random_best_client])
                    break
                
        if random.random() < self.p_6 * (1- np.exp(-epoch/200)):
            ## 交换两个类的中心点。这个是影响下面的训练的。 第六步交换两个类的中心点
            while True:
                swap_index_i = random.randint(0, self.num_cluster-1)
                swap_index = random.randint(0, self.num_cluster-1)
                if swap_index != swap_index_i:
                    self.clusters[swap_index_i].remove(self.best_client[swap_index_i])
                    self.clusters[swap_index].remove(self.best_client[swap_index])
                    self.clusters[swap_index_i].append(self.best_client[swap_index])
                    self.clusters[swap_index].append(self.best_client[swap_index_i])
                    break
        m = max(1, int(client_fraction*self.num_users +1))
        
        selected_clients =self.ActvSamp(m)
        # selected_clients = []
        # for i in range(self.num_cluster):
        #     if self.best_client[i] not in selected_clients:
        #         selected_clients.append(self.best_client[i])
        
        if not self.is_fedavg:
            original_user_embedding = self.users_embeddings.weight.data.clone()
            original_item_embedding = self.items_embeddings.weight.data.clone()
            original_predict_layer_weight = self.predict_layer.weight.data.clone()
            original_predict_layer_bias = self.predict_layer.bias.data.clone()
            
            weighted = {k: {'user_embeddings': torch.zeros_like(self.users_embeddings.weight),
                    'item_embeddings': torch.zeros_like(self.items_embeddings.weight),
                    'aff_weights': torch.zeros_like(self.predict_layer.weight),
                    'aff_bias': torch.zeros_like(self.predict_layer.bias)} for k in selected_clients}
            num_k = {k: 0 for k in selected_clients}
            for client in selected_clients:
                users_ids, items_ids, labels = training_loader.dataset.get_users_all(client)
                users_ids, items_ids, labels = users_ids.to(device), items_ids.to(device), labels.to(device)
                for _ in range(1): 
                    optimizer.zero_grad()
                    loss, loss_value = self.bce_loss(users_ids, items_ids, labels, False)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                with torch.no_grad():
                    weighted[client]['user_embeddings'] = self.users_embeddings.weight.data.clone()
                    weighted[client]['item_embeddings'] = self.items_embeddings.weight.data.clone()
                    weighted[client]['aff_weights'] = self.predict_layer.weight.data.clone()
                    weighted[client]['aff_bias'] = self.predict_layer.bias.data.clone()
                    num_k[client] = len(users_ids)
                    
                    self.users_embeddings.weight.data = original_user_embedding
                    self.items_embeddings.weight.data = original_item_embedding
                    self.predict_layer.weight.data = original_predict_layer_weight
                    self.predict_layer.bias.data = original_predict_layer_bias
            gama = np.exp(-epoch)
            self.ActvAGG(selected_clients, original_user_embedding, weighted, num_k, gama, device = device)
        
        
        
        if self.is_fedavg:
            for i in range(self.num_cluster):
                if len(self.clusters[i]) == 0:
                    print(f"cluster {i} is empty")
                    continue
                    
                # 对于每一个类，
                original_item_embedding = self.items_embeddings.weight.data.clone()
                original_predict_layer_weight = self.predict_layer.weight.data.clone()
                original_predict_layer_bias = self.predict_layer.bias.data.clone()
                list_item_embedding = []
                list_predict_layer_weight = []
                list_predict_layer_bias = []
                sum_n = 0
                selected_clients = np.random.choice(self.clusters[i], size=int(len(self.clusters[i]) * client_fraction), replace=False)
                for client in selected_clients:
                    users_ids, items_ids, labels = training_loader.dataset.get_users_all(client)
                    users_ids, items_ids, labels = users_ids.to(device), items_ids.to(device), labels.to(device)
                    for _ in range(1): 
                        optimizer.zero_grad()
                        loss, loss_value = self.bce_loss(users_ids, items_ids, labels, False)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    sum_n += len(users_ids)
                    with torch.no_grad():
                        list_item_embedding.append([self.items_embeddings.weight.data.clone(), len(users_ids)])
                        self.items_embeddings.weight.data = original_item_embedding
                        list_predict_layer_weight.append([self.predict_layer.weight.data.clone(), len(users_ids)])
                        self.predict_layer.weight.data = original_predict_layer_weight
                        list_predict_layer_bias.append([self.predict_layer.bias.data.clone(), len(users_ids)])
                        self.predict_layer.bias.data = original_predict_layer_bias
                if sum_n == 0:
                    continue
                with torch.no_grad():
                    tmp = torch.zeros_like(self.items_embeddings.weight)
                    for item_embedding, n_k in list_item_embedding:
                        tmp += (n_k / sum_n) * item_embedding.data
                    self.items_embeddings.weight.data = tmp
                    tmp = torch.zeros_like(self.predict_layer.weight)
                    for predict_layer_weight, n_k in list_predict_layer_weight:
                        tmp += (n_k / sum_n) * predict_layer_weight.data
                    self.predict_layer.weight.data = tmp
                    tmp = torch.zeros_like(self.predict_layer.bias)
                    for predict_layer_bias, n_k in list_predict_layer_bias:
                        tmp += (n_k / sum_n) * predict_layer_bias.data  
                    self.predict_layer.bias.data = tmp
