import logging
import os
from datetime import datetime
import torch
import  numpy as np
from tqdm import tqdm

def dataloader(train_path, training_user_ids, training_items_ids, training_label, negative_path, device):
    with open(train_path, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            training_user_ids.append(int(line[0]))
            training_items_ids.append(int(line[1]))
            training_label.append(1)
    print(f"正样本数量: {len(training_user_ids)}")
    num_users = max(training_user_ids)
    num_items = max(training_items_ids)
    
    training_dataset = InteractionDataset(num_items,training_user_ids, training_items_ids, training_label, ng_s=4, is_training=True)
    
    # 测试集
    test_dataset = {}
    
    # 负样本
    neg_users_ids = []
    neg_items_ids = []

    with open(negative_path, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            for i in range(100):
                neg_users_ids.append(int(line[0]))
            neg_items_ids.append(int(line[1]))
            for id in line[2].strip().split(' '):   
                training_items_ids.append(int(id))
                neg_items_ids.append(int(id))
    
            test_dataset[int(line[0])] = int(line[1])
            
    print(f"负样本数量: {len(neg_items_ids)}")
    print(f"总样本数量: {len(training_user_ids)}")
    print(f"测试集数量: {len(test_dataset)}")
    
    return training_dataset, test_dataset, neg_users_ids, neg_items_ids

def generate_file(dataset, model):
    '''
    生成训练日志
    为 log 文件夹下面日期文件夹下面。
    '''
    now = datetime.now()
    date, time = str(now).split(' ')
    time = time.replace(':', '-')

    log_directory = f"log/{date}/{dataset}_{model}/{time}"
    log_filename = f"training.log"

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_path = os.path.join(log_directory, log_filename)
    logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f'训练日志 {str(now)} 1')
    print(f"训练日志在{log_directory}")
    
    return log_directory

def filepath(dataset):
    dataset_info = 'Data/'+dataset + '.info'
    training_set = 'Data/'+dataset + '_training.dat'
    test_set = 'Data/' + dataset + '_test.dat'
    negative_set = 'Data/' + dataset + '_negative.dat'
    return dataset_info, training_set, test_set, negative_set

class InteractionDataset(torch.utils.data.Dataset):
    def __init__(self, num_item, user_ids, item_ids, labels, ng_s, transform=None, is_training=None):
        self.user_ids_pos = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids_pos = torch.tensor(item_ids, dtype=torch.long)
        self.labels_pos = torch.tensor(labels, dtype=torch.float)
        self.num_item = num_item
        self.is_training = is_training
        self.ng_s = ng_s
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float)  # 修改这里以匹配labels的初始化

    def __len__(self):
        return  (self.ng_s+1) * len(self.item_ids_pos)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.labels[idx]
    
    def get_users_all(self, user_id):
        user_indices = torch.where(self.user_ids == user_id)[0]
        if len(user_indices)>0:
            user_item_ids  = self.item_ids[user_indices]
            user_item_labels = self.labels[user_indices]
            return torch.tensor([user_id] * len(user_indices), dtype=torch.long), user_item_ids, user_item_labels
        else:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.float)            

    def generate_ngs(self):
        assert self.is_training, 'no need to sampling when testing'

        user_item_set = set(zip(self.user_ids_pos.numpy(), self.item_ids_pos.numpy()))
        
        neg_samples = []
        for u in tqdm(self.user_ids_pos, '生成负样本:'):
            u = u.item()  # 转换为Python标量
            n_negs = 0
            while n_negs < self.ng_s:
                j = np.random.randint(self.num_item, size=self.ng_s*2)  # 生成多个候选，以减少循环次数
                j = [item for item in j if (u, item) not in user_item_set][:self.ng_s-n_negs]  # 过滤已存在的正样本
                neg_samples.extend([(u, neg, 0) for neg in j])  # 生成(u, item, label)格式的负样本
                n_negs += len(j)

        user_ids_neg, item_ids_neg, labels_neg = zip(*neg_samples)  # 解压负样本列表
        self.user_ids_neg = torch.tensor(user_ids_neg, dtype=torch.long)
        self.item_ids_neg = torch.tensor(item_ids_neg, dtype=torch.long)
        self.labels_neg = torch.tensor(labels_neg, dtype=torch.float)

        # 合并正负样本
        self.user_ids = torch.cat([self.user_ids_pos, self.user_ids_neg], dim=0)
        self.item_ids = torch.cat([self.item_ids_pos, self.item_ids_neg], dim=0)
        self.labels = torch.cat([self.labels_pos, self.labels_neg], dim=0)