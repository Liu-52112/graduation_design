import numpy as np
negative_label = []
negative_users = []
negative_items = []
with open ('ncf_data/Data/ml-1m.test.negative', 'r') as file:
    for line in file:
        data = line.split('\t')
        x = data[0].split(',')
        user = int(x[0][1:])
        item1 = int(x[1][0:-1])
        negative_users.append(user)
        negative_items.append(item1)
        negative_label.append(0)
        for i in range(1,100):
            negative_users.append(user)
            negative_items.append(int(data[i]))
            negative_label.append(0)
len(negative_users)

test_data = {}
with open('ncf_data/Data/ml-1m.test.rating', 'r') as file:
    for line in file:
        line = line.rstrip().split('\t')
        test_data[int(line[0])] = int(line[1])
len(test_data)


pos_users = []
pos_items = []
pos_label = []
train_mat = []
with open("ncf_data/Data/ml-1m.train.rating", 'r') as file:
    for line in file:
        line = line.rstrip().split('\t')
        pos_users.append(int(line[0]))
        pos_items.append(int(line[1]))
        pos_label.append(1)
        train_mat.append((int(line[0]), int(line[1])))
len(pos_users)
num_users = max(pos_users)
num_users

num_users = max(pos_users)
num_items = max(pos_items)
print(len(pos_users))

from GMF import GMF
from dataprocess import *
from dataloader import *


training_dataset = InteractionDataset(num_items,pos_users, pos_items, pos_label, ng_s=4, is_training=True)
batch_size = 256
print(len(training_dataset))
shuffle = True
training_loader = torch.utils.data.DataLoader(training_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = 0)

from NCF_ import NCF

#model = GMF(num_users+1, num_items+1, 64)
#model = NCF(user_num=num_users+1, item_num=num_items+1, factor_num=64, num_layers=1, dropout=0, model='GMF')
model = NCF(user_num=num_users+1, item_num=num_items+1, factor_num=16, num_layers=3, dropout=0.0, model='NeuMF-end')
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epoch = 0
best_epoch = 0
best_hr = 0
best_ndcg = 0
no_improvement_epoch = 0
max_no_improvement_epochs = 10
total_loss = 0

from evaluate import *
negative_users = torch.tensor(negative_users)
negative_items = torch.tensor(negative_items)   
while True:
    total_loss=0
    epoch = epoch +1
    model.train()
    training_loader.dataset.generate_ngs()
    for user_ids, item_ids, labels in tqdm(training_loader, desc= "Epoch "+str(epoch)+": "):
        labels = labels.float()
        model.zero_grad()
        outputs = model(user_ids, item_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    model.eval()
    pred = predict_all_items(model,negative_users, negative_items)
    topk = get_top_k_items_for_each_user(pred, 10)
    hr = calculate_hit_rate(topk, test_data)
    ndcg = calclulate_ndcg(topk, test_data)
    avg_loss = total_loss / len(training_loader)
    print(f"Avg. Loss = {avg_loss:.4f} HR_10 = {hr:.10f} NDCG_10 = {ndcg:.10f}")
    