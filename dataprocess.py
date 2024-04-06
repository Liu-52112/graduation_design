import random 
from tqdm import tqdm  
# 数据集路径
m1_1m_data = 'ml-1m/ratings.dat'
ml_100k_data= 'ml-100k/u.data'
af_data = 'AmazonFashion/AMAZON_FASHION.csv'


dataset = {
        'm1-1m': m1_1m_data,
        'ml-100k': ml_100k_data,
        'af': af_data
    }

def initlization():
    global interactions, len_filtered_interaction, filtered_interactions
    global user_length, item_length, data_training, data_test, data_negative
    global users, items, dataset

    interactions = []
    len_filtered_interaction = 0
    filtered_interactions = []
    user_length = 0
    item_length = 0
    data_training = []
    data_test = []
    data_negative = []
    users = {}
    items = {}



def data_load(filepath):
    interactions = []
    with open(filepath, 'r') as file:
        for line in tqdm(file, desc=f"Loading {filepath}"):
            if filepath.endswith('ratings.dat'):
                
                interactions.append(line.rstrip().split("::"))
            if filepath.endswith('u.data'):
                interactions.append(line.rstrip().split('\t'))
            if filepath.endswith('AMAZON_FASHION.csv'):
                interactions.append(line.rstrip().split(','))
    return interactions
            
def data_filter(interactions):
    global len_filtered_interaction, user_length, item_length
    # 过滤交互记录不足20个的
    user_interactions_count = {}
    for interaction in interactions:
        user_id = interaction[0]
        if user_id in user_interactions_count:
            user_interactions_count[user_id] += 1
        else:
            user_interactions_count[user_id] = 1

    filtered_interactions = [interaction for interaction in interactions if user_interactions_count[interaction[0]] >= 20]
    len_filtered_interaction = len(filtered_interactions)
    
    # 建立每一个用户以及item的映射
    u = 0
    i = 0
    for interaction in filtered_interactions:
        if interaction[0] not in users:
            users[interaction[0]]= u
            u = u+1
        if interaction[1] not in items:
            items[interaction[1]]= i
            i = i+1
    user_length = len(users)
    item_length = len(items)
    
    
    return filtered_interactions

        
def data_split(filtered_interactions):
    # 取每一个用户的所有数据，形成映射。
    users_interactions = {}
    for interaction in tqdm(filtered_interactions, desc=f"Spilt the data Step 1"):
        if interaction[0] not in users_interactions:
            users_interactions[interaction[0]] = []
        users_interactions[interaction[0]].append(interaction)
    
    for user in tqdm(users_interactions, desc=f"Spilt the data Step 2"):
        # 将所有用户按时间倒序排列
        users_interactions[user] = sorted(users_interactions[user], key= lambda x:x[3], reverse=True)
        # 取第一个为测试集，其他的为训练集。
        last_interaction = users_interactions[user][0]
        data_training.extend(users_interactions[user][1:])
        data_test.append(last_interaction)
        
        
        # 生成负样本数据
        # 用户user 的正样本
        data_positive = [items[interaction[1]] for interaction in users_interactions[user]]
        # 全部样本 是从0 到 len(items)
        data_items = [x for x in range(len(items))]
        # 从全部样本中剔除正样本数据，并把取99个，与testdata 共100个形成测试集。
        data_negative_user = [item for item in data_items if item not in data_positive]
        user_test = (users[last_interaction[0]], items[last_interaction[1]])
        random.seed(0)
        if len(data_negative_user) >99:
            data_negative.append([user_test, random.sample(data_negative_user, 99)])
        
        
        
        
    # 映射用户和item 到 用户id 和item id
    data_test_final = [[users[interaction[0]], items[interaction[1]], interaction[2], interaction[3] ]for interaction in data_test]
    data_training_final = [[users[interaction[0]], items[interaction[1]], interaction[2], interaction[3]]for interaction in data_training]

    
    return data_test_final, data_training_final, data_negative

def data_save(test_data, training_data, negative_data, dataset_name):
    with open('Data/' + dataset_name + '.info', 'w') as file:
        file.write('User Number: ' + str(len(users)) + '\n')
        file.write('Item Number: ' + str(len(items)) + '\n')
        file.write('Filterd Interactions Number (>=20): ' + str(len_filtered_interaction) + '\n')
        file.write('Training Set size: ' + str(len(training_data)) + '\n')
        file.write('Test Set size: '+ str(len(test_data)) + '\n')
        file.write('Negative size: Everyone sample 99 he/she haven\'t interacted')
        
    
    # 保存测试集
    with open('Data/'+ dataset_name + '_test.dat', 'w') as file:
        for data in tqdm(test_data, desc='Writing test data'):
            file.write(str(data[0]) + '\t' + str(data[1]) + '\t' + str(data[2]) + '\t' + str(data[3]) + '\n')
            
    with open('Data/'+ dataset_name + '_training.dat', 'w') as file:
        for data in tqdm(training_data, desc='Writing training data'):
            file.write(str(data[0]) + '\t' + str(data[1]) + '\t' + str(data[2]) + '\t' + str(data[3]) + '\n')
            
    with open('Data/'+ dataset_name + '_negative.dat', 'w') as file:
        for data in tqdm(negative_data, desc='Writing negative data'):
            file.write(str(data[0][0]) + '\t' + str(data[0][1]) + '\t')  # 修正了这里的索引
            for negative_ in data[1]:
                file.write(str(negative_) + ' ')
            file.write('\n')


def main():
    for dataset_name, filepath in dataset.items():
        initlization()
        print(f"Loading data from {filepath}...")
        interactions = data_load(filepath)
        filtered_interactions = data_filter(interactions)
        test_data, training_data, negative_data = data_split(filtered_interactions)
        data_save(test_data, training_data, negative_data, dataset_name)
        print(f"Data processing for {dataset_name} completed.\n")

if __name__ == '__main__':
    main()
