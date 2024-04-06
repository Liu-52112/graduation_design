import torch
import numpy as np
from dataloader import *
from evaluate import *
from BasicMF import BasicMF
from GMF_ import GMF
from MLP import MLP
from NCF_ import NCF
from FCF import FCF
from FedFast import FedFast
import time  # 导入time模块
import argparse
from tqdm import tqdm


def get_device(device_choice):
    if device_choice == 'mps' and torch.backends.mps.is_available():
        print("MPS is available!")
        return torch.device("mps")
    elif device_choice == 'cuda' and torch.cuda.is_available():
        print("CUDA is available!")
        return torch.device("cuda")
    else:
        print("Using CPU.")
        return torch.device("cpu")

def main(args):
    dataset = args.dataset
    model = args.model
    log_path = generate_file(dataset,model ) # 生成日志文件
    device  = get_device(args.device) # 获取当前的device
    
    # 初始化以及 dataloading
    training_user_ids = []
    training_items_ids = []
    training_label = []

    test_dataset = {}
    if dataset !='pin':
        dataset_info, training_set, test_set, negative_set = filepath(dataset)
        training_dataset, test_dataset, neg_users_ids, neg_items_ids =dataloader(train_path= training_set,training_user_ids=training_user_ids, training_items_ids=training_items_ids, training_label=training_label, negative_path=negative_set, device=device )
    else:
        with open ('Data/pinterest-20.train.rating', 'r') as f:
            for line in f:
                line = line.rstrip().split('\t')
                training_user_ids.append(int(line[0]))
                training_items_ids.append(int(line[1]))
                training_label.append(int(line[2]))
        num_items = max(training_items_ids)
        training_dataset = InteractionDataset(num_items,training_user_ids, training_items_ids, training_label, ng_s=4, is_training=True)
        neg_users_ids, neg_items_ids = [], []
        test_dataset = {}
        with open ('Data/pinterest-20.test.negative', 'r') as file:
            for line in file:
                data = line.split('\t')
                x = data[0].split(',')
                user = int(x[0][1:])
                item1 = int(x[1][0:-1])
                neg_users_ids.append(user)
                neg_items_ids.append(item1)
                for i in range(1,100):
                    neg_users_ids.append(user)
                    neg_items_ids.append(int(data[i]))
                test_dataset[user] = int(item1)
    # 模型参数初始化
    batch_size = 256
    shuffle = True
    num_workers = 4
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size = batch_size, shuffle = shuffle, num_workers= 0)
    
    
    
    num_users = max(training_user_ids)
    num_items = max(training_items_ids)
    print(num_users)
    print(num_items)
    all_user_ids = torch.arange(num_users).to(device)
    all_item_idx = torch.arange(num_items).to(device)
    # model = BasicMF(num_users+1, num_items+1, 64).to(device)
    '''
    if args.model == 'mf':
        if device == 'cpu':
            model = BasicMF(num_users+1, num_items+1, 64)
            neg_users_ids = torch.tensor(neg_users_ids)
            neg_items_ids = torch.tensor(neg_items_ids)
        else:
            model = BasicMF(num_users+1, num_items+1, 64).to(device)
            neg_users_ids = torch.tensor(neg_users_ids).to(device)
            neg_items_ids = torch.tensor(neg_items_ids).to(device)
    '''
    criterion = torch.nn.BCELoss()
    if args.model == 'mf':
        model = BasicMF(num_users+1, num_items+1, 64).to(device)
        
    elif args.model == 'gmf':
        model = GMF(num_users+1, num_items+1, args.factor_size).to(device)
        #model = NCF(num_users+1, num_items+1, factor_num=64, model = 'GMF', num_layers=1, dropout=0.2).to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.model == 'mlp':
        model = MLP(num_users+1, num_items+1, hidden_layers=[64, 32, 16], output_size=1).to(device)
    elif args.model == 'ncf':
        model = NCF(num_users+1, num_items+1, factor_num=16, num_layers=3, dropout=0.2, model='NeuMF-end').to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.model == 'fcf' or args.model == 'fedavg':
        model = FCF(num_users +1, num_items +1, factor_num= 64).to(device)
    elif args.model == 'fedfast':
        model = FedFast(num_users+1, num_items+1, factor_num=args.factor_size, num_cluster=20).to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
        
    #model = BasicMF(num_users+1, num_items+1, 64).to(device)
    neg_users_ids = torch.tensor(neg_users_ids).to(device)
    neg_items_ids = torch.tensor(neg_items_ids).to(device)        
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    
    best_hr = 0  
    best_ndcg = 0  
    best_epoch = 0 
    no_improvement_count = 0  
    max_no_improvement_epochs = 20
    epoch = 0
    while True:
        total_loss = 0
        epoch +=1
        epoch_start_time = time.time()
        model.train()
        if args.model =='fcf':
            model.train_one_epoch(optimizer, training_loader, device)
        elif args.model == 'fedavg':
            model.train_one_epoch_fedavg(optimizer, training_loader, device, 0.01)
            print('ss')
        elif args.model == 'fedfast':
            model.train_one_epoch(optimizer, training_loader, device, 0.01, epoch-1)
        else:
            training_loader.dataset.generate_ngs()
            for user_ids, item_ids, labels in tqdm(training_loader, desc= "Epoch "+str(epoch)+": "):
                if (user_ids >= num_users+1).any() or (item_ids >= num_items+1).any():
                    out_of_range_user_ids = user_ids[user_ids >= num_users+1]
                    out_of_range_item_ids = item_ids[item_ids >= num_items+1]
                    print(f"Out of range user_ids: {out_of_range_user_ids}")
                    print(f"Out of range item_ids: {out_of_range_item_ids}")
                    continue 

                optimizer.zero_grad()
                try:
                    if device != 'cpu':
                        user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
                    outputs = model(user_ids, item_ids)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                except Exception as e:  
                    print(f"An error occurred: {e}")
                    print(user_ids)
                    print(item_ids)
                    print(model.user_embeddings.weight.size())
                    print(model.item_embeddings.weight.size())
        neg_items_ids = neg_items_ids.to(device)
        neg_users_ids = neg_users_ids.to(device)
        pred = predict_all_items(model,neg_users_ids, neg_items_ids)
        topk = get_top_k_items_for_each_user(pred, int(args.topk))
        hr = calculate_hit_rate(topk, test_dataset)
        ndcg = calclulate_ndcg(topk, test_dataset)
        avg_loss = total_loss / len(training_loader)
        epoch_end_time = time.time()  # 记录epoch结束的时间
        epoch_duration = epoch_end_time - epoch_start_time
        improvement = False
        if hr > best_hr or ndcg > best_ndcg:
            best_hr = max(hr, best_hr)
            best_ndcg = max(ndcg, best_ndcg)
            best_epoch = epoch
            no_improvement_count = 0
            improvement = True
            torch.save(model.state_dict(), f'{log_path}/{args.model}_{args.dataset}_64_embedding_size_model_best.pth')
            print(f"New best model saved at epoch {best_epoch} with HR = {best_hr:.4f}, NDCG = {best_ndcg:.4f}")
        else:
            no_improvement_count += 1 
        
        print(f"Avg. Loss = {avg_loss:.4f} HR_{args.topk} = {hr:.10f} NDCG_{args.topk} = {ndcg:.10f}, Duration: {epoch_duration:.2f} seconds")
        logging.info(f"Epoch {epoch}: Avg. Loss = {avg_loss:.4f} HR_{args.topk} = {hr:.10f} NDCG_{args.topk} = {ndcg:.10f}, Duration: {epoch_duration:.2f} seconds")
        
        if no_improvement_count >= max_no_improvement_epochs:
            print(f"Stopping early due to no improvement in HR and NDCG for {max_no_improvement_epochs} consecutive epochs.")
            break
        
    print(f"New best model saved at epoch {best_epoch+1} with HR = {best_hr:.4f}, NDCG = {best_ndcg:.4f}")
    logging.info(f"New best model saved at epoch {best_epoch+1} with HR = {best_hr:.4f}, NDCG = {best_ndcg:.4f}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on a specific device with a specified dataset.')
    parser.add_argument('-d', '--device', type=str, choices=['cpu', 'cuda', 'mps'], default='cpu', help='The device to use for training: cpu, cuda, or mps.')
    parser.add_argument('-m', '--model', type=str, choices=['mf', 'gmf', 'mlp', 'ncf','fcf', 'fedavg', 'fedfast'], default='mf', help='The model to use for training.')
    parser.add_argument('-data', '--dataset', type=str, default='ml-100k', help='The dataset to use for training.')
    parser.add_argument('-t', '--topk', type=str, default=20, help='The num of topk items the model recommend.')
    parser.add_argument('-f', '--factor_size', type=int, default=16, help='The size of factor number of the model recommend.')
    
    args = parser.parse_args()
    main(args)