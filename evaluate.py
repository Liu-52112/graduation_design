import torch
import numpy as np
import pandas as pd

def predict_all_items(model, user_ids, item_ids):
    model.eval()
    with torch.no_grad():
        predictions = model(user_ids, item_ids).flatten()
        df = pd.DataFrame({
            'user_id': user_ids.cpu().numpy(),
            'item_id': item_ids.cpu().numpy(),
            'prediction': predictions.cpu().numpy()
        })
        predictions_of_user = df.groupby('user_id').apply(
            lambda x: list(zip(x.item_id, x.prediction))).to_dict()
    return predictions_of_user

def get_top_k_items_for_each_user(predictions_of_user, k=10):
    top_k_items_of_user = {}
    for user_id, item_predictions in predictions_of_user.items():
        sorted_predictions = sorted(item_predictions, key=lambda x: x[1], reverse=True)
        for i in range(k):
            if user_id not in top_k_items_of_user:
                top_k_items_of_user[user_id] = [sorted_predictions[i][0]]
            else:
                top_k_items_of_user[user_id].append(sorted_predictions[i][0])
            
    return top_k_items_of_user

def calculate_hit_rate(recommended_items, test_items):
    hits = 0
    total_users = len(test_items)
    
    for user_id, true_items in test_items.items():
        pred_items = recommended_items.get(user_id, [])
        #print(pred_items)
        if true_items in pred_items:
            hits += 1

    hit_rate = hits / total_users if total_users > 0 else 0
    return hit_rate

def calclulate_ndcg(recommended_items, test_items):
    ndcg = 0
    for user_id, items_id in test_items.items():
        pred_items = recommended_items.get(user_id, [])
        try:
            rank = pred_items.index(items_id) + 1
            dcg  = 1/np.log2(rank+1)
            idcg = 1/np.log2(2)
            ndcg += dcg/idcg
        except ValueError:
            ndcg += 0
    avg_ndcg = ndcg / len(test_items) if test_items else 0 
    return avg_ndcg