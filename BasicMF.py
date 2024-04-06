from torch import nn
class BasicMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size):
        super(BasicMF, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_size)
        self.item_embeddings = nn.Embedding(num_items, embedding_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)
        
        ratings = (user_embedding *item_embedding).sum(1)
        probabilities = self.sigmoid(ratings)
        return probabilities