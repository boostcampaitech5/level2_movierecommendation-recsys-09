import torch.nn as nn
import pickle
import os


class UltraGCN(nn.Module):
    def __init__(self, **params):
        super(UltraGCN, self).__init__()
        
        self.user_num = params['user_num']
        self.item_num = params['item_num']
        self.embedding_dim = params['embedding_dim']
        
        self.delta = params['delta']
        self.lambda_ = params['lambda']

        self.user_embeds = nn.Embedding(self.user_num, self.embedding_dim)
        self.item_embeds = nn.Embedding(self.item_num, self.embedding_dim)

        with open(os.path.join(params['data_dir'], 'constraint_matrix.pickle'), 'rb') as f:
            self.constraint_mat = pickle.load(f)

        self.initial_weights()

    def initial_weights(self):
        nn.init.xavier_normal_(self.user_embeds.weight)
        nn.init.xavier_normal_(self.item_embeds.weight)

    def forward(self, data):
        
        users = data[:, 0]
        items = data[:, 1]
        
        user_embeds = self.user_embeds(users)
        item_embeds = self.item_embeds(items)
         
        return (user_embeds * item_embeds).sum(dim=-1).sigmoid()