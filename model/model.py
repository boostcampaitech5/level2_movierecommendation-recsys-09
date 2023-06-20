import math

import torch
import torch.nn as nn

import pickle
import os

import torch.nn.functional as F
from base import BaseModel
import numpy as np
from catboost import CatBoostRegressor


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

    
class MultiVAEModel(nn.Module):
    def __init__(self, **args):
        super(MultiVAEModel, self).__init__()
        self.p_dims = args["p_dims"]
        self.dropout = args["dropout_rate"]
        self.q_dims = None
        
        if self.q_dims:
            assert self.q_dims[0] == self.p_dims[-1], "In and Out dimensions must equal to each other"
            assert self.q_dims[-1] == self.p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = self.q_dims
        else:
            self.q_dims = self.p_dims[::-1]

        # Last dimension of q- network is for mean and variance
        self.temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList([nn.Linear(self.d_in, self.d_out) for
            self.d_in, self.d_out in zip(self.temp_q_dims[:-1], self.temp_q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(self.d_in, self.d_out) for
            self.d_in, self.d_out in zip(self.p_dims[:-1], self.p_dims[1:])])
        
        self.drop = nn.Dropout(self.dropout)
        self.init_weights()
    
    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)
        
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

    
# AutoRec Model
class AutoRecModel(nn.Module):
    """
    AutoRec
    
    Args:
        - input_dim: (int) input feature의 Dimension
        - emb_dim: (int) Embedding의 Dimension
        - hidden_activation: (str) hidden layer의 activation function.
        - out_activation: (str) output layer의 activation function.
    Shape:
        - Input: (torch.Tensor) input features,. Shape: (batch size, input_dim)
        - Output: (torch.Tensor) reconstructed features. Shape: (batch size, input_dim)
    """
    def __init__(self, **args):
        super(AutoRecModel, self).__init__()
        
        # initialize Class attributes
        self.input_dim = 6807  #self.args.input_dim
        self.emb_dim = args["hidden_size"]
        self.hidden_activation = args["hidden_activation"]
        self.out_activation = args["out_activation"]
        self.num_layers = args["num_layers"]
        self.dropout_rate = args["dropout_rate"]
        
        # define layers
        encoder_modules = list()
        encoder_layers = [self.input_dim] + [self.emb_dim // (2 ** i) for i in range(self.num_layers)]
        for i in range(self.num_layers):
            input_size = encoder_layers[i] 
            output_size = encoder_layers[i + 1] 
            encoder_modules.append(nn.Linear(input_size, output_size))
            activation_function = nn.Sigmoid()
            if activation_function is not None:
                encoder_modules.append(activation_function)
        encoder_modules.append(nn.Dropout(self.dropout_rate))
        
        decoder_modules = list()
        decoder_layers = encoder_layers[::-1]
        for i in range(self.num_layers):
            input_size = decoder_layers[i] 
            output_size = decoder_layers[i + 1] 
            decoder_modules.append(nn.Linear(input_size, output_size))
            activation_function = 'none'
            if activation_function is not None and (i < self.num_layers - 1):
                decoder_modules.append(activation_function)

        self.encoder = nn.Sequential(
            *encoder_modules
        )
        self.decoder = nn.Sequential(
            *decoder_modules
        )
        
        self.init_weights()

    # initialize weights
    def init_weights(self):
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight.data)
                layer.bias.data.zero_()
        
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight.data)
                layer.bias.data.zero_()

    
    def forward(self, input_feature):
        h = self.encoder(input_feature)
        output = self.decoder(h)
        
        return output

class CatBoostModel():
    def __init__(self, loss_function, task_type, sweep):
        if sweep:
            pass
        else:
            self.loss_function = loss_function
            self.task_type = task_type
            self.model = CatBoostRegressor(loss_function=self.loss_function, task_type=self.task_type)

    def train(self, train_data, valid_data):
        data_col = list(train_data.X.columns)
        data_col.remove('year')
        categorical_features = data_col
        self.model.fit(train_data.X, train_data.y, 
                       eval_set=(valid_data.X, valid_data.y), 
                       cat_features=categorical_features)

    def load_state_dict(self, model_path):
        with open(model_path, 'rb') as f:
                model = pickle.load(f)
        self.model = model
        return self.model

    def save_model_pkl(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)


# BERT4Rec Model

class Attention(nn.Module): #compute Scaled Dot Product Attention
    def __init__(self, hidden_units, dropout_rate):
        super(Attention, self).__init__()
        self.hidden_units = hidden_units
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, Q, K, V, mask):
        attn_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_units)
        attn_score = attn_score.masked_fill(mask == 0, -1e9)  # 유사도가 0인 지점은 -infinity로 보내 softmax 결과가 0이 되도록 함
        attn_dist = self.dropout(F.softmax(attn_score, dim=-1))  # attention distribution
        return torch.matmul(attn_dist, V), attn_dist # dim of output : batchSize x num_head x seqLen x hidden_units
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_units, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        assert hidden_units % num_heads == 0
        
        self.num_heads = num_heads
        self.hidden_units = hidden_units
        
        # query, key, value, output 생성을 위해 Linear 모델 생성
        self.W_Q = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_K = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_V = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_O = nn.Linear(hidden_units, hidden_units, bias=False)

        self.attention = Attention(hidden_units, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate) # dropout rate
        self.layerNorm = nn.LayerNorm(hidden_units, 1e-6) # layer normalization

    def forward(self, enc, mask):
        residual = enc # residual connection을 위해 residual 부분을 저장
        batch_size, seqlen = enc.size(0), enc.size(1)
        
        # 1) Do all the linear projections in batch
        Q = self.W_Q(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units // self.num_heads).transpose(1, 2) 
        K = self.W_K(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units // self.num_heads).transpose(1, 2)
        V = self.W_V(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units // self.num_heads).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        output, attn_dist = self.attention(Q, K, V, mask)

        # 3) "Concat" using a view
        output = output.transpose(1, 2).contiguous().view(batch_size, seqlen, -1)

        # Linear Projection, Dropout, Residual sum, and Layer Normalization
        output = self.layerNorm(self.dropout(self.W_O(output)) + residual)
        
        return output, attn_dist
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PositionwiseFeedForward, self).__init__()
        
        self.W_1 = nn.Linear(hidden_units, 4 * hidden_units) 
        self.W_2 = nn.Linear(4 * hidden_units, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.layerNorm = nn.LayerNorm(hidden_units, 1e-6) # layer normalization

    def forward(self, x):
        residual = x
        output = self.W_2(F.gelu(self.dropout(self.W_1(x)))) # activation: gelu
        output = self.layerNorm(self.dropout(output) + residual)
        return output
    
class TransformerBlock(nn.Module):
    def __init__(self, num_heads, hidden_units, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads, hidden_units, dropout_rate)
        self.pointwise_feedforward = PositionwiseFeedForward(hidden_units, dropout_rate)

    def forward(self, input_enc, mask):
        output_enc, attn_dist = self.attention(input_enc, mask)
        output_enc = self.pointwise_feedforward(output_enc)
        return output_enc, attn_dist
    
class BERT4Rec(nn.Module):
    def __init__(self, num_user, num_item, hidden_units, num_heads, num_layers, max_len, dropout_rate, device):
        super(BERT4Rec, self).__init__()

        self.num_user = num_user
        self.num_item = num_item
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.num_layers = num_layers 
        self.device = device
        
        self.item_emb = nn.Embedding(num_item + 2, hidden_units, padding_idx= 0) 
        self.pos_emb = nn.Embedding(max_len, hidden_units) # learnable positional encoding
        self.dropout = nn.Dropout(dropout_rate)
        self.emb_layernorm = nn.LayerNorm(hidden_units, eps=1e-6)
        
        self.blocks = nn.ModuleList([TransformerBlock(num_heads, hidden_units, dropout_rate) for _ in range(num_layers)])
        self.out = nn.Linear(hidden_units, num_item + 1) 
        
    def forward(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.device))
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.device))
        seqs = self.emb_layernorm(self.dropout(seqs))

        mask = torch.BoolTensor(log_seqs > 0).unsqueeze(1).repeat(1, log_seqs.shape[1], 1).unsqueeze(1).to(self.device) # mask for zero pad
        for block in self.blocks:
            seqs, attn_dist = block(seqs, mask)
        out = self.out(seqs)
        return out

    def load_state_dict(self, model_path):
        with open(model_path, 'rb') as f:
                model = pickle.load(f)
        self.model = model
        return self.model