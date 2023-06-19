import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import numpy as np
import pickle
from catboost import CatBoostRegressor


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    
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
