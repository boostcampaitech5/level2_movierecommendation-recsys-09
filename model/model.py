import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


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
