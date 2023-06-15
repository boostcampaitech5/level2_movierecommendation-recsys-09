import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

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
        
    # def predict(self, test_data):
    #     return self.model.predict(test_data)

    def load_state_dict(self, model_path):
        with open(model_path, 'rb') as f:
                model = pickle.load(f)
        self.model = model
        return self.model

    def save_model_pkl(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)