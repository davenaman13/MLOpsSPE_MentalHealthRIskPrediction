# backend/model.py

import torch
import torch.nn as nn

class MentalHealthNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MentalHealthNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.out(x)
