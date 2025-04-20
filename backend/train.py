# backend/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import MentalHealthNN
from utils import clean_data

# Load and clean dataset
df = pd.read_csv('survey.csv')
df.columns = df.columns.str.lower()
df = df.drop(columns=["timestamp", "comments"], errors='ignore')
df = clean_data(df)

# Encode categorical features
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Split data
X = df.drop('treatment', axis=1)
y = df['treatment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model
input_dim = X_train.shape[1]
model = MentalHealthNN(input_dim, hidden_dim=32, output_dim=1)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(50):
    for xb, yb in train_loader:
        preds = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save model
torch.save(model.state_dict(), 'model_weights.pth')

# Save encoders
with open('encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("✅ Training complete. Model and encoders saved.")
