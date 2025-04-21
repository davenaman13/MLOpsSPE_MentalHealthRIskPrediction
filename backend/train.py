# backend/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

from utils import clean_data

# Load and clean dataset
df = pd.read_csv('survey.csv')
df.columns = df.columns.str.lower()

# Drop irrelevant columns
df = df.drop(columns=["timestamp", "comments"], errors='ignore')

df = clean_data(df)

# Encode categorical features and store encoders
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Split data
X = df.drop('treatment', axis=1)
y = df['treatment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

base_dir = os.path.dirname(__file__)

# Save model
with open(os.path.join(base_dir, 'model_weights.pkl'), 'wb') as f:
    pickle.dump(clf, f)

# Save encoders
with open(os.path.join(base_dir, 'encoders.pkl'), 'wb') as f:
    pickle.dump(label_encoders, f)

print("✅ Training complete. Model and encoders saved.")