# backend/utils.py

import pandas as pd
import pickle
import os

def clean_data(df):
    df = df.copy()
    
    # Handle missing ages
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df = df[(df['age'] >= 15) & (df['age'] <= 100)]
        df['age'] = df['age'].fillna(df['age'].median())

    # Normalize 'gender'
    if 'gender' in df.columns:
        df['gender'] = df['gender'].str.lower().str.strip()
        df['gender'] = df['gender'].replace({
            'male': 'male', 'm': 'male', 'man': 'male',
            'female': 'female', 'f': 'female', 'woman': 'female',
            'trans': 'trans', 'transgender': 'trans', 'non-binary': 'non-binary'
        })
        df['gender'] = df['gender'].fillna('other')

    # Fill missing categorical values with 'Unknown'
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna('Unknown')

    return df


def load_encoders(path='encoders.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

def preprocess_input(df, encoders):
    df = df.copy()
    for col, encoder in encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col])
            except ValueError:
                # Handle unseen labels by mapping unknowns to a default value (e.g., the most frequent class or 0)
                known_classes = list(encoder.classes_)
                df[col] = df[col].apply(lambda x: encoder.transform([x])[0] if x in known_classes else encoder.transform([known_classes[0]])[0])
    return df
