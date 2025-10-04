# misc.py
"""
Generic utilities for data loading, splitting, preprocessing, training and evaluation.
Designed to be model-agnostic so train.py and train2.py can reuse the same functions.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

def load_data():
    """
    Load Boston dataset from the CMU statlib mirror (manual load as dataset is deprecated).
    Returns a DataFrame with feature columns and 'MEDV' target.
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    feature_names = [
        'CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',
        'DIS','RAD','TAX','PTRATIO','B','LSTAT'
    ]
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    return df

def prepare_xy(df, target_col='MEDV'):
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(float)
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_train_test(X_train, X_test):
    """
    Fit scaler on X_train and transform both train & test.
    Returns X_train_scaled, X_test_scaled, scaler_object
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return mse

def cross_val_mse(model, X, y, cv=5):
    # cross_val_score returns negative MSE when scoring='neg_mean_squared_error'
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)
    mean_mse = -scores.mean()
    return mean_mse

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
