# misc.py - simple helpers for loading data, training, evaluation
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

def load_data():
    from sklearn.datasets import load_diabetes
    d = load_diabetes()
    X, y = d.data, d.target
    return X, y

def split_and_scale(X, y, test_size=0.2, random_state=42, scale=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

def train_and_eval(model, X_train, X_test, y_train, y_test, cv=5):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    cv_scores = cross_val_score(model, np.vstack((X_train, X_test)), np.concatenate((y_train, y_test)),
                                scoring='neg_mean_squared_error', cv=cv)
    cv_mse = -cv_scores.mean()
    return test_mse, cv_mse

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
