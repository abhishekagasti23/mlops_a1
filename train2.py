# train2.py
import warnings
warnings.filterwarnings("ignore")

from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from misc import load_data, prepare_xy, split_data, scale_train_test, train_model, evaluate_model, cross_val_mse, save_model
import numpy as np

def main():
    df = load_data()
    X, y = prepare_xy(df)

    # split first, then scale (scale must be fit on train only)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled, scaler = scale_train_test(X_train, X_test)

    # For cross-val we need whole X scaled
    scaler_all = StandardScaler()
    X_scaled_all = scaler_all.fit_transform(X)

    model = KernelRidge(alpha=1.0, kernel='rbf')
    model = train_model(model, X_train_scaled, y_train)

    test_mse = evaluate_model(model, X_test_scaled, y_test)
    cv_mse = cross_val_mse(KernelRidge(alpha=1.0, kernel='rbf'), X_scaled_all, y, cv=5)

    print("=== KernelRidge Results ===")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"5-fold CV average MSE: {cv_mse:.6f}")

    save_model(model, "kernelridge_model.joblib")

if __name__ == "__main__":
    main()
