# In this we now use the validation set to tune our hyperparameters 
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix



# 1. Retrieve stock data from Yahoo Finance

ticker = "SPY"
start_date = "2010-01-01"
end_date = "2025-01-01"

df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True) # auto_adjust=True ensures that the data is adjusted for splits and dividends
prices = df["Close"].dropna().copy() # Closing prices are used to calculate returns
'''print(prices.head())'''

# Plot price vs time
plt.figure(figsize=(12, 6))
plt.plot(prices.index, prices.values, label="Adjusted Close Price")
plt.title(f"{ticker} Adjusted Close Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 2. Compute returns (i.e. percentage change in closing prices)

returns = prices.pct_change().dropna()
'''print(returns.head())'''

# Plot returns vs time
plt.figure(figsize=(12, 6))
plt.plot(returns.index, returns.values, label="Daily Returns")
plt.title(f"{ticker} Daily Returns Over Time")
plt.xlabel("Date")
plt.ylabel("Return")
plt.axhline(0, color="black", linewidth=1, alpha=0.7)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 3. Initialise hyperparameters

W_grid = [10, 20, 40]         # candidates for window length
H_grid = [3, 5, 10]           # candidates for prediction horizons
theta_grid = [0.01, 0.015, 0.02]  # candidates for spike thresholds
tau_grid = [0.4,0.5,0.6,0.7]    # candidates for decision thresholds
C_grid = [0.01, 0.1, 1.0, 10.0]  # candidates for inverse regularization strengths


# 4. Building a sliding-window dataset

def build_dataset(prices: pd.Series,returns: pd.Series, W: int, H: int, theta: float):
    """
    For each time t:
      x_t = [r_{t-W+1}, ..., r_t]
      y_t = 1 if max_{1<=h<=H} (P_{t+h} - P_t)/P_t > theta else 0
    """
    X = []
    y = []
    dates = []

    price_values = prices.values
    return_values = returns.values
    price_index = prices.index
    return_index = returns.index

    # Need t to have W past returns and H future prices available
    # Since returns start at prices index 1, map return position i to price position i+1
    for i in range(W - 1, len(return_values) - H):
        # Feature window: W returns ending at i
        x_t = return_values[i - W + 1:i + 1]

        price_pos_t = i + 1 # present day price is at index i+1
        P_t = price_values[price_pos_t] # present day price

        # Future returns over next H price steps relative to P_t
        future_prices = price_values[price_pos_t + 1: price_pos_t + H + 1]
        future_relative_returns = (future_prices - P_t) / P_t

        y_t = int(np.max(future_relative_returns) > theta) # 1 if there is a spike above the threshold, 0 otherwise

        X.append(x_t) 
        y.append(y_t)
        dates.append(return_index[i])

    X = np.array(X) # feature matrix
    y = np.array(y) # target vector
    dates = pd.Index(dates) # date index

    return X, y, dates


# 5. Train/validation/test split

# Note: The split will be 60% training, 20% validation, 20% test
# Additionally, the training/validation/test split will be time ordered,
# which means the model is trained only on past data and evaluated on
# future data. This avoids information leakage and mimics the real-world
# forecasting setting where we predict future prices using historical data.

def time_split_dataset(X, y, sample_dates):
    N = len(X)
    n_train = int(0.6 * N)
    n_val = int(0.2 * N)
    n_test = N - n_train - n_val

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]

    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]

    dates_train = sample_dates[:n_train]
    dates_val = sample_dates[n_train:n_train + n_val]
    dates_test = sample_dates[n_train + n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test, dates_train, dates_val, dates_test



# 6. Evaluation function

def evaluate_model(model, X_split, y_split, tau=0.5, split_name="Split"):
    prob = model.predict_proba(X_split)[:, 1] # An array of probabilities of a spike occuring
    pred = (prob >= tau).astype(int) # 1 if the probability of a spike occuring is greater than or equal to the threshold, 0 otherwise

    results = {
        "accuracy": accuracy_score(y_split, pred),
        "precision": precision_score(y_split, pred, zero_division=0),
        "recall": recall_score(y_split, pred, zero_division=0),
        "f1": f1_score(y_split, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_split, prob)
    }

    print(f"\n{split_name} metrics:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    return results, prob, pred


# 7. Iterative Logistic Classification

# 7. Iterative Logistic Classification

best_val_auc = -np.inf
best_config = None
best_model = None
best_scaler = None
best_data = None

for W in W_grid:
    for H in H_grid:
        for theta in theta_grid:
            X, y, sample_dates = build_dataset(prices, returns, W, H, theta)
            if X.ndim == 3 and X.shape[2] == 1:
                X = X.squeeze(-1)

            print("\n--------------------------------------------")
            print(f"Trying W={W}, H={H}, theta={theta}")
            print("Dataset shape:", X.shape)
            print("Positive label fraction:", y.mean())

            X_train, y_train, X_val, y_val, X_test, y_test, dates_train, dates_val, dates_test = time_split_dataset(X, y, sample_dates)

            print("Train size:", len(X_train))
            print("Val size:", len(X_val))
            print("Test size:", len(X_test))

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            for C in C_grid:
                model = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=1000)
                model.fit(X_train_scaled, y_train)

                val_prob = model.predict_proba(X_val_scaled)[:, 1]
                val_auc = roc_auc_score(y_val, val_prob)

                # Tune tau separately for this already-trained model
                best_tau_this_model = None
                best_f1_this_model = -np.inf

                for tau in tau_grid:
                    val_pred = (val_prob >= tau).astype(int)
                    val_f1 = f1_score(y_val, val_pred, zero_division=0)

                    print(f"W={W}, H={H}, theta={theta}, C={C}, tau={tau} | val_auc={val_auc:.4f}, val_f1={val_f1:.4f}")

                    if val_f1 > best_f1_this_model:
                        best_f1_this_model = val_f1
                        best_tau_this_model = tau

                # Compare models using AUC only once per model
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_config = {
                        "W": W,
                        "H": H,
                        "theta": theta,
                        "C": C,
                        "tau": best_tau_this_model
                    }
                    best_model = model
                    best_scaler = scaler
                    best_data = {
                        "X_train": X_train,
                        "y_train": y_train,
                        "X_val": X_val,
                        "y_val": y_val,
                        "X_test": X_test,
                        "y_test": y_test,
                        "dates_train": dates_train,
                        "dates_val": dates_val,
                        "dates_test": dates_test,
                        "X_train_scaled": X_train_scaled,
                        "X_val_scaled": X_val_scaled,
                        "X_test_scaled": X_test_scaled
                    }

print("\n============================================")
print("Best hyperparameters found:")
print(best_config)
print(f"Best validation ROC AUC: {best_val_auc:.4f}")
print("============================================")

# All of the values of the hyperparameters and dataset splits from the best model
model = best_model
scaler = best_scaler
W = best_config["W"]
H = best_config["H"]
theta = best_config["theta"]
C = best_config["C"]
tau = best_config["tau"]

X_train = best_data["X_train"]
y_train = best_data["y_train"]
X_val = best_data["X_val"]
y_val = best_data["y_val"]
X_test = best_data["X_test"]
y_test = best_data["y_test"]
dates_train = best_data["dates_train"]
dates_val = best_data["dates_val"]
dates_test = best_data["dates_test"]
X_train_scaled = best_data["X_train_scaled"]
X_val_scaled = best_data["X_val_scaled"]
X_test_scaled = best_data["X_test_scaled"]

val_results, val_prob, val_pred = evaluate_model(model, X_val_scaled, y_val, tau=tau, split_name="Validation")
test_results, test_prob, test_pred = evaluate_model(model, X_test_scaled, y_test, tau=tau, split_name="Test")


# 8. Inspect learned weights

weights = model.coef_[0] # The learned weights of the model
bias = model.intercept_[0] # The bias of the model

print("\nBias:", bias)
print("Weights shape:", weights.shape)
print("Weights:")
print(weights)


# 9. Put predictions into a dataframe

results_df = pd.DataFrame({
    "date": dates_test,
    "y_true": y_test,
    "y_prob": test_prob,
    "y_pred": test_pred
}).set_index("date")

print("\nTest prediction sample:")
print(results_df.head())


# 10. Visualisation of results 

# ROC AUC curve 
fpr, tpr, _ = roc_curve(y_test, test_prob)
plt.figure()
plt.plot(fpr, tpr, label="Model")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# Confusion Matrix
cm = confusion_matrix(y_test, test_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()