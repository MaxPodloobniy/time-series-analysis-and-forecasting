"""
Author: Maksym Podlubnyi IO-14
Implementation of Group 2 Requirements
"""

import pandas as pd
from tools import *
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# ------------------- Getting the Sample ------------------
def get_data():
    data_source = int(input("Enter the data source: 1- real data, 2- simulated "))

    # Fetch data
    if data_source == 1:
        data = pd.read_csv('nvda_stock_data.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.rename(columns={'Close      Close price adjusted for splits.': 'Close'})
        data = data.rename(columns={
            'Adj Close      Adjusted close price adjusted for splits and dividend and/or capital gain distributions.': 'Adj_Close'})
        data.dropna(inplace=True)
        data = data['Close'].values[::-1]

        # Visualize the loaded data
        plt.plot(data)
        plt.title("Real Data")
        plt.show()

        # Add outlier measurements to the data
        outlier_ratio = float(input("Enter the share of outliers in the real dataset (value between 0 and 1): "))
        data = outliers_add(data, outlier_ratio, 25)
    elif data_source == 2:
        # Generate the main trend
        sample_size = int(input("Enter the sample size "))
        data = cub_trend(sample_size)

        # Add normal noise
        distribution = random_normal(sample_size, 0, 360)
        data = data + distribution.astype('float64')

        # Add outlier measurements to the data
        outlier_ratio = float(input("Enter the share of outliers in the simulated dataset (value between 0 and 1): "))
        data = outliers_add(data, outlier_ratio, 1500)
    else:
        print("Error! Incorrect data source entered")
        exit()

    return data

sample = get_data()

# Display the data
plt.plot(sample)
plt.title("Sample with Outliers")
plt.show()

# ------------------- Outlier Detection and Sample Cleaning ------------------
print("Choose the outlier cleaning method:")
print("Isolation Forests - 1")
print("Local Outlier Factors - 2")
print("Z-score - 3")
print("DBSCAN - 4")
mode = int(input("Select the method: "))

# Attempt to remove the trend from the sample before cleaning
coefficients = find_MNK_coef(np.arange(len(sample)), sample, 3)
predictions = predict_polynomial(np.arange(len(sample)), coefficients)
residuals = sample - predictions

if mode == 1:
    # Isolation Forest
    X = residuals.reshape(-1, 1)
    clf = IsolationForest(max_samples='auto', contamination=0.01, random_state=42)
    anomalies = clf.fit_predict(X)
elif mode == 2:
    # Local Outlier Factor
    X = residuals.reshape(-1, 1)
    clf = LocalOutlierFactor(n_neighbors=10)
    anomalies = clf.fit_predict(X)
elif mode == 3:
    # Z-score
    is_anomaly = find_outliers_zscore(sample)
    anomalies = np.where(is_anomaly, -1, 1)
elif mode == 4:
    # DBSCAN
    X = residuals.reshape(-1, 1)
    dbscan = DBSCAN(eps=3, min_samples=5)
    anomalies = dbscan.fit_predict(X)
else:
    raise ValueError("Invalid mode. Choose 1, 2, 3, or 4.")

# Clean the sample from outliers
cleaned_data = sample[anomalies != -1]

plt.plot(cleaned_data)
plt.title("Sample Cleaned from Outliers")
plt.show()


# ------------------- Implementation of Recursive Smoothing ------------------
def hyperparameter_fit(measurements, loss_func, iter_num=100, learning_rate=0.05, is_gamma=0):
    losses = []

    alpha = np.random.uniform(0, 1)
    beta = np.random.uniform(0, 1)
    if is_gamma:
        gamma = np.random.uniform(0, 1)
    else:
        gamma = 0

    for i in range(iter_num):
        estimates = alpha_beta_gamma(measurements, alpha, beta, gamma)
        loss = loss_func(measurements[1:], estimates)
        losses.append(loss)

        grad_alpha, grad_beta, grad_gamma = numerical_gradients(loss_func, measurements, (alpha, beta, gamma))

        alpha -= grad_alpha * learning_rate
        beta -= grad_beta * learning_rate
        gamma -= grad_gamma * learning_rate * is_gamma

        if i % 50 == 0:
            print(f"\nIteration {i}")
            print(f"Loss value: {loss}")
            print(f"Hyperparameters: α = {alpha}; β = {beta}; γ = {gamma};")

    plt.plot(losses)
    plt.title("Loss Function over Iterations")
    plt.show()

    return alpha_beta_gamma(measurements, alpha, beta, gamma), alpha, beta, gamma


smoothed_data, pos_corr, trend_corr, acc_corr = hyperparameter_fit(sample, rmse_loss, 700, 0.02, 1)


# Determining the best polynomial fit for the data
print('\n------------------- Determining the Best Polynomial ------------------')
for i in range(1, 10):
    coefficients = find_MNK_coef(np.arange(len(cleaned_data)), cleaned_data, i)
    predictions = predict_polynomial(np.arange(len(cleaned_data)), coefficients)
    determination_coef = r_squared(cleaned_data, predictions)
    print(f'R-squared for the {i}-order polynomial: {determination_coef}')

# Create a polynomial of the specified order
level = int(input("Enter the polynomial order "))
coefficients = find_MNK_coef(np.arange(len(cleaned_data)), cleaned_data, level)
predictions = predict_polynomial(np.arange(len(cleaned_data)), coefficients)

print(f"Polynomial coefficients: {coefficients}")
print(f"R-squared: {r_squared(cleaned_data, predictions)}")
print(f"RMSE: {rmse_loss(cleaned_data, predictions)}")

# Plot the smoothed polynomial
plt.plot(np.arange(len(cleaned_data)), cleaned_data, label="Cleaned Data", color='blue')
plt.plot(np.arange(len(predictions)), predictions, label="Forecast on Cleaned Data", color='green')
plt.title(f"Smoothed {level}-order Polynomial")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

# Make future predictions
future_preds = predict_polynomial(np.arange(len(cleaned_data), int(len(cleaned_data)*1.5), 1), coefficients)
full_predictions = np.concatenate([predictions, future_preds])

# Plot the results
plt.plot(np.arange(len(cleaned_data)), cleaned_data, label="Cleaned Data", color='blue')
plt.plot(np.arange(len(predictions)), predictions, label="Forecast on Cleaned Data", color='green')
plt.plot(np.arange(len(cleaned_data), len(cleaned_data) + len(future_preds)), future_preds, label="Future Forecast", color='red', linestyle='--')

plt.title(f"{level}-order Polynomial: Forecast and Cleaned Data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
