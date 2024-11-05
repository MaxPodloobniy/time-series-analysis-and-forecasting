import numpy as np
import matplotlib.pyplot as plt

# ------------------- Z-score Algorithm for Detecting Anomalies ------------------
def find_outliers_zscore(data, threshold=3):
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    return z_scores > threshold

# ------------------- Implementation of Alpha-Beta-Gamma Filters ------------------
def alpha_beta(measurements, alpha, beta):
    estimates = []
    velocity = 0  # Trend
    estimate = measurements[0]  # Initial position estimate

    for measurement in measurements[1:]:  # Starting from the second value
        # Prediction
        estimate += velocity

        # Correction
        residual = measurement - estimate
        estimate += alpha * residual  # State correction
        velocity += beta * residual  # Trend correction

        estimates.append(estimate)
    return estimates

def alpha_beta_gamma(measurements, alpha, beta, gamma):
    estimates = []
    velocity = 0  # Trend
    acceleration = 0  # Acceleration
    estimate = measurements[0]  # Initial position estimate

    for measurement in measurements[1:]:
        # Prediction
        estimate += (velocity + 0.5 * acceleration)

        # Correction
        residual = measurement - estimate
        estimate += alpha * residual  # State correction
        velocity += beta * residual  # Trend correction
        acceleration += gamma * residual  # Acceleration correction

        estimates.append(estimate)
    return estimates

# ------------------- Implementation of Numerical Gradient Search ------------------
def numerical_gradients(loss_fn, measurements, params, epsilon=1e-6):
    gradients = np.zeros(len(params))

    for i in range(len(params)):
        curr_params = np.copy(params)

        # Forward step by epsilon
        curr_params[i] += epsilon
        predictions = alpha_beta_gamma(measurements, *curr_params)
        loss1 = loss_fn(measurements[1:], predictions) / len(measurements)

        # Backward step by epsilon
        curr_params[i] -= 2 * epsilon
        predictions = alpha_beta_gamma(measurements, *curr_params)
        loss2 = loss_fn(measurements[1:], predictions) / len(measurements)

        # Compute numerical gradient
        gradients[i] = (loss1 - loss2) / (2 * epsilon)

    return gradients

# ------------------- LOSS FUNCTIONS ------------------
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def rmse_loss(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    condition = np.abs(error) <= delta
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.mean(np.where(condition, squared_loss, linear_loss))

def mape_loss(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# ------------------- PRELIMINARY LAB FUNCTIONS ------------------

# ------------------- Generating a sample according to the normal distribution ------------------
def random_normal(n, mean, std_dev, is_plot=False):
    distribution = np.random.normal(mean, std_dev, n)

    exp_value = np.mean(distribution)
    variance = np.var(distribution)
    std_d = np.std(distribution)

    print("\n------------ Generating a sample according to the normal distribution ------------")
    print(f"Sample values {distribution[:7]}")
    print(f"Expected value: {exp_value}")
    print(f"Variance: {variance}")
    print(f"Standard deviation:  {std_d}")

    if is_plot:
        plt.hist(distribution, bins=22)
        plt.title("Sample from a normal distribution")
        plt.show()

    return distribution

# ------------------- Finding polynomial coefficients using Least Squares ------------------
def find_MNK_coef(x, y, polinom_level):
    # Creating the matrix
    matrix = np.ones((len(x), polinom_level + 1))
    for i in range(1, polinom_level + 1):
        matrix[:, i] = x ** i

    # Calculating the coefficients of polynomial regression
    coefficients = np.linalg.inv(matrix.T @ matrix) @ matrix.T @ y

    return coefficients

# ------------------- Calculating values using given coefficients ------------------
def predict_polynomial(x, coefficients):
    return sum(coef * x ** i for i, coef in enumerate(coefficients))

# ------------------- Calculating the coefficient of determination ------------------
def r_squared(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator)

# ------------------- Creating a cubic trend model ------------------
def cub_trend(n, a=0.00003, b=0.0005, c=0.2, d=7):
    values = np.arange(0, n, 1)
    values = a * (values ** 3) + b * (values ** 2) + c * values + d

    print("\n------------ Cubic trend model ------------")
    print(f"y = {a}*x^3 + {b}*x^2 + {c}*x + {d}")
    print(f"First values: {values[:7]}")

    plt.plot(values)
    plt.title("Cubic trend model")
    plt.show()

    return values

# ------------------- Adding anomalous outliers to the sample ------------------
def outliers_add(sample, out_proportion, standart_dev):
    sample_size = sample.shape[0]
    out_num = int(sample_size * out_proportion)
    outliers = random_normal(out_num, 0, standart_dev)
    index_to_outlie = np.random.choice(np.arange(sample_size), size=out_num, replace=False)

    for i in range(out_num):
        sample[index_to_outlie[i]] += outliers[i]

    return sample
