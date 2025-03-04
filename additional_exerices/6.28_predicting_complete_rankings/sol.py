import numpy as np
import cvxpy as cp

# Projection function
def Pi(y):
    """Returns the ranking of the argument."""
    y = np.array(y)
    if y.ndim == 1:
        ranking = np.argsort(np.argsort(y)) + 1
    else:
        ranking = np.argsort(np.argsort(y, axis=-1), axis=-1) + 1
    return ranking

# Data generation
N = N_test = 1000
d = 20
K = 10

np.random.seed(0)

# Generate the true theta matrix
theta_true = np.random.randn(K, d)
theta_true /= np.linalg.norm(theta_true, axis=1, keepdims=True)  # Normalize row-wise

# Sample x_i from standard Gaussian
X_test = np.hstack([np.random.randn(N_test, d - 1), np.ones((N_test, 1))])
X_train = np.hstack([np.random.randn(N, d - 1), np.ones((N, 1))])

# Generate the true features y = theta x and add noise to them
Y_train, Y_test = X_train.dot(theta_true.T), X_test.dot(theta_true.T)

# Add 15dB of noise to the observed y to generate noisy rankings
noise_snr = 15.0
sigma_noise = 10 ** (-0.05 * noise_snr) / np.sqrt(K)
Y_train, Y_test = Y_train + sigma_noise * np.random.randn(N, K), Y_test + sigma_noise * np.random.randn(N_test, K)

# Get the rankings of the observed noisy data
pi_train, pi_test = Pi(Y_train), Pi(Y_test)

# Define the optimization variable
Theta = cp.Variable((K, d))  # Predictor coefficient matrix

# Define the objective function (vectorized)
residuals = pi_train - X_train @ Theta.T
objective = cp.norm1(residuals) / (2 * N)

# Define the problem and solve
problem = cp.Problem(cp.Minimize(objective))
problem.solve()

# Optimal predictor matrix
Theta_opt = Theta.value

# Predict rankings for the test data
Y_test_pred = X_test @ Theta_opt.T  # Predicted scores
pi_test_pred = Pi(Y_test_pred)  # Predicted rankings

# Compute the average test error
test_error = (1 / (2 * N_test)) * np.sum(np.abs(pi_test - pi_test_pred))
print(f"Average Test Error: {test_error}")

# Compute the accuracy
correct_predictions = np.sum(pi_test == pi_test_pred)  # Count correct entries
total_entries = N_test * K  # Total number of entries in the test set
accuracy = correct_predictions / total_entries  # Compute accuracy
print(f"Accuracy: {accuracy * 100:.2f}%")

# Detailed accuracy breakdown
brackets = {"100% Match": 0, "25% Off": 0, "50% Off": 0, "100% Off": 0}

for i in range(N_test):
    # Compute the percentage of correct rankings for each test sample
    matches = np.sum(pi_test[i] == pi_test_pred[i])
    match_percentage = matches / K

    # Categorize into brackets
    if match_percentage == 1.0:
        brackets["100% Match"] += 1
    elif match_percentage >= 0.75:
        brackets["25% Off"] += 1
    elif match_percentage >= 0.50:
        brackets["50% Off"] += 1
    else:
        brackets["100% Off"] += 1

# Print detailed accuracy breakdown
print("\nDetailed Accuracy Breakdown:")
for category, count in brackets.items():
    percentage = (count / N_test) * 100
    print(f"{category}: {count} samples ({percentage:.2f}%)")

# Inspect predictions
print("\nSample of Predicted Scores (Y_test_pred):")
print(Y_test_pred[:5])  # Print the first 5 predicted score vectors
print("\nSample of Predicted Rankings (pi_test_pred):")
print(pi_test_pred[:5])  # Print the first 5 predicted rankings
print("\nSample of True Rankings (pi_test):")
print(pi_test[:5])  # Print the first 5 true rankings

