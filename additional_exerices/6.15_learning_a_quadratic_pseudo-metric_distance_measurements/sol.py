# Data for learning a quadratic metric
# You should use X, Y, d as well as X_test, Y_test, d_test

import numpy as np
from scipy import linalg as la
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(8)

# Problem dimensions
n = 5  # Dimension
N = 100  # Number of samples
N_test = 10  # Samples for test set

# Generate data
X = np.random.randn(n, N)
Y = np.random.randn(n, N)

X_test = np.random.randn(n, N_test)
Y_test = np.random.randn(n, N_test)

# Generate ground truth P and distances
P = np.random.randn(n, n)
P = P.dot(P.T) + np.identity(n)  # Ensure P is positive definite
sqrtP = la.sqrtm(P)

d = np.linalg.norm(sqrtP.dot(X - Y), axis=0)
d = np.maximum(d + np.random.randn(N), 0)  # Add noise and ensure non-negativity
d_test = np.linalg.norm(sqrtP.dot(X_test - Y_test), axis=0)
d_test = np.maximum(d_test + np.random.randn(N_test), 0)

# Learning the quadratic pseudo-metric
# Define the optimization variables and problem
P_var = cp.Variable((n, n), symmetric=True)  # Symmetric matrix P
constraints = [P_var >> 0]  # P ⪰ 0 (positive semidefinite constraint)

# Define the objective function
objective = 0
for i in range(N):
    z_i = X[:, i] - Y[:, i]
    objective += cp.square(d[i]**2 - cp.quad_form(z_i, P_var))

objective = objective / N  # Mean squared error

# Solve the optimization problem
problem = cp.Problem(cp.Minimize(objective), constraints)
problem.solve()

# Extract the learned P
P_learned = P_var.value

# Compute the mean squared error on the training set
train_errors = []
for i in range(N):
    z_i = X[:, i] - Y[:, i]
    d_pred = np.sqrt(z_i.T @ P_learned @ z_i)
    train_errors.append((d[i] - d_pred)**2)

train_mse = np.mean(train_errors)

# Compute the mean squared error on the test set
test_errors = []
for i in range(N_test):
    z_i = X_test[:, i] - Y_test[:, i]
    d_pred = np.sqrt(z_i.T @ P_learned @ z_i)
    test_errors.append((d_test[i] - d_pred)**2)

test_mse = np.mean(test_errors)

# Assessment 1: Compare MSE to the scale of distances
mean_d_train = np.mean(d)
std_d_train = np.std(d)
mean_d_test = np.mean(d_test)
std_d_test = np.std(d_test)

# Assessment 2: Compare to baseline (Euclidean distance, P = I)
baseline_train_errors = []
for i in range(N):
    z_i = X[:, i] - Y[:, i]
    d_pred_baseline = np.sqrt(z_i.T @ np.eye(n) @ z_i)  # Euclidean distance
    baseline_train_errors.append((d[i] - d_pred_baseline)**2)

baseline_train_mse = np.mean(baseline_train_errors)

baseline_test_errors = []
for i in range(N_test):
    z_i = X_test[:, i] - Y_test[:, i]
    d_pred_baseline = np.sqrt(z_i.T @ np.eye(n) @ z_i)  # Euclidean distance
    baseline_test_errors.append((d_test[i] - d_pred_baseline)**2)

baseline_test_mse = np.mean(baseline_test_errors)

# Report results
print("Optimal mean squared distance error on training set:", train_mse)
print("Mean squared distance error on test set:", test_mse)
print("Baseline mean squared distance error on training set (Euclidean):", baseline_train_mse)
print("Baseline mean squared distance error on test set (Euclidean):", baseline_test_mse)
print()
print("Scale of distances (training set): mean =", mean_d_train, ", std =", std_d_train)
print("Scale of distances (test set): mean =", mean_d_test, ", std =", std_d_test)

# Visualization
# Project data to 2D using PCA for visualization
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X.T).T  # Project X to 2D
Y_2d = pca.transform(Y.T).T      # Project Y to 2D

# Plot the points and distances
plt.figure(figsize=(10, 8))
for i in range(N):
    plt.plot([X_2d[0, i], Y_2d[0, i]], [X_2d[1, i], Y_2d[1, i]], 'k--', alpha=0.5)  # Line between x and y
    plt.scatter(X_2d[0, i], X_2d[1, i], color='blue', label='X (x points)' if i == 0 else "", alpha=0.7)
    plt.scatter(Y_2d[0, i], Y_2d[1, i], color='red', label='Y (y points)' if i == 0 else "", alpha=0.7)

plt.title("Visualization of Points X and Y with Distances d")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
plt.show()
