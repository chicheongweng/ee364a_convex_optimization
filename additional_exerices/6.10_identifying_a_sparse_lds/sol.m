% Initialize random generators
randn('state', 118); 
rand('state', 118);

% Dimensions
n = 32; % State dimension
m = 16; % Input dimension
T = 100; % Number of time steps

% Generate sparse random matrices A and B
A = full(sprandn(n, n, 0.2)); % 20% nonzero entries
A = 0.95 * A / max(abs(eig(A))); % Scale A to make it stable
B = full(sprandn(n, m, 0.3)); % 30% nonzero entries

% Process noise covariance
W = eye(n); % Identity matrix
Whalf = sqrtm(W); % Square root of W

% Generate input and noise
us = 10 * randn(m, T-1); % Input signal
ws = Whalf * randn(n, T); % Process noise

% Simulate the system
xs = zeros(n, T); % State trajectory
xs(:, 1) = 50 * randn(n, 1); % Initial state
for t = 1:T-1
    xs(:, t+1) = A * xs(:, t) + B * us(:, t) + ws(:, t);
end

% Prepare data for optimization
X = xs(:, 1:T-1);    % x(t)
Y = xs(:, 2:T);      % x(t+1)
U = us;              % u(t)

% Check for NaN values in the data
if any(isnan(X(:))) || any(isnan(Y(:))) || any(isnan(U(:)))
    error('Input data contains NaN values.');
end

% Regularization parameters (tunable)
lambda_A = 0.1; % Sparsity penalty for A
lambda_B = 0.1; % Sparsity penalty for B

% Initialize A_hat and B_hat
A_hat = zeros(n, n);
B_hat = zeros(n, m);

% Gradient descent parameters
alpha = 1e-4; % Reduced step size
max_iter = 1000; % Maximum iterations
tol = 1e-6; % Convergence tolerance

% Proximal gradient descent for sparse optimization
for iter = 1:max_iter
    % Compute residual
    R = Y - A_hat * X - B_hat * U;
  
    % Compute gradients
    grad_A = -2 * (R * X') / T;
    grad_B = -2 * (R * U') / T;
  
    % Clip gradients to prevent unbounded updates
    grad_A = max(min(grad_A, 1e3), -1e3);
    grad_B = max(min(grad_B, 1e3), -1e3);
  
    % Gradient descent step
    A_hat = A_hat - alpha * grad_A;
    B_hat = B_hat - alpha * grad_B;
  
    % Apply soft-thresholding for sparsity (proximal operator for L1 norm)
    A_hat = sign(A_hat) .* max(abs(A_hat) - alpha * lambda_A, 0);
    B_hat = sign(B_hat) .* max(abs(B_hat) - alpha * lambda_B, 0);
  
    % Check for NaN values
    if any(isnan(A_hat(:))) || any(isnan(B_hat(:)))
        disp('NaN detected in A_hat or B_hat during optimization.');
        break;
    end
  
    % Check convergence
    if norm(grad_A, 'fro') < tol && norm(grad_B, 'fro') < tol
        break;
    end
end

% Display the results
disp('True A:');
disp(A);
disp('Estimated A_hat:');
disp(A_hat);
disp('True B:');
disp(B);
disp('Estimated B_hat:');
disp(B_hat);

% Verify plausibility
residual_matrix = Y - A_hat * X - B_hat * U; % Residual matrix
residual = sum(sum((W^(-1/2) * residual_matrix).^2)); % LHS of the inequality
plausible_threshold = n * (T-1) + 2 * sqrt(2 * n * (T-1)); % RHS of the inequality

% Display plausibility results
disp(['Residual (LHS): ', num2str(residual)]);
disp(['Plausibility threshold (RHS): ', num2str(plausible_threshold)]);
if residual <= plausible_threshold
    disp('The solution meets the plausibility requirement.');
else
    disp('The solution does NOT meet the plausibility requirement.');
end

% Compare the actual residual to the threshold
disp('--- Comparison of Residual and Threshold ---');
disp(['Actual Residual: ', num2str(residual)]);
disp(['Threshold: ', num2str(plausible_threshold)]);
if residual <= plausible_threshold
    disp('The residual is within the threshold. The solution is plausible.');
else
    disp('The residual exceeds the threshold. The solution is NOT plausible.');
end

% Compare with true A and B
false_positives_A = sum(sum((abs(A_hat) >= 0.01) & (abs(A) < 0.01)));
false_negatives_A = sum(sum((abs(A_hat) < 0.01) & (abs(A) >= 0.01)));
false_positives_B = sum(sum((abs(B_hat) >= 0.01) & (abs(B) < 0.01)));
false_negatives_B = sum(sum((abs(B_hat) < 0.01) & (abs(B) >= 0.01)));

disp(['False positives in A: ', num2str(false_positives_A)]);
disp(['False negatives in A: ', num2str(false_negatives_A)]);
disp(['False positives in B: ', num2str(false_positives_B)]);
disp(['False negatives in B: ', num2str(false_negatives_B)]);
