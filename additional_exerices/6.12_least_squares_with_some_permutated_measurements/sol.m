rand('seed', 0); randn('seed', 0);

% Setup
m = 100;
k = 40; % max # permuted measurements
n = 20;
A = 10 * randn(m, n);
x_true = randn(n, 1); % true x value
y_true = A * x_true + randn(m, 1);

% Build permuted indices
perm_idxs = randperm(m); perm_idxs = sort(perm_idxs(1:k));
new_pos = perm_idxs(randperm(k));

% True permutation matrix
P = eye(m); P(perm_idxs, :) = P(new_pos, :);
perm_idxs(find(perm_idxs == new_pos)) = [];
y = P * y_true;

clear new_pos;

% Algorithm implementation
max_iter = 100; % Maximum number of iterations
tol = 1e-6; % Convergence tolerance

% Step 1: Initial estimate of x (ignoring permutation)
x_est = A \ y; % Ordinary least squares

% Initialize variables
P_est = eye(m); % Initial guess for P (no permutation)
residual_norm = inf; % Initialize residual norm

% Iterative refinement
for iter = 1:max_iter
    % Step 2: Estimate permutation P
    [~, sorted_y_idx] = sort(y); % Sort observed measurements
    [~, sorted_Ax_idx] = sort(A * x_est); % Sort predicted measurements
    P_est = eye(m);
    P_est(sorted_y_idx, :) = P_est(sorted_Ax_idx, :); % Match sorted indices

    % Step 3: Refine estimate of x
    x_new = A \ (P_est' * y); % Least squares with current P estimate

    % Check for convergence
    residual_norm_new = norm(A * x_new - P_est' * y);
    if abs(residual_norm_new - residual_norm) < tol
        break; % Converged
    end
    residual_norm = residual_norm_new;
    x_est = x_new; % Update x estimate
end

% Compare results
x_diff = norm(x_true - x_est); % Difference between true and estimated x
P_diff = norm(P - P_est, 'fro'); % Difference between true and estimated P

% Display results
fprintf('Difference between true x and estimated x: %f\n', x_diff);
fprintf('Difference between true P and estimated P: %f\n', P_diff);
fprintf('Number of iterations: %d\n', iter);
