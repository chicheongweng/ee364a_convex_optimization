rand('seed', 0); randn('seed', 0);

% Setup
m = 100;
k = 10; % max # permuted measurements
n = 20;
A = 500 * randn(m, n);
x_true = randn(n, 1); % true x value
y_true = A * x_true; % Without noise

% Build permuted indices
perm_idxs = randperm(m); perm_idxs = sort(perm_idxs(1:k));
new_pos = perm_idxs(randperm(k));

% True permutation matrix
P = eye(m); 
if k > 0
    P(perm_idxs, :) = P(new_pos, :);
end
perm_idxs(find(perm_idxs == new_pos)) = [];
y = P * y_true; % Apply permutation to the true y

clear new_pos;

% Algorithm implementation
max_iter = 100; % Maximum number of iterations
tol = 1e-6; % Convergence tolerance

% Step 1: Initial estimate of x (ignoring permutation)
x_est = (A' * A) \ (A' * y); % Ordinary least squares (no regularization)

% Debug: Initial x estimate
disp('Initial x_est:');
disp(x_est);

% Handle k = 0 case explicitly
if k == 0
    fprintf('No permutation (k = 0). Skipping iterations.\n');
    P_est = eye(m);
    x_diff = norm(x_true - x_est); % Difference between true and estimated x
    P_diff = norm(P - P_est, 'fro'); % Difference between true and estimated P
    fprintf('Difference between true x and estimated x: %f\n', x_diff);
    fprintf('Difference between true P and estimated P: %f\n', P_diff);
    return;
end

% Initialize variables
P_est = eye(m); % Initial guess for P (no permutation)
residual_norm = inf; % Initialize residual norm

% Iterative refinement
for iter = 1:max_iter
    % Step 2: Estimate permutation P using Hungarian algorithm
    % Compute cost matrix
    C = abs(A * x_est - y'); % Cost matrix (m x m)

    % Debug: Print cost matrix
    % fprintf('Iteration %d: Cost matrix C:\n', iter);
    % disp(C);

    % Solve assignment problem using munkres
    [assignment, ~] = munkres(C);

    % Construct permutation matrix P_est from assignment
    P_est = eye(m)(assignment, :);

    % Debug: Print estimated permutation matrix
    % fprintf('Iteration %d: Estimated P_est:\n', iter);
    % disp(P_est);

    % Step 3: Refine estimate of x
    x_new = (A' * A) \ (A' * P_est' * y); % Ordinary least squares (no regularization)

    % Debug: Print updated x estimate
    % fprintf('Iteration %d: Updated x_est:\n', iter);
    % disp(x_new);

    % Check for convergence
    residual_norm_new = norm(A * x_new - P_est' * y);
    if abs(residual_norm_new - residual_norm) < tol
        fprintf('Converged in %d iterations.\n', iter);
        break; % Converged
    end
    residual_norm = residual_norm_new;
    x_est = x_new; % Update x estimate
end

% Compare results
x_diff = norm(x_true - x_est); % Difference between true and estimated x
P_diff = norm(P - P_est, 'fro'); % Difference between true and estimated P

% Debug: Print final results
fprintf('Final Results:\n');
fprintf('True x:\n');
disp(x_true);
fprintf('Estimated x:\n');
disp(x_est);
fprintf('True P:\n');
disp(P);
fprintf('Estimated P:\n');
disp(P_est);
fprintf('Difference between true x and estimated x: %f\n', x_diff);
fprintf('Difference between true P and estimated P: %f\n', P_diff);
fprintf('Number of iterations: %d\n', iter);
