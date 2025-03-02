% Load the data from pwl_fit_data.m  
% Ensure that pwl_fit_data.m defines x and y as vectors  
pwl_fit_data; % This will execute the script and load x and y into the workspace  

% Check if x and y are loaded correctly  
if ~exist('x', 'var') || ~exist('y', 'var')  
    error('Data variables x and y are not defined in pwl_fit_data.m. Please check the file.');  
end  

pkg load optim
m = length(x); % Number of data points  

% Define the knot points for different cases  
knots = {  
    [0, 1],          % Affine fit (no internal knots)  
    [0, 0.5, 1],     % 1 internal knot  
    [0, 0.33, 0.67, 1], % 2 internal knots  
    [0, 0.25, 0.5, 0.75, 1] % 3 internal knots  
};  

% Loop over each knot configuration  
for k = 1:length(knots)  
    a = knots{k}; % Knot points  
    K = length(a) - 1; % Number of segments  
    n = length(a); % Number of knot points  

    % Construct the Lagrange basis functions  
    F = zeros(m, n); % Basis function evaluations at data points  
    for i = 1:m  
        for j = 1:n  
            if j == 1  
                % First segment  
                if x(i) <= a(j+1)  
                    F(i, j) = (a(j+1) - x(i)) / (a(j+1) - a(j));  
                end  
            elseif j == n  
                % Last segment  
                if x(i) >= a(j-1)  
                    F(i, j) = (x(i) - a(j-1)) / (a(j) - a(j-1));  
                end  
            else  
                % Middle segments  
                if x(i) >= a(j-1) && x(i) <= a(j)  
                    F(i, j) = (x(i) - a(j-1)) / (a(j) - a(j-1));  
                elseif x(i) >= a(j) && x(i) <= a(j+1)  
                    F(i, j) = (a(j+1) - x(i)) / (a(j+1) - a(j));  
                end  
            end  
        end  
    end  

    % Quadratic programming setup  
    Q = 2 * (F' * F); % Quadratic term  
    q = -2 * (F' * y); % Linear term  

    % Convexity constraints  
    A = zeros(n-2, n);  
    for i = 1:n-2  
        A(i, i) = -1;  
        A(i, i+1) = 2;  
        A(i, i+2) = -1;  
    end  
    b = zeros(n-2, 1);  

    % Solve the quadratic programming problem  
    options = optimset('Display', 'off'); % Suppress output  
    c = quadprog(Q, q, A, b, [], [], [], [], [], options);  

    % Compute the least-squares fitting cost  
    fitting_cost = sum((F * c - y).^2);  

    % Plot the results  
    figure;  
    plot(x, y, 'bo', 'MarkerSize', 8, 'DisplayName', 'Data'); hold on;  
    xx = linspace(0, 1, 1000);  
    ff = zeros(size(xx));  
    for i = 1:length(xx)  
        for j = 1:n  
            if j == 1  
                if xx(i) <= a(j+1)  
                    ff(i) = ff(i) + c(j) * (a(j+1) - xx(i)) / (a(j+1) - a(j));  
                end  
            elseif j == n  
                if xx(i) >= a(j-1)  
                    ff(i) = ff(i) + c(j) * (xx(i) - a(j-1)) / (a(j) - a(j-1));  
                end  
            else  
                if xx(i) >= a(j-1) && xx(i) <= a(j)  
                    ff(i) = ff(i) + c(j) * (xx(i) - a(j-1)) / (a(j) - a(j-1));  
                elseif xx(i) >= a(j) && xx(i) <= a(j+1)  
                    ff(i) = ff(i) + c(j) * (a(j+1) - xx(i)) / (a(j+1) - a(j));  
                end  
            end  
        end  
    end  
    plot(xx, ff, 'r-', 'LineWidth', 2, 'DisplayName', 'Fit');  
    title(sprintf('Piecewise-Linear Fit with %d Internal Knots', n-2));  
    legend show;  
    xlabel('x');  
    ylabel('f(x)');  
    grid on;  

    % Display the fitting cost  
    fprintf('Fitting cost for %d internal knots: %.4f\n', n-2, fitting_cost);  
end 
