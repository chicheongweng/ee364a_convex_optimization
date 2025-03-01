import numpy as np  
import cvxpy as cp  

# Problem Data  
A_bar = np.array([[60, 45, -8],  
                  [90, 30, -30],  
                  [0, -8, -4],  
                  [30, 10, -10]])  # Nominal matrix  
R = 0.01 * np.ones((4, 3))  # Interval radii (reduced to 0.05)  
b = np.array([-6, -3, 18, -9])  # Target vector  

# (1) Solve the nominal least-squares problem  
x_ls = np.linalg.lstsq(A_bar, b, rcond=None)[0]  # Standard least-squares solution  
nominal_residual_ls = np.linalg.norm(A_bar @ x_ls - b)  # Nominal residual norm for LS  

# (2) Solve the robust least-squares problem using CVXPY  
x_rls = cp.Variable(3)  # Robust least-squares solution (3 variables)  
t = cp.Variable(4)  # Auxiliary variables for robust constraints  

# Objective: Minimize the nominal residual norm + robust terms  
objective = cp.Minimize(cp.norm(A_bar @ x_rls - b, 2) + cp.sum(t))  

# Constraints: Robust constraints for each row of A_bar  
constraints = []  
for i in range(4):  
    constraints.append(cp.norm(cp.multiply(R[i, :], x_rls), 2) <= t[i])  # Use elementwise multiplication  

# Solve the problem  
problem = cp.Problem(objective, constraints)  
problem.solve()  

# Extract the robust solution  
x_rls_value = x_rls.value  
nominal_residual_rls = np.linalg.norm(A_bar @ x_rls_value - b)  # Nominal residual norm for RLS  
worst_case_residual_rls = nominal_residual_rls + np.sum(t.value)  # Worst-case residual norm for RLS  

# Display Results  
print("\n\n")
print("Nominal LS Solution: x_ls =", x_ls)  
print("Nominal Residual Norm (LS):", nominal_residual_ls)  
print()  
print("Robust LS Solution: x_rls =", x_rls_value)  
print("Nominal Residual Norm (RLS):", nominal_residual_rls)  
print("Worst-Case Residual Norm (RLS):", worst_case_residual_rls)  
