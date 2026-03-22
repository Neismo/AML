import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def geodesic_objective(params_path, start_point, end_point, num_steps):
    # Reshape flattened params back to path coordinates
    path = params_path.reshape((num_steps - 2, 2))
    full_path = np.vstack([start_point, path, end_point])
    
    total_length = 0
    for i in range(len(full_path) - 1):
        dt = 1.0 / num_steps
        theta = full_path[i]
        # Discrete derivative (velocity)
        v = (full_path[i+1] - full_path[i]) / dt
        
        # Metric G_bar calculation
        t1, t2 = theta[0], theta[1]
        # Metric term: 0.5 * (t2*v1 + t1*v2)^2
        # We use a small epsilon to keep it differentiable/numerical stable
        dist_sq = 0.5 * (t2 * v[0] + t1 * v[1])**2
        total_length += np.sqrt(dist_sq + 1e-6) * dt
        
    return total_length

# Configuration
start = np.array([1.0, 1.0])
end = np.array([21.0, 3/7])
N = 20 # Number of interior points

# Initial guess: straight line in parameter space
initial_path = np.linspace(start, end, N)[1:-1].flatten()

# Optimize
res = minimize(geodesic_objective, initial_path, args=(start, end, N), method='L-BFGS-B')
optimized_path = np.vstack([start, res.x.reshape((N-2, 2)), end])

# Output results
print(f"Path Length: {res.fun:.4f}")

# Visualization
plt.figure(figsize=(8, 5))
plt.plot(optimized_path[:, 0], optimized_path[:, 1], 'o-', label='Geodesic Path')
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$\theta_2$')
plt.title('Geodesic in ReLU Parameter Space')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()