# c(t) = [2t + 1, -t ** 2]
import numpy as np


def curve_length(num_steps: int) -> float:
    ts = np.linspace(0, 1, num_steps+1)
    print(ts[0], ts[-1])
    points = np.array([[2 * t + 1, -t ** 2] for t in ts])
    print(points[0], points[-1])

    total_length = 0
    for i in range(len(points) - 1):
        length = points[i] - points[i + 1]
        total_length += np.linalg.norm(length)

    return total_length

def curve_length_analytical() -> float:    
    from scipy.integrate import quad
    
    def speed(t):
        return 2 * np.sqrt(1 + t**2)
    
    length, _ = quad(speed, 0, 1)
    return length

if __name__ == "__main__":
    length = curve_length(100)
    print(f"Curve length: {length:.4f}")
    analytical_length = curve_length_analytical()
    print(f"Analytical length: {analytical_length:.4f}")