import numpy as np

A = np.array([[0, 0, 1, 1, 0, 1, 0],
              [0, 0, 0, 0, 1, 1, 1],
              [1, 0, 0, 1, 0, 1, 0],
              [1, 0, 1, 0, 0, 1, 0],
              [0, 1, 0, 0, 0, 1, 1],
              [1, 1, 1, 1, 1, 0, 1],
              [0, 1, 0, 0, 1, 1, 0]])

D = np.diag(np.sum(A, axis=1))

s = np.array([1] + [0] * 6) # start node 1 (for C)
ones = np.array([1] * 7)

def walks(t):
    return s@np.linalg.matrix_power(A, t)@ones

lambdas, E = np.linalg.eig(A)

print(np.round(lambdas, 3))
print(np.round(E, 3))
print(np.round(E[:, 0], 3))

print("Clustering Time")
print(np.linalg.inv(D@(D-np.eye(7)))@np.diag(np.linalg.matrix_power(A, 3))) # D is the degree matrix, which is diagonal with the degree of each node on the diagonal.
breakpoint()
print(np.linalg.matrix_power(A,2))