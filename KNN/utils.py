import numpy as np
from math import pi, exp, sqrt, cos


def uniform(u):
    return 1 / 2 if abs(u) < 1 else 0


def triangular(u):
    return 1 - abs(u) if abs(u) < 1 else 0


def epanechnikov(u):
    return 3 * (1 - u ** 2) / 4 if abs(u) < 1 else 0


def quartic(u):
    return 15 * (1 - u ** 2) ** 2 / 16 if abs(u) < 1 else 0


def triweight(u):
    return 35 * (1 - u ** 2) ** 3 / 32 if abs(u) < 1 else 0


def tricube(u):
    return 70 * (1 - abs(u) ** 3) ** 3 / 81 if abs(u) < 1 else 0


def gaussian(u):
    return np.exp(- u ** 2 / 2) / sqrt(2 * pi)


def cosine(u):
    return pi * cos(pi * u / 2) / 4 if abs(u) < 1 else 0


def logistic(u):
    return 1 / (np.exp(u) + 2 + np.exp(- u))


def sigmoid(u):
    return 2 / (pi * (np.exp(u) + np.exp(- u)))


def manhattan(x, y):
    ans = 0
    for j in range(len(x) - 1):
        ans += abs(x[j] - y[j])
    return ans


def euclidean(x, y):
    ans = 0
    for j in range(len(x) - 1):
        ans += (x[j] - y[j]) ** 2
    return sqrt(ans)


def chebyshev(x, y):
    ans = 0
    for j in range(len(x) - 1):
        ans = max(ans, abs(x[j] - y[j]))
    return ans
