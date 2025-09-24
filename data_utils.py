import numpy as np

def movavg(x, k=10):
    if len(x) < 1: return x
    k = max(1, min(k, len(x)))
    return np.convolve(x, np.ones(k)/k, mode="valid")
