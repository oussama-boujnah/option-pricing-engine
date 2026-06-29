# src/utils.py
import numpy as np

def validate_inputs_vectorized(S, K, T, r, sigma):
    """
    Ensures all inputs can be broadcast into NumPy arrays and meet strictly 
    defined institutional financial boundary conditions.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    if np.any(S <= 0):
        raise ValueError("Spot Price (S) must be strictly greater than zero.")
    if np.any(K <= 0):
        raise ValueError("Strike Price (K) must be strictly greater than zero.")
    if np.any(sigma <= 0):
        raise ValueError("Volatility (sigma) must be strictly greater than zero.")
    if np.any(T < 0):
        raise ValueError("Time to Expiry (T) cannot be negative.")
        
    return S, K, T, r, sigma