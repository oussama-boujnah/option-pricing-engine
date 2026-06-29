# src/black_scholes.py
import numpy as np
from scipy.stats import norm
from src.utils import validate_inputs_vectorized

def calculate_d1_d2_vectorized(S, K, T, r, sigma):
    """
    Computes intermediate components d1 and d2 across arrays safely.
    Uses array masks to eliminate division-by-zero errors at T=0.
    """
    S, K, T, r, sigma = validate_inputs_vectorized(S, K, T, r, sigma)
    
    # Prevent division by zero for expired contracts using an infinitesimal time buffer
    safe_T = np.where(T == 0, 1e-9, T)
    denominator = sigma * np.sqrt(safe_T)
    drift = (r + (sigma ** 2) / 2.0) * safe_T
    
    d1 = (np.log(S / K) + drift) / denominator
    d2 = d1 - denominator
    
    # Apply explicit asymptotic limits for true boundary state when T exactly equals 0
    d1 = np.where(T == 0, np.where(S > K, np.inf, np.where(S < K, -np.inf, 0.0)), d1)
    d2 = np.where(T == 0, np.where(S > K, np.inf, np.where(S < K, -np.inf, 0.0)), d2)
    
    return d1, d2

def price_european_option_vectorized(S, K, T, r, sigma, option_type: str = "call") -> np.ndarray:
    """
    Prices an entire matrix, curve, or array of European options simultaneously 
    utilizing analytical Black-Scholes-Merton equations.
    """
    S, K, T, r, sigma = validate_inputs_vectorized(S, K, T, r, sigma)
    d1, d2 = calculate_d1_d2_vectorized(S, K, T, r, sigma)
    
    discount = np.exp(-r * T)
    op_type = option_type.lower().strip()
    
    if op_type == "call":
        price = S * norm.cdf(d1) - K * discount * norm.cdf(d2)
        # Force analytical boundary compliance for intrinsic value at expiry
        price = np.where(T == 0, np.maximum(S - K, 0.0), price)
    elif op_type == "put":
        price = K * discount * norm.cdf(-d2) - S * norm.cdf(-d1)
        # Force analytical boundary compliance for intrinsic value at expiry
        price = np.where(T == 0, np.maximum(K - S, 0.0), price)
    else:
        raise ValueError("Option type parameter must be explicitly 'call' or 'put'")
        
    return np.maximum(price, 0.0)