# src/greeks.py
import numpy as np
from scipy.stats import norm
from src.black_scholes import calculate_d1_d2_vectorized

def calculate_greeks_vectorized(S, K, T, r, sigma, option_type: str = "call") -> dict:
    """
    Computes analytical Greeks (Delta, Gamma, Theta, Vega, Rho) supporting full 
    array broadcasting. Handles the T=0 limit defensively to avoid NaN outputs.
    """
    op_type = option_type.lower().strip()
    if op_type not in ["call", "put"]:
        raise ValueError("Option type parameter must be explicitly 'call' or 'put'")
        
    d1, d2 = calculate_d1_d2_vectorized(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)
    discount = np.exp(-r * T)
    
    # Safe structural setup for T=0 boundary tracking
    safe_T = np.where(T == 0, 1e-9, T)
    
    # Gamma and Vega share common mathematical baselines independent of option flavor
    gamma = pdf_d1 / (S * sigma * np.sqrt(safe_T))
    vega = S * np.sqrt(safe_T) * pdf_d1
    
    if op_type == "call":
        delta = norm.cdf(d1)
        theta = (-(S * pdf_d1 * sigma) / (2.0 * np.sqrt(safe_T))) - (r * K * discount * norm.cdf(d2))
        rho = K * T * discount * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1.0
        theta = (-(S * pdf_d1 * sigma) / (2.0 * np.sqrt(safe_T))) + (r * K * discount * norm.cdf(-d2))
        rho = -K * T * discount * norm.cdf(-d2)
        
    # Zero out sensitivities at expiry boundary conditions where derivative fields collapse
    gamma = np.where(T == 0, 0.0, gamma)
    vega = np.where(T == 0, 0.0, vega)
    theta = np.where(T == 0, 0.0, theta)
    rho = np.where(T == 0, 0.0, rho)
    
    if T == 0:
         if op_type == "call":
             delta = np.where(S > K, 1.0, np.where(S < K, 0.0, 0.5))
         else:
             delta = np.where(S < K, -1.0, np.where(S > K, 0.0, -0.5))

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta / 365.0,  # Standardized to per-day risk metrics
        "vega": vega / 100.0,    # Standardized to 1 percentage point move in volatility
        "rho": rho / 100.0       # Standardized to 1 percentage point move in rates
    }