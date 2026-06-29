# tests/test_engine.py
import pytest
import numpy as np
from src.black_scholes import price_european_option_vectorized
from src.greeks import calculate_greeks_vectorized
from src.implied_volatility import calculate_implied_volatility_vectorized

def test_vectorized_put_call_parity():
    """Confirms array inputs perfectly validate under the Put-Call Parity Identity."""
    S = np.array([100.0, 105.0, 95.0])
    K = np.array([100.0, 100.0, 100.0])
    T = np.array([0.5, 0.5, 0.5])
    r = np.array([0.05, 0.05, 0.05])
    sigma = np.array([0.25, 0.25, 0.25])
    
    calls = price_european_option_vectorized(S, K, T, r, sigma, "call")
    puts = price_european_option_vectorized(S, K, T, r, sigma, "put")
    
    lhs = calls + K * np.exp(-r * T)
    rhs = puts + S
    assert np.allclose(lhs, rhs, atol=1e-5)

def test_iv_solver_convergence():
    """Ensures the IV module accurately reconstructs original volatility limits."""
    S, K, T, r, target_sigma = 100.0, 102.0, 0.25, 0.03, 0.32
    mkt_call = price_european_option_vectorized(S, K, T, r, target_sigma, "call")
    
    solved_iv = calculate_implied_volatility_vectorized(mkt_call, S, K, T, r, "call")
    assert np.isclose(solved_iv, target_sigma, atol=1e-4)