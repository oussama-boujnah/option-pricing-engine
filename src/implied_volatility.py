# src/implied_volatility.py
import numpy as np
from src.black_scholes import price_european_option_vectorized
from src.greeks import calculate_greeks_vectorized

def calculate_implied_volatility_vectorized(market_price, S, K, T, r, option_type: str = "call") -> np.ndarray:
    """
    High-speed vectorized solver mapping options prices back into Implied Volatility.
    Defensively wraps 0D inputs (scalars) into 1D vectors to prevent indexing crashes.
    """
    # Force everything to be at least a 1D array to protect matrix slicing operations
    market_price = np.atleast_1d(np.asarray(market_price, dtype=float))
    S = np.atleast_1d(np.asarray(S, dtype=float))
    K = np.atleast_1d(np.asarray(K, dtype=float))
    T = np.atleast_1d(np.asarray(T, dtype=float))
    r = np.atleast_1d(np.asarray(r, dtype=float))
    
    # Keep track of whether the original user input was a standalone scalar
    is_scalar = (market_price.ndim == 1 and len(market_price) == 1)
    
    # Initial consensus estimate (20% flat volatility seed)
    sigma_guess = np.full_like(market_price, 0.20)
    
    # Mask array to track items that haven't hit convergence thresholds yet
    active_mask = np.ones_like(market_price, dtype=bool)
    
    # Stage 1: Vectorized Newton-Raphson Core Iteration Loop
    for _ in range(25):
        if not np.any(active_mask):
            break
            
        current_price = price_european_option_vectorized(
            S[active_mask], K[active_mask], T[active_mask], r[active_mask], sigma_guess[active_mask], option_type
        )
        price_err = current_price - market_price[active_mask]
        
        # Pull out raw unscaled vega array segments
        greeks = calculate_greeks_vectorized(
            S[active_mask], K[active_mask], T[active_mask], r[active_mask], sigma_guess[active_mask], option_type
        )
        vega_raw = greeks["vega"] * 100.0
        
        # De-activate components meeting tolerance bounds or displaying zero-vega traps
        converged = np.abs(price_err) < 1e-6
        vega_collapsed = np.abs(vega_raw) < 1e-5
        
        # Safe updating index setup for array slice mappings
        active_indices = np.where(active_mask)[0]
        active_mask[active_indices[converged]] = False
        
        # Isolate remaining entries that require Newton update shifts
        to_update = active_mask[active_indices] & ~vega_collapsed
        if np.any(to_update):
            idx_to_update = active_indices[to_update]
            sigma_guess[idx_to_update] -= price_err[to_update] / vega_raw[to_update]
            
        # Hard flag any boundaries leaping outside logical financial constraints
        out_of_bounds = (sigma_guess < 0.001) | (sigma_guess > 4.0)
        active_mask[out_of_bounds] = False

    # Stage 2: Vectorized Bisection Fallback for remaining unresolved points
    if np.any(active_mask):
        low = np.full_like(market_price, 1e-5)
        high = np.full_like(market_price, 4.0)
        
        for _ in range(30):
            if not np.any(active_mask):
                break
                
            mid = (low + high) / 2.0
            current_price = price_european_option_vectorized(
                S[active_mask], K[active_mask], T[active_mask], r[active_mask], mid[active_mask], option_type
            )
            err = current_price - market_price[active_mask]
            
            converged = np.abs(err) < 1e-5
            active_indices = np.where(active_mask)[0]
            active_mask[active_indices[converged]] = False
            
            # Binary space partitioning criteria maps
            too_high = err > 0
            
            # Shift search space limits
            low[active_indices[~too_high]] = mid[active_indices[~too_high]]
            high[active_indices[too_high]] = mid[active_indices[too_high]]
            
        sigma_guess[active_mask] = mid[active_mask]
        
    # Clean conversion back to scalar if the user passed an individual item
    return float(sigma_guess[0]) if is_scalar else sigma_guess