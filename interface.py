import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importation de vos modules mathématiques locaux
from src.black_scholes import price_european_option_vectorized
from src.implied_volatility import calculate_implied_volatility_vectorized

# ============================================================================
# 1. CONFIGURATION DU DASHBOARD INSTITUTIONNEL
# ============================================================================
st.set_page_config(
    page_title="Pro Quant Engine",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé avancé
st.markdown("""
    <style>
    /* Badges de Moneyness */
    .moneyness-badge {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .itm { 
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724; 
        border: 2px solid #155724;
    }
    .otm { 
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24; 
        border: 2px solid #721c24;
    }
    .atm { 
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        color: #856404; 
        border: 2px solid #856404;
    }
    
    /* Styles de métriques institutionnelles */
    .risk-metric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E86C1;
        margin: 0.5rem 0;
    }
    
    .greek-chart-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #2E86C1;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌌 Institutional Options Desk")
st.markdown("""
**Moteur de pricing Black-Scholes institutionnel** | 
Analyse avancée des Grecs • Monte Carlo VaR • Profils de risque • Portfolio Analytics
""")
st.markdown("---")

# ============================================================================
# 2. BARRE LATÉRALE : PARAMÈTRES DE MARCHÉ
# ============================================================================
st.sidebar.header("⚙️ Market Parameters")

# Section 1: Paramètres principaux
with st.sidebar.expander("📊 Underlying & Strike", expanded=True):
    S = st.number_input("Underlying Asset Price (S)", value=100.0, step=1.0, key="spot")
    K = st.number_input("Strike Price (K)", value=100.0, step=1.0, key="strike")
    moneyness_pct = ((S - K) / K * 100) if K != 0 else 0
    st.caption(f"Moneyness: {moneyness_pct:+.1f}%")

# Section 2: Durée et taux
with st.sidebar.expander("⏱️ Time & Rates", expanded=True):
    T = st.number_input("Time to Maturity (Years)", min_value=0.001, max_value=5.0, value=0.5, step=0.05, key="time")
    r = st.number_input("Risk-Free Rate (r)", value=0.05, step=0.01, key="rate")
    days_to_expiry = int(T * 365)
    st.caption(f"Days to Expiry: {days_to_expiry}")

# Section 3: Volatilité
with st.sidebar.expander("📈 Volatility", expanded=True):
    sigma = st.slider("Implied Volatility (σ)", min_value=0.01, max_value=1.50, value=0.20, step=0.01, key="vol")
    
    # Affichage du percentile de volatilité historique
    vix_proxy = sigma * 100  # Approximation simple
    st.caption(f"IV Level: {vix_proxy:.1f} (proxy VIX: {vix_proxy/2.5:.0f})")

# Section 4: Options de contrat
with st.sidebar.expander("🔧 Contract Settings", expanded=True):
    option_type = st.radio("Option Type", ["call", "put"], key="opt_type")
    dividend_yield = st.slider("Dividend Yield (q)", 0.0, 0.10, 0.0, step=0.01, key="div")
    
    # Ajustement du prix à terme pour les dividendes
    F = S * np.exp((r - dividend_yield) * T)
    st.caption(f"Forward Price: {F:.2f}")

st.sidebar.markdown("---")

# ============================================================================
# 3. FONCTIONS MATHÉMATIQUES AVANCÉES
# ============================================================================

def calculate_greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    """Calcul complet des Grecs avec ajustement dividendes"""
    T = max(0.0001, T)
    sigma = max(0.0001, sigma)
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    
    if option_type == "call":
        delta = np.exp(-q * T) * cdf_d1
        theta = (-S * pdf_d1 * sigma * np.exp(-q * T) / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * cdf_d2 
                 + q * S * np.exp(-q * T) * cdf_d1)
        rho = K * T * np.exp(-r * T) * cdf_d2
    else:
        delta = np.exp(-q * T) * (cdf_d1 - 1.0)
        theta = (-S * pdf_d1 * sigma * np.exp(-q * T) / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2) 
                 - q * S * np.exp(-q * T) * norm.cdf(-d1))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    gamma = pdf_d1 * np.exp(-q * T) / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * pdf_d1 * np.exp(-q * T) / 100.0
    theta_per_day = theta / 365.0
    
    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta_per_day,
        "rho": rho / 100.0,
        "d1": d1,
        "d2": d2
    }

def price_option(S, K, T, r, sigma, option_type="call", q=0.0):
    """Prix Black-Scholes avec dividendes"""
    T = max(0.0001, T)
    sigma = max(0.0001, sigma)
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    return price

def monte_carlo_simulation(S, K, T, r, sigma, n_simulations=10000, n_steps=252, option_type="call", q=0.0):
    """Simulation Monte Carlo avec trajectoires brownienne"""
    dt = T / n_steps
    Z = np.random.standard_normal((n_simulations, n_steps))
    
    # Génération des trajectoires
    S_paths = np.zeros((n_simulations, n_steps + 1))
    S_paths[:, 0] = S
    
    for i in range(n_steps):
        S_paths[:, i + 1] = S_paths[:, i] * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, i])
    
    # Calcul des payoffs
    S_terminal = S_paths[:, -1]
    
    if option_type == "call":
        payoff = np.maximum(S_terminal - K, 0)
    else:
        payoff = np.maximum(K - S_terminal, 0)
    
    # Actualisation
    option_price = np.exp(-r * T) * np.mean(payoff)
    std_error = np.exp(-r * T) * np.std(payoff) / np.sqrt(n_simulations)
    
    # Calcul du VaR et CVaR
    discounted_payoff = np.exp(-r * T) * payoff
    var_95 = np.percentile(discounted_payoff, 5)
    cvar_95 = np.mean(discounted_payoff[discounted_payoff <= var_95])
    
    return {
        "price": option_price,
        "std_error": std_error,
        "payoffs": payoff,
        "paths": S_paths,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "terminal_prices": S_terminal,
        "discounted_payoffs": discounted_payoff
    }

def calculate_payoff_diagram(S_range, K, T, r, sigma, option_type, q=0.0, forward_contract=False):
    """Diagramme de payoff (P&L à l'expiration)"""
    if forward_contract:
        payoff = S_range - K
        intrinsic = S_range - K
    else:
        prices = [price_option(s, K, T, r, sigma, option_type, q) for s in S_range]
        current_price = price_option(S_range[len(S_range)//2], K, T, r, sigma, option_type, q)
        
        if option_type == "call":
            intrinsic = np.maximum(S_range - K, 0)
            payoff = intrinsic - current_price
        else:
            intrinsic = np.maximum(K - S_range, 0)
            payoff = intrinsic - current_price
    
    return intrinsic, payoff

def create_greeks_heatmap(S_range, sigma_range, K, T, r, option_type, greek_name="delta", q=0.0):
    """Heatmap des Grecs sur spot/volatilité"""
    greeks_grid = np.zeros((len(sigma_range), len(S_range)))
    
    for i, sig in enumerate(sigma_range):
        for j, spot in enumerate(S_range):
            greeks_dict = calculate_greeks(spot, K, T, r, sig, option_type, q)
            greeks_grid[i, j] = greeks_dict[greek_name]
    
    return greeks_grid

# ============================================================================
# 4. CALCULS PRINCIPAUX
# ============================================================================

# Calculs BS classiques
greek_metrics = calculate_greeks(S, K, T, r, sigma, option_type, dividend_yield)
price = price_option(S, K, T, r, sigma, option_type, dividend_yield)

# Monte Carlo (en background)
if st.sidebar.checkbox("Enable Monte Carlo Analysis", value=True, key="mc_toggle"):
    with st.spinner("🎲 Simulant 10,000 trajectoires..."):
        mc_result = monte_carlo_simulation(S, K, T, r, sigma, n_simulations=10000, option_type=option_type, q=dividend_yield)
    mc_enabled = True
else:
    mc_enabled = False

# Moneyness
threshold = K * 0.01
if abs(S - K) <= threshold:
    moneyness, css_class = "At-The-Money (ATM)", "atm"
elif (option_type == "call" and S > K) or (option_type == "put" and S < K):
    moneyness, css_class = "In-The-Money (ITM)", "itm"
else:
    moneyness, css_class = "Out-Of-The-Money (OTM)", "otm"

# ============================================================================
# 5. SECTION PRINCIPALE : KPI DASHBOARD
# ============================================================================

# Badge moneyness
st.markdown(f'<div class="moneyness-badge {css_class}">État Actuel : {moneyness}</div>', unsafe_allow_html=True)

# KPI Principal
col_price, col_intrinsic, col_extrinsic = st.columns(3)

intrinsic_value = max(S - K, 0) if option_type == "call" else max(K - S, 0)
time_value = price - intrinsic_value

col_price.metric("📍 Premium (Prix BS)", f"{price:.4f} €", delta=None)
col_intrinsic.metric("💎 Valeur Intrinsèque", f"{intrinsic_value:.4f} €")
col_extrinsic.metric("⏳ Valeur Temps", f"{time_value:.4f} €")

# Grecs principaux
col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("Δ Delta", f"{greek_metrics['delta']:.4f}", help="Sensibilité au prix du sous-jacent")
col2.metric("Γ Gamma", f"{greek_metrics['gamma']:.6f}", help="Accélération du delta")
col3.metric("ν Vega", f"{greek_metrics['vega']:.4f}", help="Sensibilité à la volatilité")
col4.metric("Θ Theta", f"{greek_metrics['theta']:.4f}", help="Érosion du temps (par jour)")
col5.metric("ρ Rho", f"{greek_metrics['rho']:.4f}", help="Sensibilité aux taux")
col6.metric("Moneyness", f"{moneyness_pct:+.1f}%", help=f"(S-K)/K")

st.markdown("---")

# Monte Carlo résultats
if mc_enabled:
    mc_col1, mc_col2, mc_col3, mc_col4 = st.columns(4)
    mc_col1.metric("🎲 MC Price", f"{mc_result['price']:.4f} €", delta=f"{mc_result['price'] - price:+.4f}")
    mc_col2.metric("95% VaR", f"{mc_result['var_95']:.4f} €", help="Perte maximale avec 95% confiance")
    mc_col3.metric("95% CVaR (ES)", f"{mc_result['cvar_95']:.4f} €", help="Perte moyenne au-delà du VaR")
    mc_col4.metric("Std Error", f"±{mc_result['std_error']:.6f}", help="Erreur standard de l'estimation")

st.markdown("---")

# ============================================================================
# 6. ONGLETS DE VISUALISATION AVANCÉE
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Risk Greeks",
    "💰 Payoff & P&L",
    "🎲 Monte Carlo",
    "🔥 Greeks Heatmaps",
    "📊 Portfolio Strategy"
])

# ============================================================================
# TAB 1: RISK GREEKS ANALYSIS
# ============================================================================
with tab1:
    st.subheader("📊 Analyse de Sensibilité des Grecs")
    
    # Génération des données
    spot_range = np.linspace(max(1.0, S * 0.5), S * 1.5, 100)
    
    prices, deltas, gammas, vegas, thetas, rhos = [], [], [], [], [], []
    for s in spot_range:
        greek_dict = calculate_greeks(s, K, T, r, sigma, option_type, dividend_yield)
        p = price_option(s, K, T, r, sigma, option_type, dividend_yield)
        
        prices.append(p)
        deltas.append(greek_dict["delta"])
        gammas.append(greek_dict["gamma"])
        vegas.append(greek_dict["vega"])
        thetas.append(greek_dict["theta"])
        rhos.append(greek_dict["rho"])
    
    df_greeks = pd.DataFrame({
        "Spot": spot_range,
        "Premium": prices,
        "Delta": deltas,
        "Gamma": gammas,
        "Vega": vegas,
        "Theta": thetas,
        "Rho": rhos
    })
    
    # Graphiques Plotly (interactifs)
    chart_cols = st.columns(2)
    
    # Chart 1: Premium et Delta
    with chart_cols[0]:
        fig_premium = go.Figure()
        fig_premium.add_trace(go.Scatter(
            x=df_greeks["Spot"], y=df_greeks["Premium"],
            mode='lines', name='Premium', line=dict(color='#2E86C1', width=3)
        ))
        fig_premium.add_vline(x=S, line_dash="dash", line_color="green", annotation_text="Current Spot")
        fig_premium.update_layout(
            title="Option Premium vs Spot Price",
            xaxis_title="Spot Price (S)",
            yaxis_title="Premium (€)",
            hovermode='x unified',
            template="plotly_white"
        )
        st.plotly_chart(fig_premium, use_container_width=True)
    
    # Chart 2: Delta et Gamma
    with chart_cols[1]:
        fig_delta_gamma = go.Figure()
        fig_delta_gamma.add_trace(go.Scatter(
            x=df_greeks["Spot"], y=df_greeks["Delta"],
            mode='lines', name='Delta', line=dict(color='#E74C3C', width=3)
        ))
        fig_delta_gamma.add_trace(go.Scatter(
            x=df_greeks["Spot"], y=df_greeks["Gamma"],
            mode='lines', name='Gamma', line=dict(color='#8E44AD', width=2), yaxis="y2"
        ))
        fig_delta_gamma.add_vline(x=S, line_dash="dash", line_color="green", annotation_text="Current Spot")
        fig_delta_gamma.update_layout(
            title="Delta & Gamma Exposure",
            xaxis_title="Spot Price (S)",
            yaxis=dict(title="Delta", side='left'),
            yaxis2=dict(title="Gamma", side='right', overlaying='y'),
            hovermode='x unified',
            template="plotly_white"
        )
        st.plotly_chart(fig_delta_gamma, use_container_width=True)
    
    # Chart 3: Vega et Theta (érosion du temps)
    chart_cols2 = st.columns(2)
    
    with chart_cols2[0]:
        fig_vega = go.Figure()
        fig_vega.add_trace(go.Scatter(
            x=df_greeks["Spot"], y=df_greeks["Vega"],
            mode='lines', name='Vega', line=dict(color='#F39C12', width=3),
            fill='tozeroy', fillcolor='rgba(243, 156, 18, 0.2)'
        ))
        fig_vega.add_vline(x=S, line_dash="dash", line_color="green")
        fig_vega.update_layout(
            title="Volatility Exposure (Vega)",
            xaxis_title="Spot Price (S)",
            yaxis_title="Vega (per 1% vol change)",
            template="plotly_white",
            hovermode='x unified'
        )
        st.plotly_chart(fig_vega, use_container_width=True)
    
    with chart_cols2[1]:
        fig_theta = go.Figure()
        fig_theta.add_trace(go.Scatter(
            x=df_greeks["Spot"], y=df_greeks["Theta"],
            mode='lines', name='Theta', line=dict(color='#16A085', width=3),
            fill='tozeroy', fillcolor='rgba(22, 160, 133, 0.2)'
        ))
        fig_theta.add_vline(x=S, line_dash="dash", line_color="green")
        fig_theta.update_layout(
            title="Time Decay (Theta) - per day",
            xaxis_title="Spot Price (S)",
            yaxis_title="Theta (€/day)",
            template="plotly_white",
            hovermode='x unified'
        )
        st.plotly_chart(fig_theta, use_container_width=True)

# ============================================================================
# TAB 2: PAYOFF & P&L ANALYSIS
# ============================================================================
with tab2:
    st.subheader("💰 Diagramme de Payoff et Profil P&L")
    
    # Range de prix
    payoff_range = np.linspace(max(1.0, K * 0.5), K * 1.5, 200)
    intrinsic, pnl = calculate_payoff_diagram(payoff_range, K, T, r, sigma, option_type, dividend_yield)
    
    fig_payoff = go.Figure()
    
    # Payoff à l'expiration
    fig_payoff.add_trace(go.Scatter(
        x=payoff_range, y=intrinsic,
        mode='lines', name='Payoff at Expiry',
        line=dict(color='#2E86C1', width=3, dash='dash')
    ))
    
    # P&L actuel (si position ouverte)
    fig_payoff.add_trace(go.Scatter(
        x=payoff_range, y=pnl,
        mode='lines', name='Current P&L Profile',
        line=dict(color='#E74C3C', width=3),
        fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.1)'
    ))
    
    # Strike et spot actuels
    fig_payoff.add_vline(x=K, line_dash="solid", line_color="orange", annotation_text="Strike (K)", annotation_position="top right")
    fig_payoff.add_vline(x=S, line_dash="dash", line_color="green", annotation_text="Current Spot (S)", annotation_position="top left")
    fig_payoff.add_hline(y=0, line_dash="solid", line_color="gray")
    
    fig_payoff.update_layout(
        title=f"{option_type.upper()} Option - Payoff Diagram",
        xaxis_title="Underlying Price at Expiry (S_T)",
        yaxis_title="Profit/Loss (€)",
        template="plotly_white",
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig_payoff, use_container_width=True)
    
    # Statistiques de profitabilité
    breakeven = K if option_type == "call" else K
    max_profit = "Unlimited" if option_type == "call" else f"{(K - price):.2f}"
    max_loss = f"{price:.2f}" if option_type == "call" else "Unlimited"
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💥 Break-Even", f"{breakeven:.2f}", help="Prix où P&L = 0")
    col2.metric("📈 Max Profit", max_profit, help="Profit maximal théorique")
    col3.metric("📉 Max Loss", max_loss, help="Perte maximale théorique")
    col4.metric("💰 Premium Paid", f"{price:.4f} €")

# ============================================================================
# TAB 3: MONTE CARLO SIMULATION
# ============================================================================
with tab3:
    st.subheader("🎲 Simulation Monte Carlo - Trajectoires & VaR")
    
    if mc_enabled:
        col_mc1, col_mc2 = st.columns(2)
        
        with col_mc1:
            st.metric("MC Price Estimate", f"{mc_result['price']:.4f} €")
            st.metric("BS Price (Baseline)", f"{price:.4f} €")
            st.metric("Pricing Difference", f"{abs(mc_result['price'] - price):.6f} €")
        
        with col_mc2:
            st.metric("95% Value-at-Risk", f"{mc_result['var_95']:.4f} €")
            st.metric("95% Conditional VaR", f"{mc_result['cvar_95']:.4f} €")
            st.metric("Simulation Std Error", f"±{mc_result['std_error']:.6f} €")
        
        st.markdown("---")
        
        # Distribution des payoffs
        col_dist1, col_dist2 = st.columns(2)
        
        with col_dist1:
            fig_payoff_dist = go.Figure()
            fig_payoff_dist.add_trace(go.Histogram(
                x=mc_result['discounted_payoffs'],
                nbinsx=100,
                name='Payoff Distribution',
                marker_color='#3498db',
                opacity=0.7
            ))
            fig_payoff_dist.add_vline(x=mc_result['price'], line_dash="dash", line_color="red", annotation_text="Mean")
            fig_payoff_dist.add_vline(x=mc_result['var_95'], line_dash="dash", line_color="orange", annotation_text="95% VaR")
            fig_payoff_dist.update_layout(
                title="Distribution des Payoffs Actualisés",
                xaxis_title="Payoff (€)",
                yaxis_title="Fréquence",
                template="plotly_white"
            )
            st.plotly_chart(fig_payoff_dist, use_container_width=True)
        
        # Distribution des prix terminaux
        with col_dist2:
            fig_terminal_dist = go.Figure()
            fig_terminal_dist.add_trace(go.Histogram(
                x=mc_result['terminal_prices'],
                nbinsx=100,
                name='Terminal S_T',
                marker_color='#2ecc71',
                opacity=0.7
            ))
            fig_terminal_dist.add_vline(x=S, line_dash="dash", line_color="red", annotation_text="Initial S")
            fig_terminal_dist.update_layout(
                title="Distribution des Prix Finaux (S_T)",
                xaxis_title="Underlying Price (€)",
                yaxis_title="Fréquence",
                template="plotly_white"
            )
            st.plotly_chart(fig_terminal_dist, use_container_width=True)
        
        st.markdown("---")
        
        # Trajectoires Monte Carlo
        st.subheader("📉 Trajectoires Brownienne")
        
        n_paths_display = st.slider("Nombre de trajectoires à afficher", 10, 500, 100)
        
        fig_paths = go.Figure()
        
        # Affichage des trajectoires
        for i in range(n_paths_display):
            fig_paths.add_trace(go.Scatter(
                y=mc_result['paths'][i, :],
                mode='lines',
                name=f'Path {i+1}',
                line=dict(width=0.5),
                opacity=0.3,
                hoverinfo='skip'
            ))
        
        # Moyenne des trajectoires
        mean_path = np.mean(mc_result['paths'], axis=0)
        fig_paths.add_trace(go.Scatter(
            y=mean_path,
            mode='lines',
            name='Mean Path',
            line=dict(color='red', width=3)
        ))
        
        # Interval de confiance
        percentile_95 = np.percentile(mc_result['paths'], 95, axis=0)
        percentile_5 = np.percentile(mc_result['paths'], 5, axis=0)
        
        fig_paths.add_trace(go.Scatter(
            y=percentile_95,
            mode='lines',
            name='95th Percentile',
            line=dict(color='green', width=1, dash='dash'),
            opacity=0.5
        ))
        
        fig_paths.add_trace(go.Scatter(
            y=percentile_5,
            mode='lines',
            name='5th Percentile',
            line=dict(color='red', width=1, dash='dash'),
            opacity=0.5,
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)'
        ))
        
        fig_paths.update_layout(
            title=f"Monte Carlo Trajectoires (T={T} ans, {int(252*T)} jours)",
            xaxis_title="Trading Days",
            yaxis_title="Underlying Price (S)",
            template="plotly_white",
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig_paths, use_container_width=True)
    else:
        st.info("Activez Monte Carlo dans la sidebar pour voir les résultats")

# ============================================================================
# TAB 4: GREEKS HEATMAPS
# ============================================================================
with tab4:
    st.subheader("🔥 Heatmaps - Sensibilité des Grecs")
    
    # Configuration de la heatmap
    col_hm1, col_hm2 = st.columns(2)
    
    with col_hm1:
        greek_selected = st.selectbox("Sélectionner un Grec", ["delta", "gamma", "vega", "theta", "rho"])
    
    with col_hm2:
        n_points = st.slider("Résolution (points)", 20, 100, 50)
    
    # Ranges
    spot_hm = np.linspace(max(1.0, S * 0.6), S * 1.4, n_points)
    sigma_hm = np.linspace(0.05, min(1.0, sigma * 3), n_points)
    
    # Calcul de la heatmap
    with st.spinner(f"🔄 Calcul de la heatmap {greek_selected}..."):
        heatmap_data = create_greeks_heatmap(spot_hm, sigma_hm, K, T, r, option_type, greek_selected, dividend_yield)
    
    # Affichage
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=np.round(spot_hm, 2),
        y=np.round(sigma_hm * 100, 1),
        colorscale='RdYlGn',
        colorbar=dict(title=greek_selected.upper())
    ))
    
    fig_heatmap.update_layout(
        title=f"{greek_selected.upper()} Heatmap - {option_type} option",
        xaxis_title="Spot Price (S)",
        yaxis_title="Volatility (σ %)",
        height=600
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Explications
    st.markdown("""
    **Interprétation de la Heatmap:**
    - 🔴 **Rouge**: Valeurs élevées du Grec
    - 🟡 **Jaune**: Valeurs modérées
    - 🟢 **Vert**: Valeurs faibles ou négatives
    - **Axe X**: Variation du prix du sous-jacent
    - **Axe Y**: Variation de la volatilité implicite
    """)

# ============================================================================
# TAB 5: PORTFOLIO STRATEGY BUILDER
# ============================================================================
with tab5:
    st.subheader("📊 Portfolio Strategy Builder")
    
    st.markdown("""
    Construisez des stratégies multi-leg et analysez le profil P&L combiné.
    """)
    
    # Sélection de stratégie prédéfinie
    strategy = st.selectbox("Stratégie Prédéfinie", [
        "Simple Long Call",
        "Long Call Spread",
        "Iron Condor",
        "Straddle",
        "Calendar Spread"
    ])
    
    # Legs du portefeuille
    legs = []
    
    if strategy == "Simple Long Call":
        legs = [
            {"type": "call", "K": K, "T": T, "position": "long", "multiplier": 1}
        ]
    
    elif strategy == "Long Call Spread":
        legs = [
            {"type": "call", "K": K, "T": T, "position": "long", "multiplier": 1},
            {"type": "call", "K": K * 1.1, "T": T, "position": "short", "multiplier": -1}
        ]
    
    elif strategy == "Iron Condor":
        legs = [
            {"type": "call", "K": K * 1.05, "T": T, "position": "short", "multiplier": -1},
            {"type": "call", "K": K * 1.10, "T": T, "position": "long", "multiplier": 1},
            {"type": "put", "K": K * 0.95, "T": T, "position": "long", "multiplier": 1},
            {"type": "put", "K": K * 0.90, "T": T, "position": "short", "multiplier": -1}
        ]
    
    elif strategy == "Straddle":
        legs = [
            {"type": "call", "K": K, "T": T, "position": "long", "multiplier": 1},
            {"type": "put", "K": K, "T": T, "position": "long", "multiplier": 1}
        ]
    
    elif strategy == "Calendar Spread":
        legs = [
            {"type": "call", "K": K, "T": T * 0.5, "position": "short", "multiplier": -1},
            {"type": "call", "K": K, "T": T, "position": "long", "multiplier": 1}
        ]
    
    # Calcul du P&L portefeuille
    portfolio_range = np.linspace(max(1.0, K * 0.5), K * 1.5, 200)
    portfolio_pnl = np.zeros_like(portfolio_range)
    
    leg_details = []
    
    for leg in legs:
        leg_K = leg["K"]
        leg_T = leg["T"]
        leg_type = leg["type"]
        
        if leg_type == "call":
            leg_intrinsic = np.maximum(portfolio_range - leg_K, 0)
        else:
            leg_intrinsic = np.maximum(leg_K - portfolio_range, 0)
        
        leg_price = price_option(S, leg_K, leg_T, r, sigma, leg_type, dividend_yield)
        leg_pnl = (leg_intrinsic - leg_price) * leg["multiplier"]
        
        portfolio_pnl += leg_pnl
        
        leg_details.append({
            "Type": f"{leg_type.upper()} ({leg['position']})",
            "Strike": f"{leg_K:.2f}",
            "Maturity": f"{leg_T:.2f}y",
            "Premium": f"{leg_price:.4f}",
            "Multiplier": leg["multiplier"]
        })
    
    # Affichage détails
    st.dataframe(pd.DataFrame(leg_details), use_container_width=True)
    
    # Graphique P&L
    fig_portfolio = go.Figure()
    
    fig_portfolio.add_trace(go.Scatter(
        x=portfolio_range, y=portfolio_pnl,
        mode='lines', name=strategy,
        line=dict(color='#2E86C1', width=3),
        fill='tozeroy', fillcolor='rgba(46, 134, 193, 0.2)'
    ))
    
    fig_portfolio.add_vline(x=K, line_dash="solid", line_color="orange")
    fig_portfolio.add_vline(x=S, line_dash="dash", line_color="green")
    fig_portfolio.add_hline(y=0, line_dash="solid", line_color="gray")
    
    fig_portfolio.update_layout(
        title=f"Portfolio P&L - {strategy}",
        xaxis_title="Spot Price at Expiry",
        yaxis_title="Portfolio P&L (€)",
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig_portfolio, use_container_width=True)
    
    # Statistiques portfolio
    st.markdown("---")
    
    portfolio_max_profit = np.max(portfolio_pnl)
    portfolio_max_loss = np.min(portfolio_pnl)
    portfolio_breakeven = portfolio_range[np.argmin(np.abs(portfolio_pnl))]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Max Profit", f"{portfolio_max_profit:.2f} €")
    col2.metric("Max Loss", f"{portfolio_max_loss:.2f} €")
    col3.metric("Break-Even", f"{portfolio_breakeven:.2f}")
    col4.metric("Profit/Loss Ratio", f"{abs(portfolio_max_profit / portfolio_max_loss) if portfolio_max_loss != 0 else 0:.2f}")

# ============================================================================
# 7. SECTION FINALE : IV SOLVER
# ============================================================================
st.markdown("---")
st.subheader("🔄 Solveur de Volatilité Implicite")

col_iv1, col_iv2 = st.columns([3, 1])

with col_iv1:
    market_price = st.number_input(
        "Prix observé sur le marché (€)",
        min_value=0.01,
        value=price,
        step=0.1,
        key="market_price_input"
    )

with col_iv2:
    solve_iv = st.button("📊 Calculer l'IV", type="primary", use_container_width=True)

if solve_iv:
    try:
        with st.spinner("⏳ Résolution en cours..."):
            raw_iv = calculate_implied_volatility_vectorized(market_price, S, K, T, r, option_type)
            iv = float(raw_iv[0]) if hasattr(raw_iv, "__len__") else float(raw_iv)
            
            if np.isnan(iv) or iv <= 0:
                st.error("❌ Impossible de converger. Le prix du marché saisi viole les lois d'arbitrage.")
            else:
                col_iv_result1, col_iv_result2, col_iv_result3 = st.columns(3)
                
                col_iv_result1.success(f"✅ IV calculée: **{iv * 100:.2f}%**")
                
                spread_iv = iv - sigma
                col_iv_result2.metric("Spread IV", f"{spread_iv * 100:+.2f}%", delta=f"{spread_iv * 100:+.2f}%")
                
                # Interprétation
                if abs(spread_iv) < 0.02:
                    interpretation = "✓ Marché conforme au modèle"
                elif spread_iv > 0.02:
                    interpretation = "📈 Marché OVER-valued (acheter théorique)"
                else:
                    interpretation = "📉 Marché UNDER-valued (vendre théorique)"
                
                col_iv_result3.info(interpretation)
    
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")

st.markdown("---")
st.caption("🔬 Institutional Options Desk v2.0 | Powered by Black-Scholes & Monte Carlo | Made with ❤️ by Pro Quant")