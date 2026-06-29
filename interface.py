import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm

# Importation de vos modules mathématiques locaux
from src.black_scholes import price_european_option_vectorized
from src.implied_volatility import calculate_implied_volatility_vectorized

# --- 1. CONFIGURATION DU DASHBOARD ---
st.set_page_config(page_title="Pro Quant Engine", page_icon="🌌", layout="wide")

# Injection de CSS personnalisé pour les badges
st.markdown("""
    <style>
    .moneyness-badge {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .itm { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .otm { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .atm { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    </style>
""", unsafe_allow_html=True)

st.title("🌌 Institutional Options Desk")
st.markdown("Moteur de pricing Black-Scholes temps réel, analyse des Grecs et profils de risque de marché.")
st.markdown("---")

# --- 2. BARRE LATÉRALE : PARAMÈTRES AVEC INFOBULLES ---
st.sidebar.header("⚙️ Market Parameters")
S = st.sidebar.number_input("Underlying Asset Price (S)", value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (T in Years)", min_value=0.001, max_value=5.0, value=0.5, step=0.05)
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)
sigma = st.sidebar.slider("Volatility (σ)", min_value=0.01, max_value=1.50, value=0.20, step=0.01)
option_type = st.sidebar.selectbox("Option Type", ["call", "put"])

# --- 3. FONCTIONS MATHÉMATIQUES DES GRECS ---
def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    T = max(0.0001, T)
    sigma = max(0.0001, sigma)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    
    if option_type == "call":
        delta = cdf_d1
        theta = (-S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * cdf_d2
    else:
        delta = cdf_d1 - 1.0
        theta = (-S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * pdf_d1 / 100.0
    theta_per_day = theta / 365.0
    
    return delta, gamma, vega, theta_per_day

# --- 4. CALCULS GLOBAUX ---
# Récupération du prix
raw_price = price_european_option_vectorized(S, K, T, r, sigma, option_type)
price = float(raw_price[0]) if hasattr(raw_price, "__len__") else float(raw_price)

# Récupération des Grecs
delta, gamma, vega, theta = calculate_greeks(S, K, T, r, sigma, option_type)

# Logique de Moneyness (ITM, ATM, OTM)
threshold = K * 0.01
if abs(S - K) <= threshold:
    moneyness, css_class = "At-The-Money (ATM)", "atm"
elif (option_type == "call" and S > K) or (option_type == "put" and S < K):
    moneyness, css_class = "In-The-Money (ITM)", "itm"
else:
    moneyness, css_class = "Out-Of-The-Money (OTM)", "otm"

# --- 5. AFFICHAGE DES KPIS ---
# Badge de Moneyness
st.markdown(f'<div class="moneyness-badge {css_class}">État Actuel : {moneyness}</div>', unsafe_allow_html=True)

# Ligne des indicateurs
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Premium (Prix)", f"{price:.4f} €")
col2.metric("Delta (Δ)", f"{delta:.4f}")
col3.metric("Gamma (Γ)", f"{gamma:.4f}")
col4.metric("Vega (ν)", f"{vega:.4f}")
col5.metric("Theta (Θ)", f"{theta:.4f}")

st.markdown("---")

# --- 6. ONGLETS DE VISUALISATION AVANCÉE ---
tab1, tab2 = st.tabs(["📈 Profils de Risque (Les Grecs)", "🔄 Extracteur de Volatilité Implicite"])

with tab1:
    st.subheader("Analyse de Sensibilité selon le Prix du Sous-jacent")
    
    # Création du jeu de données pour les graphiques (-50% à +50% du prix actuel)
    spot_range = np.linspace(max(1.0, S * 0.5), S * 1.5, 100)
    
    prices, deltas, gammas, vegas, thetas = [], [], [], [], []
    for s in spot_range:
        p = price_european_option_vectorized(s, K, T, r, sigma, option_type)
        d, g, v, t = calculate_greeks(s, K, T, r, sigma, option_type)
        
        prices.append(float(p[0]) if hasattr(p, "__len__") else float(p))
        deltas.append(d)
        gammas.append(g)
        vegas.append(v)
        thetas.append(t)

    # Conversion en DataFrame
    df_chart = pd.DataFrame({
        "Spot Price": spot_range,
        "Premium": prices,
        "Delta": deltas,
        "Gamma": gammas,
        "Vega": vegas,
        "Theta": thetas
    }).set_index("Spot Price")

    # Affichage en grille 2x2
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("**1. Valeur de la Prime & Delta**")
        st.line_chart(df_chart[["Premium", "Delta"]], color=["#2E86C1", "#E74C3C"])
        
        st.markdown("**3. Sensibilité à la Volatilité (Vega)**")
        st.line_chart(df_chart[["Vega"]], color=["#F39C12"])

    with chart_col2:
        st.markdown("**2. Accélération du Risque (Gamma)**")
        st.line_chart(df_chart[["Gamma"]], color=["#8E44AD"])
        
        st.markdown("**4. Érosion du Temps (Theta)**")
        st.line_chart(df_chart[["Theta"]], color=["#16A085"])

with tab2:
    st.subheader("Solveur Inverse : Volatilité Implicite")
    st.write("Le marché observe parfois des prix différents de la théorie. Entrez le prix réel pour déduire la volatilité implicite (IV) pricée par les traders.")
    
    market_price = st.number_input("Prix observé sur le marché (€)", min_value=0.01, value=price, step=0.1)
    
    if st.button("Calculer l'IV du Marché", type="primary"):
        try:
            raw_iv = calculate_implied_volatility_vectorized(market_price, S, K, T, r, option_type)
            iv = float(raw_iv[0]) if hasattr(raw_iv, "__len__") else float(raw_iv)
            
            if np.isnan(iv) or iv <= 0:
                st.error("Impossible de converger. Le prix du marché saisi viole les lois d'arbitrage.")
            else:
                st.success(f"La volatilité implicite du marché est de **{iv * 100:.2f} %**")
                
                # Comparaison visuelle
                delta_iv = iv - sigma
                st.metric(label="Spread de Volatilité (Market IV vs Model σ)", 
                          value=f"{iv * 100:.2f} %", 
                          delta=f"{delta_iv * 100:.2f} %")
        except Exception as e:
            st.error(f"Erreur de résolution : {e}")