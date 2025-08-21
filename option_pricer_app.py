
# option_pricer_app.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import streamlit as st

st.set_page_config(page_title="Option Pricing Playground",
                   page_icon="💹",
                   layout="wide")

# =========================
# Utility Functions
# =========================

def to_years(days: float) -> float:
    return max(days, 0.0) / 365.0

def clamp_sigma(s: float) -> float:
    # avoid divide-by-zero or negative sigma
    return max(1e-6, float(s))

# --- Black-Scholes (no dividends) ---
def black_scholes(S, K, r, T, sigma, option_type="call"):
    sigma = clamp_sigma(sigma)
    if T <= 0:
        return max(0.0, (S - K) if option_type == "call" else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# --- Binomial Tree (Cox-Ross-Rubinstein) ---
def binomial_option(S, K, r, T, sigma, steps=200, option_type="call"):
    steps = max(1, int(steps))
    if T <= 0:
        return max(0.0, (S - K) if option_type == "call" else (K - S))
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp(r * dt) - d) / (u - d)
    p = min(max(p, 0.0), 1.0)  # guard against numerical issues

    # terminal prices
    j = np.arange(steps + 1)
    prices = S * (u ** j) * (d ** (steps - j))

    if option_type == "call":
        values = np.maximum(prices - K, 0.0)
    else:
        values = np.maximum(K - prices, 0.0)

    # backward induction
    for _ in range(steps, 0, -1):
        values = disc * (p * values[1:] + (1 - p) * values[:-1])

    return float(values[0])

# --- Monte Carlo (European) ---
def monte_carlo_option(S, K, r, T, sigma, simulations=100_000, seed=None, option_type="call"):
    if T <= 0:
        return max(0.0, (S - K) if option_type == "call" else (K - S))
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(simulations)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)
    price = np.exp(-r * T) * payoff.mean()
    stderr = np.exp(-r * T) * payoff.std(ddof=1) / np.sqrt(simulations)
    return float(price), float(stderr)

# Vectorized BS for grids
def bs_grid(S, r, option_type, strikes, volatilities, T_years):
    # strikes: (nK,), volatilities: (nV,)
    K = strikes[None, :]          # shape (1, nK)
    sig = volatilities[:, None]   # shape (nV, 1)
    T = T_years
    sig = np.maximum(sig, 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T)) if T > 0 else np.full_like(K, np.inf)
    d2 = d1 - sig * np.sqrt(T) if T > 0 else np.full_like(K, np.inf)
    if option_type == "call":
        prices = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        prices = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return prices

@st.cache_data(show_spinner=False)
def compute_heatmap(method, S, r, T_years, option_type, k_min, k_max, k_points,
                    v_min, v_max, v_points, steps, sims, seed):
    strikes = np.linspace(k_min, k_max, k_points)
    volatilities = np.linspace(v_min, v_max, v_points)
    heat = np.zeros((v_points, k_points))

    if method == "Black-Scholes":
        heat = bs_grid(S, r, option_type, strikes, volatilities, T_years)
    elif method == "Binomial":
        for i, sigma in enumerate(volatilities):
            for j, K in enumerate(strikes):
                heat[i, j] = binomial_option(S, K, r, T_years, sigma, steps=steps, option_type=option_type)
    else:  # Monte Carlo
        for i, sigma in enumerate(volatilities):
            for j, K in enumerate(strikes):
                price, _ = monte_carlo_option(S, K, r, T_years, sigma, simulations=sims, seed=seed, option_type=option_type)
                heat[i, j] = price

    df = pd.DataFrame(heat, index=np.round(volatilities, 4), columns=np.round(strikes, 4))
    df.index.name = "Volatility"
    df.columns.name = "Strike"
    return df

# =========================
# Sidebar Controls
# =========================
with st.sidebar:
    st.title("⚙️ Inputs")
    option_type = st.radio("Option Type", ["call", "put"], horizontal=True)
    S = st.number_input("Spot Price (S₀)", min_value=0.0, value=100.0, step=1.0, format="%.4f")
    K = st.number_input("Strike (K)", min_value=0.0, value=100.0, step=1.0, format="%.4f")
    r_pct = st.number_input("Risk-free rate r (annual, %)", min_value=-100.0, max_value=100.0, value=5.0, step=0.10, format="%.2f")
    r = r_pct / 100.0
    T_days = st.number_input("Time to expiry (days)", min_value=0.0, value=30.0, step=1.0, format="%.2f")
    T_years = to_years(T_days)
    sigma_pct = st.slider("Implied Volatility σ (%, annualized)", min_value=5.0, max_value=35.0, value=25.0, step=0.1)
    sigma = sigma_pct / 100.0

    st.markdown("---")
    st.subheader("Model Settings")
    steps = st.slider("Binomial steps", min_value=10, max_value=2000, value=300, step=10)
    sims = st.slider("Monte Carlo simulations (for single price output)", min_value=1_000, max_value=500_000, value=100_000, step=1_000)
    seed = st.number_input("Random seed (MC)", value=42, step=1)

    st.markdown("---")
    st.subheader("Heatmap Grid")
    k_range_pct = st.slider("Strike range (as % of S₀)", 50, 200, (70, 130), step=5)
    k_points = st.slider("Strike points", 10, 100, 41, step=1)
    v_range_pct = st.slider("Volatility range (%, annualized)", 5, 300, (5, 150), step=5)
    v_points = st.slider("Volatility points", 10, 100, 41, step=1)

    k_min = (k_range_pct[0] / 100.0) * S
    k_max = (k_range_pct[1] / 100.0) * S
    v_min = v_range_pct[0] / 100.0
    v_max = v_range_pct[1] / 100.0

# =========================
# Main Layout
# =========================

st.title("💹 Option Pricing Playground")
st.caption("European options with no dividends. Methods: Black–Scholes (closed form), Binomial (CRR), and Monte Carlo.")

col1, col2, col3 = st.columns(3, gap="large")

# --- Single-Point Prices ---
with col1:
    st.subheader("Black–Scholes")
    bs_price = black_scholes(S, K, r, T_years, sigma, option_type)
    st.metric("Price", f"{bs_price:,.4f}")
with col2:
    st.subheader("Binomial (CRR)")
    bin_price = binomial_option(S, K, r, T_years, sigma, steps=steps, option_type=option_type)
    st.metric("Price", f"{bin_price:,.4f}")
with col3:
    st.subheader("Monte Carlo")
    mc_price, mc_se = monte_carlo_option(S, K, r, T_years, sigma, simulations=sims, seed=seed, option_type=option_type)
    st.metric("Price", f"{mc_price:,.4f}")
    st.caption(f"Std. error ≈ {mc_se:,.6f}")

# =========================
# Heatmap
# =========================
st.markdown("---")
st.subheader("Heatmap")

hm_method = st.radio("Heatmap method", ["Black-Scholes", "Binomial", "Monte Carlo"], horizontal=True)
if hm_method == "Monte Carlo":
    st.info("For the heatmap, Monte Carlo uses fewer simulations per grid point for performance. Adjust below if needed.")

hm_sims = st.slider("Monte Carlo simulations per grid point (heatmap)", min_value=500, max_value=50_000, value=5_000, step=500) if hm_method == "Monte Carlo" \
    else None

df_heat = compute_heatmap(hm_method, S, r, T_years, option_type,
                          k_min, k_max, k_points,
                          v_min, v_max, v_points,
                          steps, hm_sims if hm_sims else 0, seed)

# Display heatmap using matplotlib
fig, ax = plt.subplots()
im = ax.imshow(df_heat.values, origin="lower", aspect="auto",
               extent=[df_heat.columns.min(), df_heat.columns.max(),
                       df_heat.index.min(), df_heat.index.max()],
               cmap="RdYlGn")
ax.set_xlabel("Strike (K)")
ax.set_ylabel("Volatility (σ)")
ax.set_title(f"{hm_method} {option_type.capitalize()} Price Heatmap\n(T = {T_days:.0f} days, S₀ = {S:.2f}, r = {r_pct:.2f}%)")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Option Price")
st.pyplot(fig, clear_figure=True, use_container_width=True)

# =========================
# Data Table & Download
# =========================
st.markdown("---")
st.subheader("Heatmap Data")
st.dataframe(df_heat, use_container_width=True)

csv = df_heat.to_csv().encode("utf-8")
st.download_button("Download heatmap CSV", data=csv, file_name="heatmap.csv", mime="text/csv")

# =========================
# Footer
# =========================
st.markdown("""
---
**Notes**
- Assumes European exercise, no dividends.
- Binomial uses Cox–Ross–Rubinstein model.
- Monte Carlo uses geometric Brownian motion and reports a standard error for the single-point estimate.
- Heatmap axes: Strike (x-axis) vs Volatility (y-axis). You can adjust grid ranges and density in the sidebar.
""")
