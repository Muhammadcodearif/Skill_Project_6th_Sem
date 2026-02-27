import streamlit as st
from streamlit_autorefresh import st_autorefresh
import numpy as np
import pandas as pd
import yfinance as yf
import ta
import torch
import torch.nn as nn
import plotly.graph_objects as go
import gymnasium as gym
from stable_baselines3 import PPO
import google.generativeai as genai
import os
from dotenv import load_dotenv
from datetime import datetime


# PAGE CONFIG


st.set_page_config(layout="wide")
st.title("🚀 AI Quant Trading Platform")


# LIVE SETTINGS


st.sidebar.header("⚡ Live Settings")

refresh_sec = st.sidebar.slider(
    "Refresh Interval (seconds)",
    min_value=5,
    max_value=60,
    value=15
)

st_autorefresh(interval=refresh_sec * 1000, key="live_refresh")

st.sidebar.success(f"Live Updates Running 🔴 ({refresh_sec}s)")

if st.sidebar.button("🔄 Refresh Now"):
    st.cache_data.clear()
    st.rerun()


# LOAD GEMINI API (AUTO MODEL)


load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

gen_model = None

if GEMINI_KEY:

    try:
        genai.configure(api_key=GEMINI_KEY)

      
        models = genai.list_models()

        model_name = None

        for m in models:
            if "generateContent" in m.supported_generation_methods:
                model_name = m.name
                break

        if model_name:
            gen_model = genai.GenerativeModel(model_name)
        else:
            st.warning("No compatible Gemini model found")

    except Exception as e:
        st.warning(f"Gemini Error: {e}")
        gen_model = None


# DATA FUNCTION 


@st.cache_data(ttl=10)
def get_data(ticker):

    df = yf.download(
        ticker,
        period="1d",
        interval="1m",
        auto_adjust=True,
        progress=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    for col in df.columns:
        df[col] = pd.Series(df[col]).astype(float).values.flatten()

    df.dropna(inplace=True)

    close = pd.Series(df["Close"].values.flatten(), index=df.index)
    volume = pd.Series(df["Volume"].values.flatten(), index=df.index)

    df["RSI"] = ta.momentum.RSIIndicator(close=close).rsi()
    df["MACD"] = ta.trend.MACD(close=close).macd()

    df["Returns"] = close.pct_change()
    df["Volatility"] = df["Returns"].rolling(20).std()

    df["Volume_Spike"] = volume / volume.rolling(20).mean()
    df["VWAP"] = (close * volume).cumsum() / volume.cumsum()
    df["VWAP_Dev"] = abs(close - df["VWAP"]) / df["VWAP"]

    df.dropna(inplace=True)

    return df


# FLASH CRASH DETECTION


def flash_crash(df):

    latest = df.iloc[-1]
    score = 0

    if abs(latest["Returns"]) > 0.02:
        score += 2

    if latest["Volatility"] > df["Volatility"].mean() * 2:
        score += 2

    if latest["Volume_Spike"] > 3:
        score += 2

    if latest["VWAP_Dev"] > 0.02:
        score += 2

    return min(score * 12, 100)


# TRANSFORMER MODEL


class TransformerModel(nn.Module):

    def __init__(self, input_dim=4):
        super().__init__()

        self.embed = nn.Linear(input_dim, 32)
        encoder = nn.TransformerEncoderLayer(32, 4)
        self.transformer = nn.TransformerEncoder(encoder, 2)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):

        x = self.embed(x)
        x = self.transformer(x)
        return self.fc(x[-1])


def train_transformer(df):

    model = TransformerModel()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    X = torch.tensor(
        df[["RSI", "MACD", "Returns", "Volatility"]].values,
        dtype=torch.float32
    )

    y = torch.tensor(df["Close"].values, dtype=torch.float32)

    for _ in range(30):
        optimizer.zero_grad()
        out = model(X.unsqueeze(1))
        loss = loss_fn(out.squeeze(), y[-1])
        loss.backward()
        optimizer.step()

    return model



# PREDICTION (FIXED)


def predict_price(model, df):

    X = torch.tensor(
        df[["RSI", "MACD", "Returns", "Volatility"]].values,
        dtype=torch.float32
    )

    pred = model(X.unsqueeze(1)).detach().numpy()

    pred_value = float(np.ravel(pred)[-1])

    return pred_value


# TRADING SIGNAL


def trading_signal(pred, price, rsi):

    if pred > price * 1.01 and rsi < 70:
        return "BUY"

    elif pred < price * 0.99 and rsi > 30:
        return "SELL"

    return "HOLD"


# BACKTEST


def backtest(df):

    returns = df["Returns"].dropna()
    equity = (1 + returns).cumprod()

    sharpe = np.sqrt(252) * returns.mean() / returns.std()
    max_dd = (equity / equity.cummax() - 1).min()

    return sharpe, max_dd, equity


# PORTFOLIO OPTIMIZER


def risk_parity(returns):

    cov = returns.cov()
    inv_vol = 1 / np.sqrt(np.diag(cov))
    weights = inv_vol / np.sum(inv_vol)

    return weights


# RL ENVIRONMENT


class TradingEnv(gym.Env):

    def __init__(self, df):

        super().__init__()

        self.df = df.reset_index()
        self.idx = 0
        self.position = 0

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,)
        )

    def reset(self, seed=None, options=None):

        self.idx = 0
        self.position = 0
        return self._obs(), {}

    def _obs(self):

        row = self.df.iloc[self.idx]

        return np.array([
            row["RSI"],
            row["MACD"],
            row["Returns"],
            row["Volatility"]
        ])

    def step(self, action):

        prev_position = self.position
        self.position = action - 1

        reward = self.df.iloc[self.idx]["Returns"] * self.position
        reward -= 0.001 * abs(self.position - prev_position)

        self.idx += 1
        done = self.idx >= len(self.df) - 1

        return self._obs(), reward, done, False, {}


def train_rl(df):

    env = TradingEnv(df)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=2000)

    return model


# SCANNER


def scanner(stocks):

    results = []

    for s in stocks:

        data = get_data(s)

        if data.empty:
            continue

        crash_prob = flash_crash(data)
        latest = data.iloc[-1]

        results.append({
            "Stock": s,
            "Price": latest["Close"],
            "RSI": latest["RSI"],
            "Crash Risk %": crash_prob
        })

    return pd.DataFrame(results)


# UI


stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

selected = st.selectbox("Select Stock", stocks)

df = get_data(selected)

if df.empty:
    st.warning("No data available")
    st.stop()

latest_price = df["Close"].iloc[-1]

model_tf = train_transformer(df)
pred_price = predict_price(model_tf, df)

crash_prob = flash_crash(df)
signal = trading_signal(pred_price, latest_price, df["RSI"].iloc[-1])

sharpe, max_dd, equity = backtest(df)

# Metrics
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Live Price", round(latest_price, 2))
col2.metric("Predicted Price", round(pred_price, 2))
col3.metric("Crash Risk %", crash_prob)
col4.metric("Signal", signal)
col5.metric("Sharpe", round(sharpe, 2))

st.caption(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")

# Price chart
st.subheader("📈 Price & Prediction")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price"))
fig.add_trace(go.Scatter(
    x=[df.index[-1]],
    y=[pred_price],
    mode="markers",
    marker=dict(size=12),
    name="Prediction"
))
st.plotly_chart(fig, use_container_width=True)

# Equity curve
st.subheader("📊 Equity Curve")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df.index[:len(equity)], y=equity))
st.plotly_chart(fig2, use_container_width=True)

# Scanner
st.subheader("🔍 Multi Stock Scanner")
st.dataframe(scanner(stocks))

# Portfolio
st.subheader("📊 Portfolio Optimizer")

returns = pd.DataFrame()

for s in stocks:
    data = get_data(s)
    if not data.empty:
        returns[s] = data["Returns"]

weights = risk_parity(returns.dropna())

st.dataframe(pd.DataFrame({
    "Stock": returns.columns,
    "Weight": weights
}))

# RL
st.subheader("🤖 RL Agent")

if st.button("Train RL Agent"):
    train_rl(df)
    st.success("RL Agent Trained")


# GENAI PANEL

st.subheader("🧠 AI Decision System")

if gen_model:

    if "ai_decision" not in st.session_state:
        st.session_state.ai_decision = None

    if st.button("🧠 Generate AI Decision"):

        with st.spinner("AI is analyzing market..."):

            prompt = f"""
            Institutional Trading AI:

            Price: {latest_price}
            Predicted: {pred_price}
            Crash Risk: {crash_prob}
            Signal: {signal}

            Provide market insight and recommendation.
            """

            try:
                response = gen_model.generate_content(prompt)
                decision = response.text if hasattr(response, "text") else str(response)

                st.session_state.ai_decision = decision

            except Exception as e:
                st.warning(f"GenAI Error: {e}")

    
    if st.session_state.ai_decision:
        st.success("AI Decision Generated ✅")
        st.write(st.session_state.ai_decision)

else:
    st.info("Add GEMINI_API_KEY")
