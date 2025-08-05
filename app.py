import plotly.express as px
import ta
import os
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model, Sequential

import plotly.graph_objs as go


# ✅ Pretrained model paths
PRETRAINED_MODELS = {
    "RELIANCE.NS": "models/reliance_gru.h5",
    "TCS.NS": "models/tcs_gru.h5",
    "ADANIPORTS.NS": "models/adaniports_gru.h5",
    "INFY.NS": "models/infy_gru.h5",
    "ITC.NS": "models/itc_gru.h5"
}

# ✅ Utilities
@st.cache_data

# this code for bringing data from yfinance

# def load_stock_data(ticker):
#
#     df = yf.download(ticker, start="2015-01-01", end="2025-06-30")
#     # filepath = os.path.join("data", f"{ticker}.csv")
#
#     # ✅ Load CSV
#     # df = pd.read_csv(filepath, skiprows=2, parse_dates=["Date"])
#     # df = pd.read_csv(filepath,header=[0, 1])
#     # st.write(df)
#     # ✅ FLATTEN MULTIINDEX COLUMNS IF PRESENT
#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = ['_'.join(col).strip() for col in df.columns.values]
#
#     # ✅ Rename common OHLCV if needed (e.g., Close_ADANIPORTS.NS → Close)
#     for col in df.columns:
#         if col.startswith("Open_"): df.rename(columns={col: "Open"}, inplace=True)
#         if col.startswith("High_"): df.rename(columns={col: "High"}, inplace=True)
#         if col.startswith("Low_"): df.rename(columns={col: "Low"}, inplace=True)
#         if col.startswith("Close_"): df.rename(columns={col: "Close"}, inplace=True)
#         if col.startswith("Volume_"): df.rename(columns={col: "Volume"}, inplace=True)
#
#     df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
#
#     close_series = df['Close']
#
#     # ✅ Add Technical Indicators
#     df['RSI'] = ta.momentum.RSIIndicator(close=close_series, window=14).rsi()
#     df['SMA'] = ta.trend.SMAIndicator(close=close_series, window=20).sma_indicator()
#     df['EMA'] = ta.trend.EMAIndicator(close=close_series, window=20).ema_indicator()
#     macd = ta.trend.MACD(close=close_series)
#     df['MACD'] = macd.macd()
#     df['MACD_signal'] = macd.macd_signal()
#     df['MACD_diff'] = macd.macd_diff()
#
#     df.dropna(inplace=True)
#     return df

def load_stock_data(ticker):
    # 📁 Load from local CSV
    filepath = os.path.join("data", f"{ticker}.csv")

    df = pd.read_csv(filepath, header=None)

    # Extract column names from the first row
    column_names = df.iloc[0].tolist()

    # Extract actual data starting from row 3 (i.e., index 3)
    df = df.iloc[3:].copy()

    # Assign the correct column names
    df.columns = column_names

    # Rename 'Price' to 'Date' and convert to datetime
    df.rename(columns={"Price": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.set_index("Date", inplace=True)

    # Convert all other columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # st.write(df)

    # 📈 Technical Indicators
    close_series = df["Close"]
    df["RSI"] = ta.momentum.RSIIndicator(close=close_series, window=14).rsi()
    df["SMA"] = ta.trend.SMAIndicator(close=close_series, window=20).sma_indicator()
    df["EMA"] = ta.trend.EMAIndicator(close=close_series, window=20).ema_indicator()

    macd = ta.trend.MACD(close=close_series)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_diff"] = macd.macd_diff()

    # 🧹 Drop rows with NaNs from indicators
    df.dropna(inplace=True)

    return df

def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step])
        y.append(data[i + time_step][0])  # Target is 'Close' (first feature)
    return np.array(X), np.array(y)

def forecast_next_10_days(model, last_seq, scaler):

    preds = []
    input_seq = last_seq.copy()  # shape: (time_steps, features)

    for _ in range(10):
        # Predict next 'Close' value using model
        pred_close = model.predict(input_seq.reshape(1, input_seq.shape[0], input_seq.shape[1]), verbose=0)[0][0]

        # Copy the last feature vector and update only the 'Close' value (index 0)
        last_features = input_seq[-1].copy()
        last_features[0] = pred_close

        # Append new features to the sequence
        input_seq = np.concatenate((input_seq[1:], last_features.reshape(1, -1)), axis=0)
        preds.append(last_features)  # Save full feature vector for inverse_transform

    preds = np.array(preds)  # shape: (10, features)
    return scaler.inverse_transform(preds)[:, 0]



def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"Model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2 Score": round(r2, 4), "MAPE (%)": round(mape, 2)}


# ✅ Streamlit UI
st.title("📈 Stock Price Prediction with GRU + Stored Model Comparison")

stock_options = list(PRETRAINED_MODELS.keys())
selected = st.selectbox("Choose a stock:", stock_options)

run = st.button("Predict")

if run:
    ticker =  selected


    df = load_stock_data(ticker)


    st.subheader("📜 Historical Closing Price - Last 15 Years")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
    fig_hist.update_layout(title=f"Historical Close Price for {ticker} (2010–2025)", xaxis_title='Date',
                           yaxis_title='Price (INR)')
    st.plotly_chart(fig_hist)

    with st.expander("📊 View Technical Indicators"):
        st.subheader("Close Price with SMA & EMA")

        st.line_chart(df[['Close', 'SMA', 'EMA']])

        st.subheader("RSI Indicator")
        st.line_chart(df[['RSI']])


        if 'MACD' in df.columns and 'MACD_signal' in df.columns:
            st.subheader("MACD and MACD Signal")
            st.line_chart(df[['MACD', 'MACD_signal']])
        else:
            st.warning("⚠️ MACD columns not found in dataframe.")

    feature_cols = ['Close', 'RSI', 'SMA', 'EMA', 'MACD', 'MACD_signal', 'MACD_diff']
    scaled_df = df[feature_cols]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(scaled_df)


    X, y = create_sequences(scaled)

    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])  # Shape = (samples, timesteps, features)

    train_size = int(len(X)*0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    if ticker in PRETRAINED_MODELS:
        model = load_model(PRETRAINED_MODELS[ticker])
        st.write("Model input shape:", model.input_shape)  # e.g., (None, 60, 1)

        st.write("Test data shape:", X.shape) 
        y_pred = model.predict(X_test)

        dummy_pad = np.zeros((len(y_test), scaled.shape[1] - 1))
        y_test_rescaled = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), dummy_pad)))[:, 0]
        y_pred_rescaled = scaler.inverse_transform(np.hstack((y_pred, dummy_pad)))[:, 0]


        results_df = pd.read_csv("models/all_model_results.csv")


        # Forecast plot
        test_dates = df.index[-len(y_test_rescaled):]
        future_preds = forecast_next_10_days(model, X[-1], scaler)
        future_dates = [df.index[-1] + timedelta(days=i + 1) for i in range(10)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_dates, y=y_test_rescaled.flatten(), name='Actual'))
        fig.add_trace(go.Scatter(x=test_dates, y=y_pred_rescaled.flatten(), name='Predicted'))
        fig.add_trace(
            go.Scatter(x=future_dates, y=future_preds.flatten(), name='10-Day Forecast', line=dict(dash='dot')))
        fig.update_layout(title=f"Prediction for {ticker}", xaxis_title='Date', yaxis_title='Price (INR)')
        st.plotly_chart(fig)

        st.subheader("📊 Model Comparison for Selected Stock")
        comparison_df = results_df[results_df["ticker"] == ticker]

        st.dataframe(comparison_df.sort_values(by="RMSE"))

        fig_bar = px.bar(comparison_df.sort_values(by="RMSE"), x="Model", y="RMSE", color="Model",
                             title="📉 RMSE Comparison Across Models")
        st.plotly_chart(fig_bar)
