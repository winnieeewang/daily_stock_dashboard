import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os

st.set_page_config(layout="wide", page_title="每日股票看板")

st.title("📈 每日股票看板")
st.caption(f"数据更新于：{datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ---------- 配置 ----------
STOCKS = ["MU", "AAOI", "GOOGL", "MSFT", "AMZN", "MRVL", "LITE", "SNDK", "NVDA", "0700.HK", "9988.HK"]
MACRO_SYMBOLS = {
    "VIX": "^VIX",
    "美元指数": "DX-Y.NYB",
    "标普500": "^GSPC",
    "纳斯达克100": "^NDX",
    "黄金": "GC=F",
    "WTI原油": "CL=F",
    "10年期美债收益率": "^TNX",
}

# ---------- 数据加载（实时获取） ----------
@st.cache_data(ttl=3600)
def load_data():
    # 宏观数据
    macro_data = {}
    for name, sym in MACRO_SYMBOLS.items():
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="5d")
            if not hist.empty:
                macro_data[name] = hist['Close'].iloc[-1]
            else:
                macro_data[name] = None
        except:
            macro_data[name] = None
    
    # 个股数据
    stock_data = {}
    for sym in STOCKS:
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="3mo")
            if not hist.empty:
                # 计算技术指标
                close = hist['Close']
                high = hist['High']
                low = hist['Low']
                volume = hist['Volume']
                # 简单计算（完整指标可保留原技术面）
                stock_data[sym] = {
                    "close": close,
                    "high": high,
                    "low": low,
                    "volume": volume,
                    "ma20": close.rolling(20).mean(),
                    "ma50": close.rolling(50).mean(),
                }
            else:
                stock_data[sym] = None
        except:
            stock_data[sym] = None
    
    return macro_data, stock_data

macro_data, stock_data = load_data()

# ---------- 宏观仪表盘 ----------
st.header("🌍 宏观数据")
cols = st.columns(len(macro_data))
for i, (name, val) in enumerate(macro_data.items()):
    with cols[i]:
        st.metric(label=name, value=f"{val:.2f}" if val else "N/A")

# ---------- 个股K线图 ----------
st.header("📊 个股走势")
for sym in STOCKS:
    if stock_data[sym] is None:
        st.warning(f"无法获取 {sym} 数据")
        continue
    df = pd.DataFrame({
        "Open": stock_data[sym]["close"],  # 简化，实际需用OHLC
        "High": stock_data[sym]["high"],
        "Low": stock_data[sym]["low"],
        "Close": stock_data[sym]["close"],
        "Volume": stock_data[sym]["volume"]
    })
    df.index = pd.to_datetime(df.index)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name=sym), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=stock_data[sym]["ma20"], line=dict(color='orange', width=1), name="MA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=stock_data[sym]["ma50"], line=dict(color='green', width=1), name="MA50"), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="成交量"), row=2, col=1)
    fig.update_layout(height=600, title_text=sym)
    st.plotly_chart(fig, use_container_width=True)

# ---------- 报告区域（可选：调用DeepSeek生成文字报告） ----------
if st.button("生成今日分析报告（消耗API）"):
    with st.spinner("正在生成..."):
        # 此处可调用 DeepSeek API 生成报告，展示在下方
        st.info("报告生成功能预留，可集成你的 stock_dashboard.py 中的 prompt 逻辑")
