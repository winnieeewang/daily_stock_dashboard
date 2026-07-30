import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import yfinance as yf

# ---------- 页面配置 ----------
st.set_page_config(
    layout="wide",
    page_title="每日股票看板",
    page_icon="📈",
    initial_sidebar_state="collapsed"
)

# ---------- 自定义 CSS（深色金融风） ----------
st.markdown("""
<style>
    /* 全局背景 */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #141b2d 50%, #1a2335 100%);
    }
    /* 卡片样式 */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(0, 200, 255, 0.3);
        box-shadow: 0 8px 40px rgba(0, 200, 255, 0.08);
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff, #7b61ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-label {
        font-size: 14px;
        color: rgba(255,255,255,0.6);
        letter-spacing: 0.5px;
        font-weight: 400;
    }
    .section-title {
        font-size: 22px;
        font-weight: 600;
        color: #ffffff;
        border-left: 3px solid #00d4ff;
        padding-left: 16px;
        margin: 30px 0 20px 0;
        letter-spacing: 0.5px;
    }
    /* K线图容器 */
    .chart-container {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 16px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        margin-bottom: 20px;
    }
    /* 底部信息 */
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.2);
        font-size: 12px;
        padding: 40px 0 20px 0;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 40px;
    }
    /* 标签 */
    .badge {
        display: inline-block;
        padding: 2px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.3px;
    }
    .badge-bull { background: rgba(0, 230, 118, 0.15); color: #00e676; }
    .badge-bear { background: rgba(255, 82, 82, 0.15); color: #ff5252; }
    .badge-neutral { background: rgba(255, 193, 7, 0.15); color: #ffd54f; }
    /* 滚动条 */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); border-radius: 3px; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.25); }
</style>
""", unsafe_allow_html=True)

# ---------- 标题 ----------
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
    <div>
        <span style="font-size: 32px; font-weight: 700; background: linear-gradient(135deg, #00d4ff, #7b61ff, #ff6b9d); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">📈 每日股票看板</span>
        <span style="font-size: 14px; color: rgba(255,255,255,0.3); margin-left: 16px; font-weight: 300;">QUANT DASHBOARD</span>
    </div>
    <div style="font-size: 13px; color: rgba(255,255,255,0.25); background: rgba(255,255,255,0.05); padding: 6px 16px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.06);">
        📅 {update_time}
    </div>
</div>
<div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.2), transparent); margin: 12px 0 24px 0;"></div>
""".format(update_time=datetime.now().strftime('%Y-%m-%d %H:%M')), unsafe_allow_html=True)

# ---------- 配置 ----------
STOCKS = ["MU", "AAOI", "GOOGL", "MSFT", "ORCL", "TSLA", "AMZN", "SPCX", "SKHY", "MRVL", "LITE", "SNDK", "NVDA", "0700.HK", "0883.HK", "3750.HK"]
MACRO_SYMBOLS = {
    "VIX": "^VIX",
    "美元指数": "DX-Y.NYB",
    "标普500": "^GSPC",
    "纳斯达克100": "^NDX",
    "黄金": "GC=F",
    "WTI原油": "CL=F",
    "10Y美债": "^TNX",
}

# ---------- 加载数据 ----------
def load_csv_data():
    macro_df = None
    stocks_df = None
    sox_df = None
    if os.path.exists("data/macro.csv"):
        macro_df = pd.read_csv("data/macro.csv")
    if os.path.exists("data/stocks.csv"):
        stocks_df = pd.read_csv("data/stocks.csv")
    if os.path.exists("data/sox.csv"):
        sox_df = pd.read_csv("data/sox.csv")
    return macro_df, stocks_df, sox_df

macro_df, stocks_df, sox_df = load_csv_data()

# 如果 CSV 不存在，实时获取
if macro_df is None:
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
    macro_df = pd.DataFrame([macro_data])

# ---------- 宏观仪表盘 ----------
st.markdown('<div class="section-title">🌍 宏观数据</div>', unsafe_allow_html=True)

if macro_df is not None and not macro_df.empty:
    cols = st.columns(len(macro_df.columns))
    for i, col_name in enumerate(macro_df.columns):
        val = macro_df.iloc[0][col_name]
        if pd.notna(val):
            try:
                val_float = float(val)
                if col_name == "VIX":
                    color = "🟢" if val_float < 18 else "🟡" if val_float < 25 else "🔴"
                elif col_name in ["黄金", "WTI原油", "美元指数"]:
                    color = "🟡"
                else:
                    color = "🔵"
                display_val = f"{val_float:.2f}"
            except:
                display_val = str(val)
                color = "⚪"
        else:
            display_val = "N/A"
            color = "⚪"
        
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{color} {col_name}</div>
                <div class="metric-value">{display_val}</div>
            </div>
            """, unsafe_allow_html=True)

# ---------- SOX 信号 ----------
st.markdown('<div class="section-title">📉 SOX 半导体指数</div>', unsafe_allow_html=True)

if sox_df is not None and not sox_df.empty:
    sox_row = sox_df.iloc[0]
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📊 最新价</div>
            <div class="metric-value">{sox_row.get('最新价', 'N/A')}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📉 回撤</div>
            <div class="metric-value" style="color: {'#ff5252' if float(str(sox_row.get('回撤%', '0')).replace('%','')) < -20 else '#ffd54f'};">{sox_row.get('回撤%', 'N/A')}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📈 RSI(14)</div>
            <div class="metric-value">{sox_row.get('RSI', 'N/A')}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        status = "🐻 熊市" if sox_row.get('技术性熊市') else "🐂 非熊市"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📌 状态</div>
            <div class="metric-value" style="font-size: 18px; color: {'#ff5252' if sox_row.get('技术性熊市') else '#00e676'};">{status}</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📌 MA20</div>
            <div class="metric-value" style="font-size: 20px;">{sox_row.get('MA20', 'N/A')}</div>
        </div>
        """, unsafe_allow_html=True)

    if '信号列表' in sox_row and sox_row['信号列表']:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.03); border-radius: 12px; padding: 12px 20px; margin-top: 12px; border-left: 2px solid rgba(0,212,255,0.3);">
            <span style="color: rgba(255,255,255,0.5); font-size: 12px;">⚡ 关键信号</span>
            <span style="color: rgba(255,255,255,0.7); font-size: 13px; margin-left: 12px;">{sox_row['信号列表']}</span>
        </div>
        """, unsafe_allow_html=True)
else:
    st.warning("暂无 SOX 数据")

# ---------- 个股 K 线图 ----------
st.markdown('<div class="section-title">📊 个股走势</div>', unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_stock_hist(symbol):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="3mo")
        if not hist.empty:
            return hist
        return None
    except:
        return None

# 从 CSV 读取今日涨跌幅
stock_changes = {}
if stocks_df is not None and not stocks_df.empty:
    for _, row in stocks_df.iterrows():
        sym = row.get('symbol', '')
        if sym:
            stock_changes[sym] = row.get('涨跌幅%', 0)

# 分组显示（2列网格）
for i in range(0, len(STOCKS), 2):
    cols = st.columns(2)
    for j, sym in enumerate(STOCKS[i:i+2]):
        with cols[j]:
            hist = get_stock_hist(sym)
            if hist is None:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.03); border-radius: 16px; padding: 30px; text-align: center; border: 1px solid rgba(255,255,255,0.06);">
                    <span style="color: rgba(255,255,255,0.3);">⚠️ 无法获取 {sym} 数据</span>
                </div>
                """, unsafe_allow_html=True)
                continue
            
            # 计算均线
            hist['MA20'] = hist['Close'].rolling(20).mean()
            hist['MA50'] = hist['Close'].rolling(50).mean()
            
            # 涨跌幅
            change = stock_changes.get(sym, 0)
            change_color = "#00e676" if change > 0 else "#ff5252" if change < 0 else "#ffd54f"
            change_symbol = "▲" if change > 0 else "▼" if change < 0 else "●"
            
            # 标题带涨跌幅
            display_name = sym.replace(".HK", "")
            title_html = f"""
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #ffffff; font-weight: 600; font-size: 16px;">{display_name}</span>
                <span style="color: {change_color}; font-size: 14px; font-weight: 500;">{change_symbol} {change:.2f}%</span>
            </div>
            """
            st.markdown(title_html, unsafe_allow_html=True)
            
            # 绘制K线图（简化版，更清晰）
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                row_heights=[0.7, 0.3])
            
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name=sym,
                increasing_line_color='#00e676',
                decreasing_line_color='#ff5252'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist['MA20'], 
                line=dict(color='rgba(255, 193, 7, 0.8)', width=1.5), 
                name="MA20"
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist['MA50'], 
                line=dict(color='rgba(0, 212, 255, 0.8)', width=1.5), 
                name="MA50"
            ), row=1, col=1)
            
            fig.add_trace(go.Bar(
                x=hist.index, y=hist['Volume'], 
                name="成交量",
                marker_color='rgba(0, 212, 255, 0.3)',
                marker_line_width=0
            ), row=2, col=1)
            
            fig.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_rangeslider_visible=False,
                showlegend=False,
                font=dict(color='rgba(255,255,255,0.4)', size=10),
            )
            fig.update_xaxes(
                gridcolor='rgba(255,255,255,0.04)',
                zerolinecolor='rgba(255,255,255,0.04)',
                color='rgba(255,255,255,0.3)',
            )
            fig.update_yaxes(
                gridcolor='rgba(255,255,255,0.04)',
                zerolinecolor='rgba(255,255,255,0.04)',
                color='rgba(255,255,255,0.3)',
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ---------- 底部 ----------
st.markdown(f"""
<div class="footer">
    <span style="opacity: 0.6;">⚡ 数据每日自动更新 · 系统运行正常 · {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
</div>
""", unsafe_allow_html=True)
