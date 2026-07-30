import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import yfinance as yf
import numpy as np

st.set_page_config(layout="wide", page_title="每日股票看板", initial_sidebar_state="expanded")

# ---------- 自定义 CSS ----------
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #141b2d 50%, #1a2335 100%); }
    .metric-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px 24px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(0,200,255,0.3);
        box-shadow: 0 8px 40px rgba(0,200,255,0.08);
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
    }
    .stock-card {
        background: rgba(255,255,255,0.03);
        border-radius: 16px;
        padding: 16px;
        border: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 12px;
    }
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.2);
        font-size: 12px;
        padding: 40px 0 20px 0;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- 侧边栏设置 ----------
st.sidebar.title("⚙️ 控制面板")
refresh = st.sidebar.button("🔄 强制刷新数据")
if refresh:
    st.cache_data.clear()
    st.rerun()

# 股票列表（与采集脚本一致）
STOCKS = [
    "MU", "AAOI", "GOOGL", "MSFT", "AMZN", "MRVL", "LITE", "SNDK", "NVDA", "ORCL", "SPCX", "SKHY", "TSLA",
    "0700.HK", "0883.HK", "3750.HK"
]

# 时间周期选择
period_map = {"1个月": "1mo", "3个月": "3mo", "6个月": "6mo", "1年": "1y"}
selected_period = st.sidebar.selectbox("📅 K线图周期", list(period_map.keys()), index=1)

# 排序选项
sort_option = st.sidebar.selectbox("📊 个股排序", ["默认顺序", "涨幅高→低", "涨幅低→高", "PE高→低", "PE低→高"])

# ---------- 标题 ----------
st.markdown(f"""
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
    <div>
        <span style="font-size: 32px; font-weight: 700; background: linear-gradient(135deg, #00d4ff, #7b61ff, #ff6b9d); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">📈 每日股票看板</span>
        <span style="font-size: 14px; color: rgba(255,255,255,0.3); margin-left: 16px;">QUANT DASHBOARD</span>
    </div>
    <div style="font-size: 13px; color: rgba(255,255,255,0.25); background: rgba(255,255,255,0.05); padding: 6px 16px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.06);">
        📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>
</div>
<hr style="border: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(0,212,255,0.2), transparent); margin: 12px 0 24px 0;">
""", unsafe_allow_html=True)

# ---------- 加载数据 ----------
@st.cache_data(ttl=3600)
def load_csv_data():
    macro_df = pd.read_csv("data/macro.csv") if os.path.exists("data/macro.csv") else None
    stocks_df = pd.read_csv("data/stocks.csv") if os.path.exists("data/stocks.csv") else None
    sox_df = pd.read_csv("data/sox.csv") if os.path.exists("data/sox.csv") else None
    report = ""
    if os.path.exists("data/report.md"):
        with open("data/report.md", "r", encoding="utf-8") as f:
            report = f.read()
    return macro_df, stocks_df, sox_df, report

macro_df, stocks_df, sox_df, report = load_csv_data()

# ---------- 宏观仪表盘 ----------
st.markdown('<div class="section-title">🌍 宏观数据</div>', unsafe_allow_html=True)
if macro_df is not None and not macro_df.empty:
    # 选择显示的重要宏观列
    macro_cols = ["VIX", "美元指数", "标普500", "纳斯达克100", "黄金", "WTI原油", "10年期美债收益率",
                  "美国国债规模", "芝加哥联储杠杆指数", "2年期实际利率", "FINRA保证金债务", "Volume PCR", "OI PCR"]
    available = [c for c in macro_cols if c in macro_df.columns]
    cols = st.columns(len(available))
    for i, col in enumerate(available):
        val = macro_df.iloc[0][col]
        if isinstance(val, (int, float)):
            display = f"{val:.2f}" if col != "FINRA保证金债务" else f"{val:,.0f}"
        else:
            display = str(val)
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{col}</div>
                <div class="metric-value">{display}</div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.warning("暂无宏观数据")

# ---------- SOX ----------
st.markdown('<div class="section-title">📉 SOX 半导体指数</div>', unsafe_allow_html=True)
if sox_df is not None and not sox_df.empty:
    sox_row = sox_df.iloc[0]
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">📊 最新价</div><div class="metric-value">{sox_row.get('最新价','N/A')}</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">📉 回撤</div><div class="metric-value" style="color: {'#ff5252' if float(sox_row.get('回撤%',0)) < -20 else '#ffd54f'};">{sox_row.get('回撤%','N/A')}%</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">📈 RSI(14)</div><div class="metric-value">{sox_row.get('RSI','N/A')}</div></div>""", unsafe_allow_html=True)
    with col4:
        status = "🐻 熊市" if sox_row.get('技术性熊市') else "🐂 非熊市"
        st.markdown(f"""<div class="metric-card"><div class="metric-label">📌 状态</div><div class="metric-value" style="font-size: 18px; color: {'#ff5252' if sox_row.get('技术性熊市') else '#00e676'};">{status}</div></div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">📌 MA20</div><div class="metric-value" style="font-size: 20px;">{sox_row.get('MA20','N/A')}</div></div>""", unsafe_allow_html=True)
    if '信号列表' in sox_row and sox_row['信号列表']:
        st.markdown(f"""<div style="background: rgba(255,255,255,0.03); border-radius: 12px; padding: 12px 20px; margin-top: 12px; border-left: 2px solid rgba(0,212,255,0.3);"><span style="color: rgba(255,255,255,0.5);">⚡ 关键信号</span><span style="color: rgba(255,255,255,0.7); margin-left: 12px;">{sox_row['信号列表']}</span></div>""", unsafe_allow_html=True)
else:
    st.warning("暂无 SOX 数据")

# ---------- 个股展示（支持排序） ----------
st.markdown('<div class="section-title">📊 个股数据</div>', unsafe_allow_html=True)
if stocks_df is not None and not stocks_df.empty:
    # 排序
    if sort_option == "涨幅高→低":
        stocks_df = stocks_df.sort_values("涨跌幅%", ascending=False)
    elif sort_option == "涨幅低→高":
        stocks_df = stocks_df.sort_values("涨跌幅%", ascending=True)
    elif sort_option == "PE高→低":
        stocks_df = stocks_df.sort_values("PE Ratio", ascending=False, na_position='last')
    elif sort_option == "PE低→高":
        stocks_df = stocks_df.sort_values("PE Ratio", ascending=True, na_position='last')
    else:
        # 默认顺序按 STOCKS 列表
        stocks_df['order'] = stocks_df['symbol'].map({s:i for i,s in enumerate(STOCKS)})
        stocks_df = stocks_df.sort_values('order')
    
    for _, row in stocks_df.iterrows():
        sym = row['symbol']
        if row.get('error'):
            st.warning(f"无法获取 {sym} 数据")
            continue
        close = row.get('收盘价', 'N/A')
        change = row.get('涨跌幅%', 0)
        vol_status = row.get('量比状态', 'N/A')
        pe = row.get('PE Ratio', 'N/A')
        rsi = row.get('RSI(14)', 'N/A')
        ma20 = row.get('MA20', 'N/A')
        change_color = "#00e676" if change > 0 else "#ff5252" if change < 0 else "#ffd54f"
        change_symbol = "▲" if change > 0 else "▼" if change < 0 else "●"
        display_name = sym.replace(".HK", "")
        st.markdown(f"""
        <div class="stock-card">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
                <div>
                    <span style="font-weight: 600; font-size: 18px; color: #ffffff;">{display_name}</span>
                    <span style="margin-left: 12px; font-size: 14px; color: {change_color};">{change_symbol} {change:.2f}%</span>
                </div>
                <div style="display: flex; gap: 20px; flex-wrap: wrap; font-size: 13px; color: rgba(255,255,255,0.6);">
                    <span>💰 收盘: {close}</span>
                    <span>📈 PE: {pe}</span>
                    <span>📊 量比: {vol_status}</span>
                    <span>📉 RSI: {rsi:.2f}</span>
                    <span>📌 MA20: {ma20:.2f}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # K线图（根据所选周期）
        try:
            hist = yf.download(sym, period=period_map[selected_period], progress=False)
            if not hist.empty:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name=sym), row=1, col=1)
                # 均线
                ma20_hist = hist['Close'].rolling(20).mean()
                ma50_hist = hist['Close'].rolling(50).mean()
                fig.add_trace(go.Scatter(x=hist.index, y=ma20_hist, line=dict(color='orange', width=1), name="MA20"), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=ma50_hist, line=dict(color='cyan', width=1), name="MA50"), row=1, col=1)
                fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name="成交量", marker_color='rgba(0,212,255,0.3)'), row=2, col=1)
                fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False, showlegend=False)
                fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.04)', color='rgba(255,255,255,0.3)')
                fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.04)', color='rgba(255,255,255,0.3)')
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        except:
            st.caption("K线图加载失败")
else:
    st.warning("暂无个股数据")

# ---------- AI 报告 ----------
if report:
    st.markdown("---")
    st.markdown('<div class="section-title">📝 每日分析报告（含操作建议）</div>', unsafe_allow_html=True)
    st.markdown(report)
else:
    st.info("今日报告尚未生成，请等待数据更新。")

# ---------- 历史趋势图（宏观指标） ----------
st.markdown("---")
st.markdown('<div class="section-title">📈 宏观历史趋势</div>', unsafe_allow_html=True)
if macro_df is not None and len(macro_df) > 1:
    # 选择展示的宏观指标（需历史数据，我们这里用最近30天的每日数据，但macro.csv只有一行，所以不能绘历史）
    # 替代：使用 yfinance 实时获取部分宏观历史
    macro_hist_symbols = {"VIX": "^VIX", "10年期美债收益率": "^TNX", "标普500": "^GSPC"}
    st.info("以下展示部分宏观指标近30天走势（实时获取）")
    for name, sym in macro_hist_symbols.items():
        try:
            hist = yf.download(sym, period="1mo", progress=False)
            if not hist.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name=name, line=dict(color='#00d4ff')))
                fig.update_layout(height=200, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)', color='rgba(255,255,255,0.3)')
                fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)', color='rgba(255,255,255,0.3)')
                st.plotly_chart(fig, use_container_width=True)
        except:
            pass
else:
    st.caption("暂无历史数据，需多日积累")

# ---------- 底部 ----------
st.markdown(f"""
<div class="footer">
    <span>⚡ 数据每日自动更新 · 系统运行正常 · {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
</div>
""", unsafe_allow_html=True)
