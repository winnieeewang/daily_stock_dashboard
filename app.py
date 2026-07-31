import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import json
import yfinance as yf

st.set_page_config(layout="wide", page_title="每日股票看板", initial_sidebar_state="collapsed")

# ---------- 自定义 CSS（浅色专业风格） ----------
st.markdown("""
<style>
    /* 全局背景 - 浅色 */
    .stApp {
        background: #f5f7fa;
    }
    
    /* 顶部导航栏 */
    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 24px;
        background: #ffffff;
        border-bottom: 1px solid #e8ecf1;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .top-nav .logo {
        font-size: 20px;
        font-weight: 700;
        color: #1a2335;
        letter-spacing: 1px;
    }
    .top-nav .logo span { color: #0066cc; }
    .top-nav .search-area {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .top-nav .search-box {
        background: #f0f2f6;
        border: 1px solid #dce0e6;
        border-radius: 8px;
        padding: 6px 16px;
        color: #1a2335;
        font-size: 13px;
        width: 280px;
    }
    .top-nav .push-btn {
        background: rgba(0,102,204,0.08);
        color: #0066cc;
        border: 1px solid rgba(0,102,204,0.15);
        border-radius: 8px;
        padding: 6px 16px;
        font-size: 13px;
        cursor: pointer;
    }
    
    /* 左侧导航 */
    .side-nav {
        display: flex;
        flex-direction: column;
        gap: 4px;
        padding: 8px 0;
    }
    .side-nav .nav-item {
        padding: 10px 16px;
        border-radius: 8px;
        color: #5a6a7a;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .side-nav .nav-item:hover {
        background: #f0f4f9;
        color: #1a2335;
    }
    .side-nav .nav-item.active {
        background: rgba(0,102,204,0.08);
        color: #0066cc;
        border-left: 3px solid #0066cc;
    }
    .side-nav .nav-divider {
        border-top: 1px solid #e8ecf1;
        margin: 8px 0;
    }
    
    /* 主内容卡片 */
    .main-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 20px 24px;
        border: 1px solid #e8ecf1;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .main-card .card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
    }
    .main-card .card-title {
        font-size: 16px;
        font-weight: 600;
        color: #1a2335;
    }
    .main-card .card-time {
        font-size: 12px;
        color: #8a9aa8;
    }
    
    /* 持仓卡片 */
    .position-card {
        background: #fafbfc;
        border-radius: 16px;
        padding: 20px 24px;
        border: 1px solid #e8ecf1;
        margin-bottom: 12px;
        transition: all 0.3s ease;
    }
    .position-card:hover {
        border-color: #c0d0e0;
        background: #f5f8fc;
    }
    .pos-symbol {
        font-size: 22px;
        font-weight: 700;
        color: #1a2335;
    }
    .pos-name {
        font-size: 14px;
        color: #8a9aa8;
        margin-left: 8px;
    }
    .pos-price {
        font-size: 20px;
        font-weight: 600;
        color: #1a2335;
    }
    .pos-change {
        font-size: 16px;
        font-weight: 500;
        margin-left: 10px;
    }
    .pos-change.up { color: #00a854; }
    .pos-change.down { color: #e53e3e; }
    .pos-tag {
        padding: 2px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
    }
    .pos-tag.buy { background: rgba(0,168,84,0.12); color: #007a3d; }
    .pos-tag.hold { background: rgba(237,162,0,0.12); color: #b37400; }
    .pos-tag.sell { background: rgba(229,62,62,0.12); color: #b22222; }
    
    /* 狙击点位卡片 */
    .sniper-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 12px 0;
    }
    .sniper-item {
        background: #fafbfc;
        border-radius: 12px;
        padding: 12px 16px;
        border: 1px solid #e8ecf1;
        text-align: center;
    }
    .sniper-item .label {
        font-size: 10px;
        color: #8a9aa8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .sniper-item .value {
        font-size: 15px;
        font-weight: 600;
        color: #1a2335;
        margin-top: 2px;
    }
    .sniper-item .value.buy1 { color: #0066cc; }
    .sniper-item .value.buy2 { color: #00a854; }
    .sniper-item .value.stop { color: #e53e3e; }
    .sniper-item .value.target { color: #d4a017; }
    
    /* 板块标签 */
    .sector-tag {
        display: inline-block;
        background: rgba(0,102,204,0.06);
        color: #0066cc;
        padding: 2px 14px;
        border-radius: 20px;
        font-size: 12px;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    
    /* 情绪标签 */
    .sentiment-badge {
        padding: 4px 16px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 500;
    }
    .sentiment-badge.bull { background: rgba(0,168,84,0.1); color: #007a3d; }
    .sentiment-badge.bear { background: rgba(229,62,62,0.1); color: #b22222; }
    .sentiment-badge.neutral { background: rgba(237,162,0,0.1); color: #b37400; }
    
    /* 评分圆环容器 */
    .score-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    
    /* 底部 */
    .footer {
        text-align: center;
        color: #b0c0d0;
        font-size: 11px;
        padding: 20px 0 10px 0;
        border-top: 1px solid #e8ecf1;
        margin-top: 30px;
    }
    
    /* 报告框 */
    .report-box {
        background: #fafbfc;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #e8ecf1;
        color: #1a2335;
        font-size: 14px;
        line-height: 1.6;
    }
    
    /* SOX 信号文本 */
    .sox-signal {
        background: #f0f4f9;
        border-radius: 8px;
        padding: 8px 16px;
        margin-top: 8px;
        border-left: 3px solid #0066cc;
        color: #2a3a4a;
        font-size: 13px;
    }
    
    /* 微指标卡片 */
    .metric-card {
        background: #fafbfc;
        border-radius: 12px;
        padding: 12px 16px;
        border: 1px solid #e8ecf1;
        text-align: center;
    }
    .metric-card .label {
        font-size: 11px;
        color: #8a9aa8;
        text-transform: uppercase;
    }
    .metric-card .value {
        font-size: 20px;
        font-weight: 600;
        color: #1a2335;
    }
    
    /* 侧边栏背景 */
    .css-1d391kg { background: #ffffff; }
    .css-1d391kg .stRadio label { color: #1a2335; }
    
    /* 调整输入框等 */
    .stSelectbox label, .stRadio label, .stButton button {
        color: #1a2335 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- 顶部导航栏 ----------
st.markdown("""
<div class="top-nav">
    <div class="logo">📈 DSA<span>.</span></div>
    <div class="search-area">
        <input class="search-box" placeholder="输入股票代码或名称，如600519、贵州茅台、AAPL" disabled>
        <span class="push-btn">🔔 推送通知</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- 侧边栏（左侧导航） ----------
with st.sidebar:
    st.markdown("""
    <div class="side-nav">
        <div class="nav-item active">🏠 首页</div>
        <div class="nav-item">📊 历史分析</div>
        <div class="nav-item">🔄 重新分析</div>
        <div class="nav-item">💬 追问AI</div>
        <div class="nav-item">📄 完整分析报告</div>
        <div class="nav-item">❓ 问股</div>
        <div class="nav-divider"></div>
        <div class="nav-item" style="font-size:12px;color:#8a9aa8;">⚙️ 设置</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    view = st.sidebar.radio("📌 视图切换", ["📊 大盘概览", "📈 个股详情", "🎯 AI决策卡片", "📝 每日报告"], label_visibility="collapsed")

# ---------- 加载数据 ----------
@st.cache_data(ttl=3600)
def load_data():
    macro_df = pd.read_csv("data/macro.csv") if os.path.exists("data/macro.csv") else None
    stocks_df = pd.read_csv("data/stocks.csv") if os.path.exists("data/stocks.csv") else None
    sox_df = pd.read_csv("data/sox.csv") if os.path.exists("data/sox.csv") else None
    report = ""
    if os.path.exists("data/report.md"):
        with open("data/report.md", "r", encoding="utf-8") as f:
            report = f.read()
    return macro_df, stocks_df, sox_df, report

@st.cache_data(ttl=3600)
def load_cards():
    if os.path.exists("data/cards.json"):
        try:
            with open("data/cards.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

macro_df, stocks_df, sox_df, report = load_data()
cards_data = load_cards()

def score_gauge(score, operation):
    color = "#00a854" if operation == "买入" else "#d4a017" if operation == "观望" else "#e53e3e"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'font': {'color': '#1a2335', 'size': 28}, 'suffix': {'text': '%'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#dce0e6', 'tickfont': {'color': '#5a6a7a'}},
            'bar': {'color': color},
            'bgcolor': 'rgba(255,255,255,0.8)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(229,62,62,0.1)'},
                {'range': [30, 60], 'color': 'rgba(237,162,0,0.1)'},
                {'range': [60, 100], 'color': 'rgba(0,168,84,0.1)'}
            ],
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(height=140, margin=dict(l=10, r=10, t=10, b=10),
                       paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#1a2335'))
    return fig

# ---------- 主内容 ----------
if view == "📊 大盘概览":
    st.markdown('<div class="main-card"><div class="card-header"><span class="card-title">🌍 宏观数据</span><span class="card-time">实时</span></div>', unsafe_allow_html=True)
    if macro_df is not None and not macro_df.empty:
        macro_cols = ["VIX", "美元指数", "标普500", "纳斯达克100", "黄金", "WTI原油", "10年期美债收益率"]
        available = [c for c in macro_cols if c in macro_df.columns]
        cols = st.columns(len(available))
        for i, col_name in enumerate(available):
            val = macro_df.iloc[0][col_name]
            display = f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
            with cols[i]:
                st.metric(label=col_name, value=display)
    else:
        st.warning("暂无宏观数据")
    st.markdown('</div>', unsafe_allow_html=True)

    # SOX
    st.markdown('<div class="main-card"><div class="card-header"><span class="card-title">📉 SOX 半导体指数</span></div>', unsafe_allow_html=True)
    if sox_df is not None and not sox_df.empty:
        sox_row = sox_df.iloc[0]
        cols = st.columns(5)
        with cols[0]: st.metric("📊 最新价", f"{sox_row.get('最新价', 'N/A'):.2f}")
        with cols[1]: st.metric("📉 回撤", f"{sox_row.get('回撤%', 0):.2f}%")
        with cols[2]: st.metric("📈 RSI(14)", f"{sox_row.get('RSI', 'N/A'):.2f}")
        with cols[3]: st.metric("📌 状态", "熊市" if sox_row.get('技术性熊市') else "非熊市")
        with cols[4]: st.metric("📌 MA20", f"{sox_row.get('MA20', 'N/A'):.2f}")
        if '信号列表' in sox_row and sox_row['信号列表']:
            signals = sox_row['信号列表']
            if isinstance(signals, str):
                signals = signals.split('；')
            st.markdown(f"<div class='sox-signal'>⚡ 关键信号：{'；'.join(signals)}</div>", unsafe_allow_html=True)
    else:
        st.warning("暂无 SOX 数据")
    st.markdown('</div>', unsafe_allow_html=True)

elif view == "📈 个股详情":
    st.markdown('<div class="main-card"><div class="card-header"><span class="card-title">📊 持仓个股</span><span class="card-time">技术面分析</span></div>', unsafe_allow_html=True)
    if stocks_df is not None and not stocks_df.empty:
        for _, row in stocks_df.iterrows():
            sym = row['symbol']
            if row.get('error'):
                continue
            close = row.get('收盘价', 0)
            change = row.get('涨跌幅%', 0)
            pe = row.get('PE Ratio', 'N/A')
            rsi = row.get('RSI(14)', 50)
            change_color = "up" if change > 0 else "down" if change < 0 else ""
            change_symbol = "▲" if change > 0 else "▼" if change < 0 else "●"
            display_name = sym.replace(".HK", "")
            
            st.markdown(f"""
            <div class="position-card">
                <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;">
                    <div style="display:flex;align-items:center;gap:12px;">
                        <span class="pos-symbol">{display_name}</span>
                        <span class="pos-price">${close:.2f}</span>
                        <span class="pos-change {change_color}">{change_symbol} {change:.2f}%</span>
                        <span style="font-size:12px;color:#8a9aa8;">PE: {pe if pe != 'N/A' else 'N/A'} | RSI: {rsi:.1f}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("暂无个股数据")
    st.markdown('</div>', unsafe_allow_html=True)

elif view == "🎯 AI决策卡片":
    st.markdown('<div class="main-card"><div class="card-header"><span class="card-title">🎯 AI 决策卡片</span><span class="card-time">DeepSeek 分析</span></div>', unsafe_allow_html=True)
    if cards_data and cards_data.get("stocks"):
        st.caption(f"生成时间：{cards_data.get('generated_at', 'N/A')}")
        for card in cards_data["stocks"]:
            sym = card.get("symbol", "N/A")
            op = card.get("operation", "观望")
            trend = card.get("trend", "")
            core_view = card.get("core_view", "")
            op_color = {"买入": "#00a854", "观望": "#d4a017", "卖出": "#e53e3e"}.get(op, "#0066cc")
            
            col_left, col_right = st.columns([3, 1])
            with col_left:
                st.markdown(f"""
                <div class="position-card">
                    <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                        <span class="pos-symbol" style="font-size:20px;">{sym}</span>
                        <span class="pos-tag" style="background:{op_color}22;color:{op_color};">{op}</span>
                        <span style="color:#8a9aa8;font-size:13px;">{trend}</span>
                    </div>
                    <div style="color:#2a3a4a;font-size:14px;line-height:1.6;">{core_view}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # 板块联动
                sectors = card.get("sectors", [])
                if sectors:
                    tags = "".join([f'<span class="sector-tag">{s}</span>' for s in sectors])
                    st.markdown(f'<div style="margin:8px 0 12px 0;"><span style="color:#8a9aa8;font-size:12px;">板块联动：</span>{tags}</div>', unsafe_allow_html=True)
                
                # 狙击点位
                sniper = card.get("sniper", {}) or {}
                st.markdown("""
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0;">
                """, unsafe_allow_html=True)
                sniper_data = [
                    ("🎯 理想买入", sniper.get("ideal_buy", "-"), "buy1"),
                    ("📈 二次买入", sniper.get("second_buy", "-"), "buy2"),
                    ("🛑 止损位", sniper.get("stop_loss", "-"), "stop"),
                    ("🏁 止盈目标", sniper.get("target", "-"), "target"),
                ]
                for label, val, cls in sniper_data:
                    st.markdown(f"""
                    <div class="sniper-item">
                        <div class="label">{label}</div>
                        <div class="value {cls}">{val}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # 催化 & 风险
                cc1, cc2 = st.columns(2)
                with cc1:
                    catalysts = card.get("catalysts", [])
                    if catalysts:
                        st.markdown(f'<div style="font-size:13px;color:#00a854;">✨ {"; ".join(catalysts)}</div>', unsafe_allow_html=True)
                with cc2:
                    risks = card.get("risks", [])
                    if risks:
                        st.markdown(f'<div style="font-size:13px;color:#e53e3e;">🚨 {"; ".join(risks)}</div>', unsafe_allow_html=True)
                
                # 查看分析依据
                with st.expander("📊 查看分析依据（原始技术指标）"):
                    stock_row = stocks_df[stocks_df['symbol'] == sym]
                    if not stock_row.empty:
                        row = stock_row.iloc[0]
                        fields = ['收盘价', '涨跌幅%', '成交量', '量比状态', 'RSI(14)', 'MACD', 'MACD信号', 
                                  'MA5', 'MA20', 'MA50', 'MA200', '布林上轨', '布林中轨', '布林下轨', 'ATR', 'PE Ratio']
                        cols = st.columns(4)
                        for idx, field in enumerate(fields):
                            val = row.get(field, 'N/A')
                            if isinstance(val, float):
                                val = f"{val:.2f}"
                            with cols[idx % 4]:
                                st.metric(field, val)
            with col_right:
                st.plotly_chart(score_gauge(card.get("score", 50), op), use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("📌 决策卡片尚未生成，请等待每日数据更新")
    st.markdown('</div>', unsafe_allow_html=True)

elif view == "📝 每日报告":
    st.markdown('<div class="main-card"><div class="card-header"><span class="card-title">📝 每日大盘总览</span><span class="card-time">AI 生成</span></div>', unsafe_allow_html=True)
    if report:
        st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)
    else:
        st.info("📌 今日报告尚未生成")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- 底部 ----------
st.markdown("""
<div class="footer">
    ⚡ 数据每日自动更新 · 系统运行正常
</div>
""", unsafe_allow_html=True)
