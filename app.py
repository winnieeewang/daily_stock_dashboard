import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import json
import yfinance as yf
import numpy as np

st.set_page_config(layout="wide", page_title="每日量化看盘 | Winnie Wang", initial_sidebar_state="expanded")

# ==================== 浅色专业主题 CSS ====================
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .stock-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid #e8eaed;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        margin-bottom: 14px;
        transition: all 0.2s ease;
    }
    .stock-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); border-color: #d0d5dd; }
    .tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
        line-height: 1.4;
    }
    .tag-buy { background: #e6f4ea; color: #137333; border: 1px solid #b7e1c6; }
    .tag-sell { background: #fce8e8; color: #c5221f; border: 1px solid #f5b8b8; }
    .tag-hold { background: #fff8e1; color: #b06000; border: 1px solid #ffe082; }
    .tag-bull { background: #e8f0fe; color: #1967d2; border: 1px solid #c4d8f5; }
    .tag-bear { background: #fce8e8; color: #c5221f; border: 1px solid #f5b8b8; }
    .tag-range { background: #f3e8fd; color: #7b1fa2; border: 1px solid #e1bee7; }
    .tag-reversal { background: #f3e8fd; color: #7b1fa2; border: 1px solid #e1bee7; }
    .tag-rebound { background: #e3f2fd; color: #1565c0; border: 1px solid #90caf9; }
    .tag-none { background: #f5f5f5; color: #757575; border: 1px solid #e0e0e0; }
    .bf-strong { background: #fce8e8; color: #c5221f; border: 1px solid #f5b8b8; }
    .bf-mid { background: #fff3e0; color: #e65100; border: 1px solid #ffcc80; }
    .bf-weak { background: #fff8e1; color: #f57f17; border: 1px solid #ffe082; }
    .bf-none { background: #f5f5f5; color: #9e9e9e; border: 1px solid #e0e0e0; }
    .price-big { font-size: 26px; font-weight: 700; color: #202124; letter-spacing: -0.5px; }
    .change-up { color: #00c853; font-weight: 700; font-size: 15px; }
    .change-down { color: #ff1744; font-weight: 700; font-size: 15px; }
    .metric-row { font-size: 13px; color: #5f6368; margin-top: 6px; }
    .metric-row b { color: #202124; font-weight: 600; }
    .advice-box {
        background: #f8f9fa;
        border-left: 3px solid #1967d2;
        padding: 10px 14px;
        border-radius: 0 10px 10px 0;
        margin-top: 10px;
        font-size: 13px;
        color: #3c4043;
        line-height: 1.5;
    }
    .advice-title { font-size: 11px; font-weight: 700; color: #1967d2; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; }
    .lev-danger { color: #c5221f; font-weight: 700; }
    .lev-warn { color: #e65100; font-weight: 600; }
    .lev-safe { color: #137333; font-weight: 500; }
    .sentiment-track {
        height: 10px;
        border-radius: 5px;
        background: linear-gradient(90deg, #d32f2f 0%, #f9a825 25%, #fbc02d 50%, #7cb342 75%, #388e3c 100%);
        position: relative;
        margin-top: 6px;
    }
    .sentiment-thumb {
        position: absolute;
        top: -5px;
        width: 20px;
        height: 20px;
        background: #fff;
        border: 3px solid #333;
        border-radius: 50%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        transform: translateX(-50%);
    }
    .news-item { font-size: 13px; color: #3c4043; padding: 6px 0; border-bottom: 1px solid #f0f0f0; }
    .news-item:last-child { border-bottom: none; }
    .source-link { font-size: 10px; color: #1967d2; text-decoration: none; margin-left: 4px; }
    .source-link:hover { text-decoration: underline; }
    hr.soft { border: 0; height: 1px; background: #e8eaed; margin: 16px 0; }
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #202124;
        margin: 24px 0 14px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .author-bar {
        background: linear-gradient(90deg, #1967d2 0%, #4285f4 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
    }
    .index-card {
        background: #fff;
        border: 1px solid #e8eaed;
        border-radius: 12px;
        padding: 14px;
        text-align: center;
    }
    .index-card .label { font-size: 11px; color: #9e9e9e; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .index-card .value { font-size: 20px; font-weight: 700; color: #202124; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

# ==================== 辅助函数（标签生成器） ====================
def bottom_fishing_badge(score: int):
    if score >= 70:
        return f'<span class="tag bf-strong">🔥 强抄底信号 {score}分</span>'
    elif score >= 50:
        return f'<span class="tag bf-mid">⚡ 抄底机会 {score}分</span>'
    elif score >= 30:
        return f'<span class="tag bf-weak">👀 观察区 {score}分</span>'
    else:
        return f'<span class="tag bf-none">抄底 {score}分</span>'

def reversal_badge(signal: str, conf: str):
    if signal == "反转":
        return f'<span class="tag tag-reversal">🚀 反转（{conf}）</span>'
    elif signal == "反弹":
        return f'<span class="tag tag-rebound">📈 反弹（{conf}）</span>'
    else:
        return f'<span class="tag tag-none">➖ 无信号</span>'

# ==================== 侧边栏 ====================
st.sidebar.title("⚙️ 控制面板")
if st.sidebar.button("🔄 强制刷新数据", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

period_map = {"1个月": "1mo", "3个月": "3mo", "6个月": "6mo", "1年": "1y"}
selected_period = st.sidebar.selectbox("📅 K线图周期", list(period_map.keys()), index=1)

st.sidebar.divider()
st.sidebar.caption("💡 分析师看盘顺序：情绪 → 宏观 → 消息 → 个股结构 → 短中期策略")

# ==================== 数据加载 ====================
@st.cache_data(ttl=3600)
def load_all_data():
    out = {}
    files = {
        "macro": "data/macro.csv",
        "stocks": "data/stocks.csv",
        "sox": "data/sox.csv",
        "sp500": "data/sp500.csv",
        "cards": "data/cards.json",
        "leverage": "data/leverage_risk.json",
        "news": "data/news.json",
        "report": "data/report.md",
    }
    for key, path in files.items():
        if not os.path.exists(path):
            out[key] = None
            continue
        try:
            if path.endswith(".csv"):
                out[key] = pd.read_csv(path)
            elif path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    out[key] = json.load(f)
            elif path.endswith(".md"):
                with open(path, "r", encoding="utf-8") as f:
                    out[key] = f.read()
        except Exception:
            out[key] = None
    return out

data = load_all_data()
macro_df = data.get("macro")
stocks_df = data.get("stocks")
sox_df = data.get("sox")
sp500_df = data.get("sp500")
cards_data = data.get("cards") or {}
leverage_data = data.get("leverage") or {}
news_data = data.get("news") or {}
report_md = data.get("report") or ""

cards_list = cards_data.get("stocks", []) if isinstance(cards_data, dict) else []
cards_map = {c["symbol"]: c for c in cards_list} if isinstance(cards_list, list) else {}
leverage_map = leverage_data.get("stocks", {}) if isinstance(leverage_data, dict) else {}

# ==================== 标题 + 作者授权 ====================
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">
    <div>
        <span style="font-size:26px;font-weight:800;color:#202124;">📈 每日量化看盘</span>
        <span style="font-size:13px;color:#5f6368;margin-left:10px;">PRO DASHBOARD</span>
        <span class="author-bar" style="margin-left:12px;">👤 Author: Winnie Wang</span>
    </div>
    <div style="font-size:12px;color:#5f6368;background:#fff;padding:4px 14px;border-radius:20px;border:1px solid #e8eaed;">
        📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>
</div>
<hr class="soft">
""", unsafe_allow_html=True)

# ==================== 1. 市场情绪仪表盘 ====================
st.markdown('<div class="section-title">🎚️ 市场情绪仪表盘</div>', unsafe_allow_html=True)

sentiment_score = 50
sentiment_label = "—"
if macro_df is not None and not macro_df.empty:
    s_idx = macro_df.iloc[0].get("情绪指数")
    s_lab = macro_df.iloc[0].get("情绪标签")
    if pd.notna(s_idx):
        sentiment_score = float(s_idx)
    if pd.notna(s_lab):
        sentiment_label = str(s_lab)

sentiment_color_map = {
    "极度贪婪": "#388e3c", "贪婪": "#7cb342", "中性": "#f9a825",
    "恐惧": "#f57c00", "极度恐惧": "#d32f2f"
}
sentiment_color = sentiment_color_map.get(sentiment_label, "#5f6368")

c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1, 1])
with c1:
    st.markdown(f"""
    <div class="stock-card" style="text-align:center;">
        <div style="font-size:12px;color:#5f6368;font-weight:600;letter-spacing:0.5px;">CNN 风格情绪指数</div>
        <div style="font-size:36px;font-weight:800;color:{sentiment_color};margin:4px 0;">{sentiment_score:.1f}</div>
        <div class="tag" style="background:{sentiment_color}15;color:{sentiment_color};border-color:{sentiment_color}40;">{sentiment_label}</div>
        <div class="sentiment-track">
            <div class="sentiment-thumb" style="left:{sentiment_score}%;border-color:{sentiment_color};"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:10px;color:#9e9e9e;margin-top:4px;">
            <span>极度恐惧</span><span>中性</span><span>极度贪婪</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    vix = macro_df.iloc[0].get("VIX", "N/A") if macro_df is not None else "N/A"
    vix_color = "#c5221f" if isinstance(vix, (int, float)) and vix > 25 else "#202124"
    st.markdown(f"""
    <div class="stock-card" style="text-align:center;">
        <div style="font-size:12px;color:#5f6368;font-weight:600;">VIX 波动率</div>
        <div style="font-size:28px;font-weight:700;color:{vix_color};margin-top:6px;">{vix if isinstance(vix, str) else f"{vix:.2f}"}</div>
        <div style="font-size:11px;color:#9e9e9e;margin-top:4px;">{'>25 警戒' if isinstance(vix, (int, float)) and vix > 25 else '正常区间'}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    sox_price = sox_df.iloc[0].get("最新价", "N/A") if sox_df is not None else "N/A"
    sox_draw = sox_df.iloc[0].get("回撤", 0) if sox_df is not None else 0
    sox_bear = sox_df.iloc[0].get("技术性熊市", False) if sox_df is not None else False
    sox_color = "#c5221f" if sox_bear else "#202124"
    st.markdown(f"""
    <div class="stock-card" style="text-align:center;">
        <div style="font-size:12px;color:#5f6368;font-weight:600;">SOX 半导体</div>
        <div style="font-size:28px;font-weight:700;color:{sox_color};margin-top:6px;">{sox_price if isinstance(sox_price, str) else f"{sox_price:.2f}"}</div>
        <div style="font-size:11px;color:#9e9e9e;margin-top:4px;">回撤 {sox_draw:.1f}% {'🐻 熊市' if sox_bear else ''}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    tnx = macro_df.iloc[0].get("10年期美债收益率", "N/A") if macro_df is not None else "N/A"
    st.markdown(f"""
    <div class="stock-card" style="text-align:center;">
        <div style="font-size:12px;color:#5f6368;font-weight:600;">10Y 美债收益率</div>
        <div style="font-size:28px;font-weight:700;color:#202124;margin-top:6px;">{tnx if isinstance(tnx, str) else f"{tnx:.2f}"}%</div>
        <div style="font-size:11px;color:#9e9e9e;margin-top:4px;">利率风向标</div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    dx = macro_df.iloc[0].get("美元指数", "N/A") if macro_df is not None else "N/A"
    st.markdown(f"""
    <div class="stock-card" style="text-align:center;">
        <div style="font-size:12px;color:#5f6368;font-weight:600;">美元指数</div>
        <div style="font-size:28px;font-weight:700;color:#202124;margin-top:6px;">{dx if isinstance(dx, str) else f"{dx:.2f}"}</div>
        <div style="font-size:11px;color:#9e9e9e;margin-top:4px;">流动性锚</div>
    </div>
    """, unsafe_allow_html=True)

# ==================== 2. 历史跨资产对比图（3/5/10/15年可调） ====================
st.markdown('<div class="section-title">📊 历史跨资产对比（标普500 / SOX / 10Y美债 / Mag 7）</div>', unsafe_allow_html=True)

hist_years = st.segmented_control("时间区间", ["3年", "5年", "10年", "15年"], default="10年")
period_str = {"3年": "3y", "5年": "5y", "10年": "10y", "15年": "15y"}[hist_years]

@st.cache_data(ttl=7200)
def fetch_history(period: str):
    tickers = {
        "标普500": "^GSPC",
        "SOX半导体": "^SOX",
        "10Y美债": "^TNX",
        "Mag7-MSFT": "MSFT",
        "Mag7-AAPL": "AAPL",
        "Mag7-GOOGL": "GOOGL",
        "Mag7-AMZN": "AMZN",
        "Mag7-NVDA": "NVDA",
        "Mag7-META": "META",
        "Mag7-TSLA": "TSLA",
    }
    result = {}
    for name, sym in tickers.items():
        try:
            df = yf.download(sym, period=period, progress=False)
            if not df.empty and len(df) > 1:
                close = df["Close"].dropna()
                if len(close) > 0 and close.iloc[0] != 0:
                    norm = (close / close.iloc[0]) * 100
                    result[name] = norm
        except Exception:
            pass
    return result

hist_data = fetch_history(period_str)

if hist_data:
    fig_hist = go.Figure()
    colors = {
        "标普500": "#1967d2", "SOX半导体": "#7b1fa2", "10Y美债": "#e65100",
        "Mag7-MSFT": "#00c853", "Mag7-AAPL": "#757575", "Mag7-GOOGL": "#4285f4",
        "Mag7-AMZN": "#ff9100", "Mag7-NVDA": "#76ff03", "Mag7-META": "#00838f", "Mag7-TSLA": "#d32f2f"
    }
    for name, series in hist_data.items():
        fig_hist.add_trace(go.Scatter(
            x=series.index, y=series.values,
            name=name, line=dict(color=colors.get(name, "#333"), width=1.5),
            hovertemplate="%{y:.1f}<extra>" + name + "</extra>"
        ))
    fig_hist.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(color="#5f6368", size=11),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        xaxis=dict(gridcolor="#f0f0f0", color="#9e9e9e"),
        yaxis=dict(
            gridcolor="#f0f0f0",
            color="#9e9e9e",
            title=dict(text="归一化指数 (起点=100)", font=dict(size=11)),
        ),
        hovermode="x unified",
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
else:
    st.info("历史数据加载中...")

# ==================== 3. 宏观速览（带数据来源链接） ====================
st.markdown('<div class="section-title">🌍 宏观速览 & 政策消息</div>', unsafe_allow_html=True)

SOURCE_LINKS = {
    "VIX": ("{:.2f}", "CBOE", "https://www.cboe.com/tradable_products/vix/"),
    "美元指数": ("{:.2f}", "ICE / WSJ", "https://www.wsj.com/market-data/quotes/futures/DX"),
    "标普500": ("{:.2f}", "S&P Dow Jones", "https://www.spglobal.com/spdji/en/indices/equity/sp-500/"),
    "纳斯达克100": ("{:.2f}", "Nasdaq", "https://www.nasdaq.com/market-activity/index/ndx"),
    "黄金": ("{:.2f}", "COMEX/CME", "https://www.cmegroup.com/markets/metals/precious/gold.html"),
    "WTI原油": ("{:.2f}", "NYMEX/CME", "https://www.cmegroup.com/markets/energy/crude-oil/light-sweet-crude.html"),
    "10年期美债收益率": ("{:.2f}%", "U.S. Treasury", "https://home.treasury.gov/data/treasury-coupon-issues-and-corporate-bond-yield-curve/daily-treasury-par-coupon-yield-rates"),
    "美国国债规模": ("{:.2f}T", "FRED / U.S. Treasury", "https://fred.stlouisfed.org/series/GFDEBTN"),
    "芝加哥联储杠杆指数": ("{:.2f}", "Chicago Fed / FRED", "https://fred.stlouisfed.org/series/NFCILEVERAGE"),
    "2年期实际利率（近似）": ("{}", "FRED (DGS2−T5YIE)", "https://fred.stlouisfed.org/series/DGS2"),
    "FINRA保证金债务": ("{}", "FINRA", "https://www.finra.org/rules-guidance/key-topics/margin-accounts"),
    "FINRA保证金债务YoY%": ("{:.1f}%", "FINRA", "https://www.finra.org/rules-guidance/key-topics/margin-accounts"),
    "Volume PCR": ("{:.2f}", "Alpha Vantage", "https://www.alphavantage.co/documentation/"),
    "OI PCR": ("{:.2f}", "Alpha Vantage", "https://www.alphavantage.co/documentation/"),
}

macro_cols = st.columns([2, 1])

with macro_cols[0]:
    if macro_df is not None and not macro_df.empty:
        m = macro_df.iloc[0].to_dict()
        keys = ["VIX", "标普500", "纳斯达克100", "黄金", "WTI原油", "美元指数",
                "10年期美债收益率", "美国国债规模", "芝加哥联储杠杆指数",
                "2年期实际利率（近似）", "FINRA保证金债务YoY%", "Volume PCR", "OI PCR"]
        avail = [k for k in keys if k in m]
        cols_per_row = 4
        for i in range(0, len(avail), cols_per_row):
            r = st.columns(cols_per_row)
            for j, k in enumerate(avail[i:i+cols_per_row]):
                v = m[k]
                fmt, src_name, src_link = SOURCE_LINKS.get(k, ("{}", "Yahoo Finance", "https://finance.yahoo.com"))
                if isinstance(v, (int, float)):
                    if "T" in fmt and v > 1e12:
                        disp = fmt.format(v / 1e12)
                    elif "%" in fmt and not str(v).endswith("%"):
                        disp = f"{v:.2f}%"
                    else:
                        disp = fmt.format(v) if "{}" not in fmt or isinstance(v, str) else f"{v:.2f}"
                else:
                    disp = str(v)
                with r[j]:
                    st.markdown(f"""
                    <div class="index-card" style="position:relative;">
                        <div class="label">{k}</div>
                        <div class="value">{disp}</div>
                        <a href="{src_link}" target="_blank" class="source-link">🔗 {src_name}</a>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("暂无宏观数据")

with macro_cols[1]:
    st.markdown('<div style="font-size:13px;font-weight:700;color:#5f6368;margin-bottom:8px;">📰 宏观 & 政策消息</div>', unsafe_allow_html=True)
    if news_data:
        all_news = []
        for sym, items in list(news_data.items())[:3]:
            if isinstance(items, list):
                for it in items[:2]:
                    title = it.get("title", "") if isinstance(it, dict) else str(it)
                    all_news.append(title)
        for title in all_news[:6]:
            st.markdown(f'<div class="news-item">• {title}</div>', unsafe_allow_html=True)
    else:
        st.caption("暂无新闻数据。如需自动抓取，请配置 SERPAPI_KEY。")

# ==================== 4. SOX + 标普500 双指数信号 ====================
st.markdown('<div class="section-title">📉 大盘指数信号（SOX + 标普500）</div>', unsafe_allow_html=True)

idx1, idx2 = st.columns(2)

with idx1:
    if sox_df is not None and not sox_df.empty:
        s = sox_df.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        metrics = [("最新价", s.get("最新价"), "{:.2f}"), ("回撤", s.get("回撤"), "{:.1f}%"), ("RSI", s.get("RSI"), "{:.1f}"), ("MA20", s.get("MA20"), "{:.2f}")]
        for col, (label, val, fmt) in zip([c1, c2, c3, c4], metrics):
            with col:
                disp = fmt.format(val) if pd.notna(val) else "N/A"
                st.markdown(f'<div class="index-card"><div class="label">{label}</div><div class="value">{disp}</div></div>', unsafe_allow_html=True)
        signals = str(s.get("信号列表", "")).split("；")
        if signals and signals[0]:
            st.caption("SOX: " + " | ".join([f"`{s}`" for s in signals if s]))
    else:
        st.info("暂无 SOX 数据")

with idx2:
    if sp500_df is not None and not sp500_df.empty:
        p = sp500_df.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        metrics = [("最新价", p.get("最新价"), "{:.2f}"), ("回撤", p.get("回撤%"), "{:.1f}%"), ("RSI", p.get("RSI"), "{:.1f}"), ("MA200", p.get("MA200"), "{:.2f}")]
        for col, (label, val, fmt) in zip([c1, c2, c3, c4], metrics):
            with col:
                disp = fmt.format(val) if pd.notna(val) else "N/A"
                st.markdown(f'<div class="index-card"><div class="label">{label}</div><div class="value">{disp}</div></div>', unsafe_allow_html=True)
        signals = str(p.get("信号列表", "")).split("；")
        if signals and signals[0]:
            st.caption("标普500: " + " | ".join([f"`{s}`" for s in signals if s]))
    else:
        st.info("暂无 标普500 数据")

# ==================== 5. 个股决策卡片（核心作战区） ====================
st.markdown('<div class="section-title">🎯 个股决策卡片</div>', unsafe_allow_html=True)

if stocks_df is None or stocks_df.empty:
    st.warning("⚠️ 暂无个股数据，请先运行 stock_dashboard.py")
    st.stop()

display_df = stocks_df.copy()
for old, new in [("涨跌幅%", "涨跌幅"), ("RSI(14)", "RSI_14")]:
    if old in display_df.columns and new not in display_df.columns:
        display_df[new] = display_df[old]

if cards_map:
    display_df["AI评分"] = display_df["symbol"].map(lambda x: cards_map.get(x, {}).get("score"))
    display_df["操作"] = display_df["symbol"].map(lambda x: cards_map.get(x, {}).get("operation", "—"))
    display_df["趋势"] = display_df["symbol"].map(lambda x: cards_map.get(x, {}).get("trend", "—"))
    display_df["核心观点"] = display_df["symbol"].map(lambda x: cards_map.get(x, {}).get("core_view", ""))
    display_df["短期建议"] = display_df["symbol"].map(lambda x: cards_map.get(x, {}).get("short_term", ""))
    display_df["中期建议"] = display_df["symbol"].map(lambda x: cards_map.get(x, {}).get("mid_term", ""))
    display_df["sectors"] = display_df["symbol"].map(lambda x: cards_map.get(x, {}).get("sectors", []))
    display_df["catalysts"] = display_df["symbol"].map(lambda x: cards_map.get(x, {}).get("catalysts", []))
    display_df["risks"] = display_df["symbol"].map(lambda x: cards_map.get(x, {}).get("risks", []))
    display_df["sniper"] = display_df["symbol"].map(lambda x: cards_map.get(x, {}).get("sniper", {}))

# 筛选项
f1, f2, f3, f4 = st.columns(4)
with f1:
    market_filter = st.selectbox("🏷 市场", ["全部", "美股", "港股"])
with f2:
    op_choices = ["全部"] + sorted(display_df.get("操作", pd.Series(["买入", "观望", "卖出"])).unique().tolist())
    op_filter = st.selectbox("📌 操作", op_choices)
with f3:
    rev_filter = st.selectbox("📐 反弹/反转", ["全部", "反转", "反弹", "无"])
with f4:
    bf_filter = st.selectbox("🔥 抄底信号", ["全部", "强信号(≥70)", "有机会(≥50)", "观察区(≥30)"])

if market_filter == "美股":
    display_df = display_df[~display_df["symbol"].str.endswith(".HK")]
elif market_filter == "港股":
    display_df = display_df[display_df["symbol"].str.endswith(".HK")]

if op_filter != "全部" and "操作" in display_df.columns:
    display_df = display_df[display_df["操作"] == op_filter]

if rev_filter != "全部":
    display_df = display_df[display_df.get("反弹反转信号", pd.Series([""] * len(display_df))) == rev_filter]

if bf_filter != "全部":
    scores = display_df.get("抄底评分", pd.Series([0] * len(display_df)))
    if bf_filter.startswith("强"):
        display_df = display_df[scores >= 70]
    elif bf_filter.startswith("有"):
        display_df = display_df[scores >= 50]
    elif bf_filter.startswith("观"):
        display_df = display_df[scores >= 30]

if "AI评分" in display_df.columns and display_df["AI评分"].notna().any():
    display_df = display_df.sort_values("AI评分", ascending=False, na_position="last")
elif "抄底评分" in display_df.columns:
    display_df = display_df.sort_values("抄底评分", ascending=False)

st.caption(f"共 {len(display_df)} / {len(stocks_df)} 只")

def op_tag_class(op): return {"买入": "tag-buy", "卖出": "tag-sell", "观望": "tag-hold"}.get(op, "tag-none")
def trend_tag_class(tr): return {"看多": "tag-bull", "看空": "tag-bear", "震荡": "tag-range"}.get(tr, "tag-none")
def rev_tag_class(rv): return {"反转": "tag-reversal", "反弹": "tag-rebound"}.get(rv, "tag-none")
def bf_tag_class(score):
    if score >= 70: return "bf-strong"
    if score >= 50: return "bf-mid"
    if score >= 30: return "bf-weak"
    return "bf-none"

for i in range(0, len(display_df), 3):
    cols = st.columns(3)
    for j in range(3):
        if i + j >= len(display_df):
            break
        r = display_df.iloc[i + j]
        sym = r["symbol"]
        card = cards_map.get(sym, {})

        with cols[j]:
            op = r.get("操作", "—")
            tr = r.get("趋势", "—")
            rev = r.get("反弹反转信号", "无")
            rev_conf = r.get("反弹反转置信度", "低")
            bf_score = int(r.get("抄底评分", 0))
            poc = r.get("资金集中价位")
            conc = r.get("集中度", 0)

            trend_color = {
                "看多": "border-left: 4px solid #52c41a;",
                "看空": "border-left: 4px solid #ff4d4f;",
                "震荡": "border-left: 4px solid #faad14;",
            }.get(tr, "border-left: 4px solid #d9d9d9;")

            # 卡片头部
            st.markdown(f"""
            <div style="{trend_color} background:#fafafa; padding:16px; border-radius:8px; margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="margin:0;">{sym}</h3>
                    <span style="font-size:12px; color:#666;">{tr}</span>
                </div>
                <div style="font-size:24px; font-weight:bold; margin:8px 0;">
                    {r.get('收盘价',0):.2f} 
                    <span style="font-size:14px; color:{'#52c41a' if r.get('涨跌幅',0)>=0 else '#ff4d4f'};">
                        {'+' if r.get('涨跌幅',0)>=0 else ''}{r.get('涨跌幅',0):.2f}%
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ========== 卡片内迷你周K线图 ==========
            try:
                hist = yf.download(sym, period="3mo", interval="1wk", progress=False)
                if not hist.empty and len(hist) > 3:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=hist.index,
                        open=hist["Open"],
                        high=hist["High"],
                        low=hist["Low"],
                        close=hist["Close"],
                        increasing_line_color="#00c853",
                        decreasing_line_color="#ff1744",
                        name=sym,
                    ))
                    if len(hist) >= 10:
                        ma10 = hist["Close"].rolling(10).mean()
                        fig.add_trace(go.Scatter(
                            x=hist.index, y=ma10,
                            line=dict(color="#f9a825", width=1.2),
                            name="MA10w", showlegend=False
                        ))
                    if len(hist) >= 30:
                        ma30 = hist["Close"].rolling(30).mean()
                        fig.add_trace(go.Scatter(
                            x=hist.index, y=ma30,
                            line=dict(color="#1967d2", width=1.2),
                            name="MA30w", showlegend=False
                        ))
                    fig.update_layout(
                        height=160,
                        margin=dict(l=0, r=0, t=2, b=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(visible=False, showgrid=False, rangeslider_visible=False),
                        yaxis=dict(
                            visible=True,
                            showgrid=False,
                            side="right",
                            tickfont=dict(size=9, color="#9e9e9e"),
                        ),
                        showlegend=False,
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"wk_{sym}_{i}_{j}")
                else:
                    st.caption(f"📉 {sym} 周K数据不足")
            except Exception:
                pass
            # =====================================

            # 核心指标
            m1, m2, m3 = st.columns(3)
            m1.metric("RSI", f"{r.get('RSI_14',0):.1f}")
            m2.metric("MACD", f"{r.get('MACD',0):.3f}")
            m3.metric("PE", r.get("PE_Ratio", "N/A"))

            # 标签：抄底 + 反弹/反转
            st.markdown(
                f"<div style='margin:6px 0;'>{bottom_fishing_badge(r.get('抄底评分', 0))}  {reversal_badge(r.get('反弹反转信号','无'), r.get('反弹反转置信度','低'))}</div>",
                unsafe_allow_html=True,
            )

            # 抄底依据
            reasons = r.get("抄底依据", [])
            if reasons and isinstance(reasons, list) and len(reasons) > 0:
                st.caption("🎯 " + " / ".join(reasons))

            # 反弹反转描述
            desc = r.get("反弹反转描述", "")
            if desc and desc != "暂无明确信号":
                st.caption(f"📐 {desc}")

            # POC
            if pd.notna(poc) and poc:
                st.caption(f"💰 POC 资金集中区: **{poc:.2f}** (占比 {conc:.1f}%)")

            # 杠杆预警
            lev_info = leverage_map.get(sym, {})
            lev_details = lev_info.get("details", {})
            max_risk = "低"
            min_atr = 999
            for ld in lev_details.values():
                if ld.get("风险等级") == "高":
                    max_risk = "高"
                elif ld.get("风险等级") == "中" and max_risk != "高":
                    max_risk = "中"
                atr_val = ld.get("距强平ATR倍数", 999)
                if isinstance(atr_val, (int, float)) and atr_val < min_atr:
                    min_atr = atr_val

            bf_text = f"🔥 抄底 {bf_score}分" if bf_score >= 50 else f"👀 抄底 {bf_score}分" if bf_score >= 30 else f"抄底 {bf_score}分"
            if max_risk == "高" or (isinstance(min_atr, (int, float)) and min_atr < 3):
                lev_text = f'<span class="lev-danger">⚠️ 杠杆高危 ({min_atr:.1f}倍ATR)</span>'
            elif max_risk == "中" or (isinstance(min_atr, (int, float)) and min_atr < 6):
                lev_text = f'<span class="lev-warn">⚡ 杠杆警戒 ({min_atr:.1f}倍ATR)</span>'
            else:
                lev_text = f'<span class="lev-safe">🛡️ 杠杆安全 ({min_atr:.1f}倍ATR)</span>' if min_atr != 999 else ""

            st.markdown(f"""
                <div style="margin-top:8px;">
                    <span class="tag {bf_tag_class(bf_score)}">{bf_text}</span>
                    {lev_text}
                </div>
            """, unsafe_allow_html=True)

            # AI 核心观点
            core = r.get("核心观点", "")
            if core:
                st.markdown(f'<div style="font-size:13px;color:#3c4043;margin-top:8px;line-height:1.5;">💡 {core}</div>', unsafe_allow_html=True)

            # 短中期策略
            short_term = r.get("短期建议", "") or card.get("short_term", "")
            mid_term = r.get("中期建议", "") or card.get("mid_term", "")
            if short_term or mid_term:
                advice_html = ""
                if short_term:
                    advice_html += f'<div class="advice-title">短期策略 (1-3天)</div><div>{short_term}</div>'
                if mid_term:
                    advice_html += f'<div class="advice-title" style="margin-top:6px;">中期策略 (1-4周)</div><div>{mid_term}</div>'
                st.markdown(f'<div class="advice-box">{advice_html}</div>', unsafe_allow_html=True)

            # 展开详情
            with st.expander("🔍 展开详情"):
                sniper = r.get("sniper", {}) or card.get("sniper", {})
                if sniper:
                    s1, s2, s3, s4 = st.columns(4)
                    for c, (label, val, color) in zip(
                        [s1, s2, s3, s4],
                        [("理想买入", sniper.get("ideal_buy", "—"), "#1967d2"),
                         ("二次加仓", sniper.get("second_buy", "—"), "#137333"),
                         ("止损位", sniper.get("stop_loss", "—"), "#c5221f"),
                         ("止盈目标", sniper.get("target", "—"), "#e65100")]
                    ):
                        with c:
                            st.markdown(f"""
                            <div style="background:#f8f9fa;border-radius:8px;padding:8px;text-align:center;border-top:2px solid {color};">
                                <div style="font-size:10px;color:#9e9e9e;font-weight:600;">{label}</div>
                                <div style="font-size:13px;color:#202124;font-weight:600;margin-top:2px;">{val}</div>
                            </div>
                            """, unsafe_allow_html=True)

                if lev_details:
                    st.write("**⚠️ 杠杆强平线**")
                    lev_cols = st.columns(len(lev_details))
                    for idx, (lev_key, ld) in enumerate(lev_details.items()):
                        atr_mult = ld.get("距强平ATR倍数", 999)
                        if isinstance(atr_mult, (int, float)):
                            lc, bc = ("#c5221f", "#fce8e8") if atr_mult < 3 else (("#e65100", "#fff3e0") if atr_mult < 6 else ("#137333", "#e6f4ea"))
                        else:
                            lc, bc = "#9e9e9e", "#f5f5f5"
                        with lev_cols[idx]:
                            st.markdown(f"""
                            <div style="background:{bc};border-radius:8px;padding:8px;text-align:center;border:1px solid {lc}30;">
                                <div style="font-size:10px;color:#5f6368;font-weight:600;">{lev_key} 强平价</div>
                                <div style="font-size:14px;color:#202124;font-weight:700;">${ld.get('强平价','N/A')}</div>
                                <div style="font-size:11px;color:{lc};font-weight:600;">{atr_mult}倍ATR</div>
                            </div>
                            """, unsafe_allow_html=True)

                sectors = r.get("sectors", []) or card.get("sectors", [])
                if sectors:
                    st.caption("🏷 " + " · ".join(sectors))

                catas = r.get("catalysts", []) or card.get("catalysts", [])
                risks = r.get("risks", []) or card.get("risks", [])
                if catas:
                    st.success("**利好**: " + " / ".join(catas))
                if risks:
                    st.error("**风险**: " + " / ".join(risks))

                sym_news = news_data.get(sym, []) if isinstance(news_data, dict) else []
                if sym_news:
                    st.write("**📰 相关新闻**")
                    for it in sym_news[:3]:
                        title = it.get("title", "") if isinstance(it, dict) else str(it)
                        st.markdown(f'<div class="news-item">• {title}</div>', unsafe_allow_html=True)

# ==================== 6. 个股周K线详细列表 ====================
st.markdown('<div class="section-title">📊 个股周K线详细走势</div>', unsafe_allow_html=True)

sort_option = st.selectbox("📊 排序方式", ["默认顺序", "涨幅高→低", "涨幅低→高", "PE高→低", "PE低→高", "抄底评分高→低"])
df_list = display_df.copy()
if sort_option == "涨幅高→低":
    df_list = df_list.sort_values("涨跌幅", ascending=False)
elif sort_option == "涨幅低→高":
    df_list = df_list.sort_values("涨跌幅", ascending=True)
elif sort_option == "PE高→低":
    df_list = df_list.sort_values("PE_Ratio", ascending=False, na_position="last")
elif sort_option == "PE低→高":
    df_list = df_list.sort_values("PE_Ratio", ascending=True, na_position="last")
elif sort_option == "抄底评分高→低":
    df_list = df_list.sort_values("抄底评分", ascending=False)

for _, row in df_list.iterrows():
    sym = row["symbol"]
    if row.get("error"):
        st.warning(f"无法获取 {sym} 数据")
        continue

    close = row.get("收盘价", 0)
    change = row.get("涨跌幅", 0)
    change_color = "#00c853" if change >= 0 else "#ff1744"
    change_sym = "▲" if change > 0 else "▼" if change < 0 else "●"
    display_name = sym.replace(".HK", "")

    st.markdown(f"""
    <div style="background:#fff;border:1px solid #e8eaed;border-radius:12px;padding:14px 18px;margin-bottom:10px;">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;">
            <div>
                <span style="font-weight:700;font-size:17px;color:#202124;">{display_name}</span>
                <span style="margin-left:10px;font-size:14px;color:{change_color};font-weight:600;">{change_sym} {change:.2f}%</span>
                <span style="margin-left:10px;font-size:14px;color:#5f6368;">${close:.2f}</span>
            </div>
            <div style="display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:#5f6368;">
                <span>RSI: {row.get('RSI_14',0):.1f}</span>
                <span>MACD: {row.get('MACD',0):.3f}</span>
                <span>PE: {row.get('PE_Ratio','N/A')}</span>
                <span>量比: {row.get('量比状态','—')}</span>
                <span>MA20: {row.get('MA20',0):.2f}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 周K线图
    try:
        hist = yf.download(sym, period=period_map[selected_period], interval="1wk", progress=False)
        if not hist.empty and len(hist) > 3:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.72, 0.28])
            fig.add_trace(go.Candlestick(
                x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"],
                name=sym, increasing_line_color="#00c853", decreasing_line_color="#ff1744"
            ), row=1, col=1)
            if len(hist) >= 10:
                ma10 = hist["Close"].rolling(10).mean()
                fig.add_trace(go.Scatter(x=hist.index, y=ma10, line=dict(color="#f9a825", width=1.2), name="MA10w"), row=1, col=1)
            if len(hist) >= 30:
                ma30 = hist["Close"].rolling(30).mean()
                fig.add_trace(go.Scatter(x=hist.index, y=ma30, line=dict(color="#1967d2", width=1.2), name="MA30w"), row=1, col=1)
            fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], name="成交量", marker_color="rgba(0,0,0,0.1)"), row=2, col=1)
            fig.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_rangeslider_visible=False,
                showlegend=False,
                font=dict(color="#5f6368", size=11)
            )
            fig.update_xaxes(gridcolor="#f0f0f0", color="#9e9e9e")
            fig.update_yaxes(gridcolor="#f0f0f0", color="#9e9e9e")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"detail_wk_{sym}")
        else:
            st.caption(f"📉 {sym} 周K数据不足")
    except Exception:
        st.caption(f"⚠️ {sym} 周K线图加载失败")

# ==================== 7. AI 大盘总览报告 ====================
if report_md:
    st.markdown('<div class="section-title">📝 AI 大盘总览</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="background:#fff;border:1px solid #e8eaed;border-radius:14px;padding:20px;color:#3c4043;font-size:14px;line-height:1.7;">{report_md}</div>', unsafe_allow_html=True)

# ==================== 底部 ====================
st.markdown(f"""
<div style="text-align:center;color:#9e9e9e;font-size:11px;padding:30px 0 10px;border-top:1px solid #e8eaed;margin-top:30px;">
    ⚡ 数据每日自动更新 · Author: Winnie Wang · {datetime.now().strftime('%Y-%m-%d %H:%M')} · 仅供研究参考，不构成投资建议
</div>
""", unsafe_allow_html=True)
