"""
app.py — 投资分析工作台 (Investment Copilot) v2.0

主入口 (Streamlit)。包含：
  - 🏠 Dashboard（首页）
      · 顶部状态条：US/HK/CN 市场状态 + F&G + 情绪 + VIX
      · 今日重点 / Top 机会 / Top 风险
      · 美股 / 港股全市场热力图
      · Economic Calendar + FedWatch
      · Morning Brief / Evening Recap
  - 🔍 个股深度分析（三栏布局）
      · 左：自选 / 历史分析 / 搜索
      · 中：股票概览 / AI 摘要 / K 线 / 资金 / 策略
      · 右：评分 / 风险 / 新闻 / 市场
  - 📊 历史跨资产对比（修复 NVDA 拆股 + 10Y 量纲，新增 Price/Return%/LogReturn 三档）

数据来源：
  - data/ 目录下的 CSV / JSON（由 stock_dashboard.py 每日生成）
  - yfinance 实时拉取
  - SerpApi 抓新闻
  - DeepSeek 生成 AI 摘要 / Brief
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

import utils as U  # 本地工具模块

# ---------------------------------------------------------------------------
# Streamlit 配置
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Investment Copilot | Winnie Wang",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("copilot_app")

# 全局 CSS（深色卡片 + 浅色文字，适配投资分析专业感）
GLOBAL_CSS = """
<style>
    :root {
        --bg: #f6f7fb;
        --card: #ffffff;
        --border: #e6e8ef;
        --text: #1a1d29;
        --text-dim: #6b7280;
        --accent: #2563eb;
        --up: #dc2626;   /* 涨 → 红（A股惯例） */
        --down: #16a34a; /* 跌 → 绿 */
        --neutral: #9ca3af;
    }
    .stApp { background: var(--bg); }
    .block-container { padding-top: 1.5rem; }

    .top-bar {
        background: linear-gradient(135deg, #1e293b 0%, #2563eb 100%);
        color: white;
        padding: 14px 20px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 14px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.12);
    }
    .top-bar .title { font-size: 22px; font-weight: 800; letter-spacing: -0.3px; }
    .top-bar .sub { font-size: 12px; opacity: 0.85; margin-top: 2px; }
    .top-bar .pill { background: rgba(255,255,255,0.18); padding: 4px 10px; border-radius: 12px; font-size: 12px; }

    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .card h4 { margin: 0 0 10px 0; font-size: 14px; color: var(--text-dim); font-weight: 600; }
    .card .big { font-size: 26px; font-weight: 800; color: var(--text); }
    .card .delta-up { color: var(--up); font-weight: 700; }
    .card .delta-down { color: var(--down); font-weight: 700; }

    .pill-up { background: #fef2f2; color: var(--up); padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; }
    .pill-down { background: #f0fdf4; color: var(--down); padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; }
    .pill-neutral { background: #f1f5f9; color: #475569; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; }

    .section-title {
        font-size: 16px;
        font-weight: 700;
        color: var(--text);
        margin: 18px 0 10px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-title .accent { color: var(--accent); }

    .stock-row {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 6px;
        transition: all 0.15s;
    }
    .stock-row:hover { border-color: var(--accent); transform: translateX(2px); }
    .stock-row.active { border-color: var(--accent); background: #eff6ff; }

    .ai-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 4px solid #0284c7;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 14px;
        line-height: 1.75;
        color: #0c4a6e;
    }
    .ai-box .label { font-size: 11px; font-weight: 700; color: #0369a1; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }

    .news-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 6px;
    }
    .news-card .title { font-size: 13px; font-weight: 600; color: var(--text); line-height: 1.4; }
    .news-card .meta { font-size: 10px; color: var(--text-dim); margin-top: 4px; }
    .news-card a { color: var(--accent); text-decoration: none; }
    .news-card a:hover { text-decoration: underline; }

    .gauge-bg { background: linear-gradient(90deg, #dc2626 0%, #f59e0b 25%, #fbbf24 50%, #84cc16 75%, #16a34a 100%); height: 8px; border-radius: 4px; position: relative; }
    .gauge-thumb { position: absolute; top: -4px; width: 16px; height: 16px; background: white; border: 3px solid #1a1d29; border-radius: 50%; transform: translateX(-50%); }

    .heat-tile { padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
    .heat-up { background: #fee2e2; color: #b91c1c; }
    .heat-down { background: #dcfce7; color: #166534; }
    .heat-flat { background: #f1f5f9; color: #475569; }

    .brief-box {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: #e2e8f0;
        padding: 22px 26px;
        border-radius: 14px;
        line-height: 1.85;
        font-size: 14px;
    }
    .brief-box h1, .brief-box h2, .brief-box h3 { color: #f8fafc; margin: 16px 0 8px 0; }
    .brief-box h1 { font-size: 18px; }
    .brief-box h2 { font-size: 16px; }
    .brief-box h3 { font-size: 15px; }
    .brief-box strong { color: #fbbf24; }
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 环境变量
# ---------------------------------------------------------------------------
SERPAPI_KEY = os.environ.get("SERPAPI", "") or st.secrets.get("SERPAPI", "") if hasattr(st, "secrets") else os.environ.get("SERPAPI", "")
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "") or (st.secrets.get("DEEPSEEK_API_KEY", "") if hasattr(st, "secrets") else "")
FRED_KEY = os.environ.get("FRED_API", "") or (st.secrets.get("FRED_API", "") if hasattr(st, "secrets") else "")

DATA_DIR = Path("data")


# ---------------------------------------------------------------------------
# 数据加载（带兜底）
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_static_data() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    files = {
        "macro": DATA_DIR / "macro.csv",
        "stocks": DATA_DIR / "stocks.csv",
        "sox": DATA_DIR / "sox.csv",
        "sp500": DATA_DIR / "sp500.csv",
        "cards": DATA_DIR / "cards.json",
        "leverage": DATA_DIR / "leverage_risk.json",
        "news": DATA_DIR / "news.json",
        "report": DATA_DIR / "report.md",
        "weekly": DATA_DIR / "weekly_report.md",
        "morning": DATA_DIR / "morning_brief.md",
        "evening": DATA_DIR / "evening_recap.md",
    }
    for key, p in files.items():
        if not p.exists():
            out[key] = None
            continue
        try:
            if p.suffix == ".csv":
                out[key] = pd.read_csv(p)
            elif p.suffix == ".json":
                with open(p, "r", encoding="utf-8") as f:
                    out[key] = json.load(f)
            else:  # md
                with open(p, "r", encoding="utf-8") as f:
                    out[key] = f.read()
        except Exception as e:
            logger.warning("读 %s 失败: %s", p, e)
            out[key] = None
    return out


DATA = load_static_data()
STOCKS_DF: Optional[pd.DataFrame] = DATA.get("stocks")
MACRO_DF: Optional[pd.DataFrame] = DATA.get("macro")
SOX_DF: Optional[pd.DataFrame] = DATA.get("sox")
SP500_DF: Optional[pd.DataFrame] = DATA.get("sp500")
CARDS_DATA = DATA.get("cards") or {}
NEWS_DATA = DATA.get("news") or {}
CARDS_LIST = CARDS_DATA.get("stocks", []) if isinstance(CARDS_DATA, dict) else []
CARDS_MAP = {c.get("symbol"): c for c in CARDS_LIST} if isinstance(CARDS_LIST, list) else {}
LEV_MAP = (DATA.get("leverage") or {}).get("stocks", {}) if isinstance(DATA.get("leverage"), dict) else {}


# ---------------------------------------------------------------------------
# 通用 helper
# ---------------------------------------------------------------------------
def color_for_change(v: float) -> str:
    """根据中国惯例：涨红跌绿。"""
    if v > 0:
        return "var(--up)"  # 红
    if v < 0:
        return "var(--down)"  # 绿
    return "var(--text-dim)"


def fmt_pct(v: float, with_sign: bool = True) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    arrow = "▲" if v > 0 else ("▼" if v < 0 else "●")
    sign = "+" if v > 0 and with_sign else ""
    return f"{arrow} {sign}{v:.2f}%"


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:  # noqa: BLE001
        return default


# ---------------------------------------------------------------------------
# 侧边栏：导航 + 控制
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Investment Copilot")
    st.caption("Author: Winnie Wang")
    st.divider()
    page = st.radio(
        "导航",
        ["🏠 Dashboard", "🔍 个股深度分析", "📊 跨资产对比", "📰 新闻中心", "⚙️ 数据诊断"],
        label_visibility="collapsed",
    )
    st.divider()
    if st.button("🔄 强制刷新数据", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("数据每日 GitHub Actions 自动更新；手动刷新会清缓存。")
    st.divider()
    period_map = {"1月": "1mo", "3月": "3mo", "6月": "6mo", "1年": "1y", "2年": "2y", "5年": "5y"}
    selected_period = st.selectbox("📅 K线周期", list(period_map.keys()), index=2)
    st.divider()
    st.caption("💡 分析师看盘顺序：\n情绪 → 宏观 → 消息 → 个股结构 → 短中期策略")


# ---------------------------------------------------------------------------
# 顶部状态条
# ---------------------------------------------------------------------------
status = U.market_status_now()
st.markdown(
    f"""
<div class="top-bar">
    <div>
        <div class="title">📈 Investment Copilot</div>
        <div class="sub">专业投资分析工作台 · {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
    </div>
    <div style="display:flex;gap:10px;align-items:center;">
        <span class="pill">🇺🇸 US: {status['us']}</span>
        <span class="pill">🇭🇰 HK: {status['hk']}</span>
        <span class="pill">🇨🇳 CN: {status['cn']}</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 页面：Dashboard
# ---------------------------------------------------------------------------
def page_dashboard():
    st.markdown('<div class="section-title"><span class="accent">🏠</span>市场全景 Dashboard</div>', unsafe_allow_html=True)

    # ===== 第一行：F&G / 情绪 / VIX / SOX / 10Y =====
    fg = U.calculate_fear_greed()
    macro_row = MACRO_DF.iloc[0] if MACRO_DF is not None and not MACRO_DF.empty else None
    sox_row = SOX_DF.iloc[0] if SOX_DF is not None and not SOX_DF.empty else None
    sentiment_score = safe_float(macro_row.get("情绪指数"), 50.0) if macro_row is not None else 50.0
    sentiment_label = str(macro_row.get("情绪标签", "中性")) if macro_row is not None else "中性"

    fg_label = fg.get("label", "中性")
    fg_score = fg.get("score", 50.0)
    fg_color = {"极度贪婪": "#16a34a", "贪婪": "#84cc16", "中性": "#fbbf24", "恐惧": "#f97316", "极度恐惧": "#dc2626"}.get(fg_label, "#9ca3af")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f"""
<div class="card" style="text-align:center;">
    <h4>😨 Fear &amp; Greed</h4>
    <div class="big" style="color:{fg_color};">{fg_score:.0f}</div>
    <div class="pill-neutral" style="display:inline-block;margin-top:4px;">{fg_label}</div>
    <div class="gauge-bg" style="margin-top:8px;">
        <div class="gauge-thumb" style="left:{fg_score}%;border-color:{fg_color};"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:9px;color:var(--text-dim);margin-top:4px;">
        <span>恐惧</span><span>中性</span><span>贪婪</span>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
    with c2:
        sent_color = {"极度贪婪": "#16a34a", "贪婪": "#84cc16", "中性": "#fbbf24", "恐惧": "#f97316", "极度恐惧": "#dc2626"}.get(sentiment_label, "#9ca3af")
        st.markdown(
            f"""
<div class="card" style="text-align:center;">
    <h4>🌡️ 自建情绪指数</h4>
    <div class="big" style="color:{sent_color};">{sentiment_score:.0f}</div>
    <div class="pill-neutral" style="display:inline-block;margin-top:4px;">{sentiment_label}</div>
    <div class="gauge-bg" style="margin-top:8px;">
        <div class="gauge-thumb" style="left:{sentiment_score}%;border-color:{sent_color};"></div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
    with c3:
        vix = safe_float(macro_row.get("VIX"), 0.0) if macro_row is not None else 0.0
        vix_color = "var(--up)" if vix > 25 else "var(--text)"
        st.markdown(
            f'<div class="card" style="text-align:center;"><h4>🌪️ VIX</h4><div class="big" style="color:{vix_color};">{vix:.2f}</div><div style="font-size:11px;color:var(--text-dim);margin-top:4px;">{"⚠️ 警戒" if vix > 25 else "正常"}</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        sox_px = safe_float(sox_row.get("最新价"), 0.0) if sox_row is not None else 0.0
        sox_dd = safe_float(sox_row.get("回撤"), 0.0) if sox_row is not None else 0.0
        sox_bear = bool(sox_row.get("技术性熊市", False)) if sox_row is not None else False
        st.markdown(
            f'<div class="card" style="text-align:center;"><h4>💾 SOX 半导体</h4><div class="big">{sox_px:.0f}</div><div style="font-size:11px;color:var(--text-dim);margin-top:4px;">回撤 {sox_dd:.1f}% {"🐻" if sox_bear else ""}</div></div>',
            unsafe_allow_html=True,
        )
    with c5:
        tnx = safe_float(macro_row.get("10年期美债收益率"), 0.0) if macro_row is not None else 0.0
        st.markdown(
            f'<div class="card" style="text-align:center;"><h4>🏛️ 10Y 美债</h4><div class="big">{tnx:.2f}%</div><div style="font-size:11px;color:var(--text-dim);margin-top:4px;">利率风向标</div></div>',
            unsafe_allow_html=True,
        )

    # ===== 第二行：今日重点 + Top 机会 + Top 风险 =====
    st.markdown('<div class="section-title">🎯 今日重点 & 自选扫描</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.2, 1, 1])

    # 今日重点：基于自选股 + 宏观 + 财报/事件，AI 生成的 Top 3 关注点
    with c1:
        st.markdown('<div class="card"><h4>📌 今日重点</h4>', unsafe_allow_html=True)
        # 静态 + 动态混合
        focuses = []
        if STOCKS_DF is not None and not STOCKS_DF.empty:
            # 涨幅最大
            if "涨跌幅" in STOCKS_DF.columns:
                top_up = STOCKS_DF.nlargest(1, "涨跌幅").iloc[0]
                focuses.append(
                    f"🚀 <b>{top_up['symbol']}</b> 今日涨 {safe_float(top_up.get('涨跌幅')):.2f}%，关注能否突破 / 短线见顶"
                )
                top_dn = STOCKS_DF.nsmallest(1, "涨跌幅").iloc[0]
                focuses.append(
                    f"🔻 <b>{top_dn['symbol']}</b> 今日跌 {safe_float(top_dn.get('涨跌幅')):.2f}%，关注是否到支撑 / 风险扩大"
                )
            # RSI 超卖 / 超买
            if "RSI_14" in STOCKS_DF.columns:
                oversold = STOCKS_DF[STOCKS_DF["RSI_14"] < 30]
                if not oversold.empty:
                    focuses.append(
                        f"🟢 RSI 超卖: {', '.join(oversold['symbol'].head(3).tolist())}，可能反弹"
                    )
        # 财报/事件
        cal = U.fetch_economic_calendar(SERPAPI_KEY)
        today_str = datetime.now().strftime("%m-%d")
        todays = [e for e in cal if today_str in e.get("date", "")]
        if not todays:
            todays = cal[:2]
        for e in todays[:2]:
            focuses.append(
                f"📅 {e.get('date','')} {e.get('event','')} ({e.get('importance','')})"
            )
        if not focuses:
            focuses = ["数据加载中…"]
        for f in focuses:
            st.markdown(f"<div style='font-size:13px;line-height:1.7;margin:6px 0;'>{f}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Top 机会
    with c2:
        st.markdown('<div class="card"><h4>🟢 Top 机会 (AI 评分)</h4>', unsafe_allow_html=True)
        if STOCKS_DF is not None and not STOCKS_DF.empty:
            tmp = STOCKS_DF.copy()
            if "AI评分" not in tmp.columns and CARDS_MAP:
                tmp["AI评分"] = tmp["symbol"].map(lambda x: CARDS_MAP.get(x, {}).get("score"))
            if "AI评分" in tmp.columns and tmp["AI评分"].notna().any():
                top = tmp[tmp["AI评分"].notna()].nlargest(5, "AI评分")
            else:
                top = tmp.head(5)
            for _, r in top.iterrows():
                sym = r["symbol"]
                score = r.get("AI评分", "—")
                chg = safe_float(r.get("涨跌幅"))
                st.markdown(
                    f"<div class='stock-row' style='display:flex;justify-content:space-between;align-items:center;'><span><b>{sym}</b> · 评分 <b style='color:var(--accent);'>{score}</b></span><span style='color:{color_for_change(chg)};font-weight:600;font-size:12px;'>{fmt_pct(chg)}</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("暂无个股数据")
        st.markdown("</div>", unsafe_allow_html=True)

    # Top 风险
    with c3:
        st.markdown('<div class="card"><h4>🔴 Top 风险</h4>', unsafe_allow_html=True)
        if STOCKS_DF is not None and not STOCKS_DF.empty:
            risks = []
            tmp = STOCKS_DF.copy()
            # RSI 超买
            if "RSI_14" in tmp.columns:
                for _, r in tmp[tmp["RSI_14"] > 70].head(3).iterrows():
                    risks.append((r["symbol"], f"RSI {safe_float(r.get('RSI_14')):.0f} 超买", "🟠"))
            # 跌幅大
            if "涨跌幅" in tmp.columns:
                for _, r in tmp.nsmallest(3, "涨跌幅").iterrows():
                    if not any(s == r["symbol"] for s, *_ in risks):
                        risks.append((r["symbol"], f"今日 {fmt_pct(safe_float(r.get('涨跌幅')))}", "🔴"))
            # 杠杆高危
            for sym, lev in (LEV_MAP or {}).items():
                if lev.get("综合风险等级") == "高" and not any(s == sym for s, *_ in risks):
                    risks.append((sym, "杠杆高危", "⚠️"))
            for sym, msg, ic in risks[:6]:
                st.markdown(
                    f"<div class='stock-row' style='display:flex;justify-content:space-between;align-items:center;'><span><b>{sym}</b> · {msg}</span><span>{ic}</span></div>",
                    unsafe_allow_html=True,
                )
            if not risks:
                st.caption("当前无显著风险信号")
        else:
            st.caption("暂无数据")
        st.markdown("</div>", unsafe_allow_html=True)

    # ===== 第三行：美股 / 港股热力图 =====
    st.markdown('<div class="section-title"><span class="accent">🔥</span>全市场热力图</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("🇺🇸 美股全市场 (按板块 / 当日涨跌幅)")
        with st.spinner("拉取美股热力图…"):
            us_hm = U.build_heatmap_data(U.US_HEATMAP_TICKERS)
        _draw_heatmap(us_hm, "sector", "symbol", "change_pct", "weight")
    with c2:
        st.caption("🇭🇰 港股全市场 (按板块 / 当日涨跌幅)")
        with st.spinner("拉取港股热力图…"):
            hk_hm = U.build_heatmap_data(U.HK_HEATMAP_TICKERS)
        _draw_heatmap(hk_hm, "sector", "symbol", "change_pct", "weight")

    # ===== 第四行：经济日历 + FedWatch =====
    st.markdown('<div class="section-title"><span class="accent">📅</span>经济日程 & FedWatch</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown('<div class="card"><h4>📆 Economic Calendar (未来 30 天)</h4>', unsafe_allow_html=True)
        cal = U.fetch_economic_calendar(SERPAPI_KEY)
        for e in cal[:12]:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid var(--border);font-size:13px;'>"
                f"<span style='min-width:80px;color:var(--text-dim);font-weight:600;'>{e.get('date','')}</span>"
                f"<span style='min-width:50px;color:var(--text-dim);font-size:11px;'>{e.get('time','')}</span>"
                f"<span style='flex:1;'>{e.get('event','')}</span>"
                f"<span style='font-size:11px;color:var(--text-dim);'>{e.get('importance','')}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h4>🎯 FedWatch 下次会议概率</h4>', unsafe_allow_html=True)
        fw = U.calc_fedwatch_from_futures()
        if "error" in fw:
            st.warning(f"FedWatch 暂不可用: {fw['error']}")
        else:
            st.markdown(
                f"<div style='font-size:13px;line-height:1.8;'>"
                f"下次会议: <b>{fw['next_meeting']}</b><br>"
                f"当前 FFR: <b>{fw['current_ffr']:.2f}%</b> · 隐含利率: <b>{fw['implied_rate']:.2f}%</b><br>"
                f"市场预期: <b>{fw['verdict']}</b></div>",
                unsafe_allow_html=True,
            )
            st.progress(int(fw["prob_cut"]), text=f"降息 {fw['prob_cut']:.0f}%")
            st.progress(int(fw["prob_hold"]), text=f"维持 {fw['prob_hold']:.0f}%")
            st.progress(int(fw["prob_hike"]), text=f"加息 {fw['prob_hike']:.0f}%")
            st.caption(f"数据源: {fw['source']}")
        st.markdown("</div>", unsafe_allow_html=True)

    # ===== 第五行：Morning Brief / Evening Recap =====
    st.markdown('<div class="section-title"><span class="accent">🧠</span>AI 报告 · Morning Brief &amp; Evening Recap</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### ☀️ Morning Brief (盘前)")
        mb = DATA.get("morning") or "（尚未生成）首次运行 stock_dashboard.py 后会自动写入 data/morning_brief.md"
        st.markdown(f'<div class="brief-box">{mb}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown("##### 🌙 Evening Recap (盘后)")
        er = DATA.get("evening") or "（尚未生成）首次运行 stock_dashboard.py 后会自动写入 data/evening_recap.md"
        st.markdown(f'<div class="brief-box">{er}</div>', unsafe_allow_html=True)


def _draw_heatmap(df: pd.DataFrame, sector_col: str, sym_col: str, color_col: str, size_col: str):
    """绘制 plotly treemap 热力图。"""
    if df is None or df.empty:
        st.info("热力图数据为空")
        return
    try:
        fig = px.treemap(
            df,
            path=[px.Constant("全市场"), sector_col, sym_col],
            values=size_col,
            color=color_col,
            color_continuous_scale=["#16a34a", "#84cc16", "#f1f5f9", "#fbbf24", "#dc2626"],
            color_continuous_midpoint=0,
            range_color=(-3, 3),
            custom_data=[color_col, "price"],
        )
        fig.update_traces(
            textinfo="label+text",
            texttemplate="<b>%{label}</b><br>%{customdata[0]:.2f}%",
            hovertemplate="<b>%{label}</b><br>涨跌: %{customdata[0]:.2f}%<br>价格: %{customdata[1]:.2f}<extra></extra>",
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=380, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:  # noqa: BLE001
        st.warning(f"热力图渲染失败: {e}")


# ---------------------------------------------------------------------------
# 页面：个股深度分析（三栏布局）
# ---------------------------------------------------------------------------
def page_stock_deepdive():
    st.markdown('<div class="section-title"><span class="accent">🔍</span>个股深度分析（三栏工作台）</div>', unsafe_allow_html=True)

    if STOCKS_DF is None or STOCKS_DF.empty:
        st.warning("⚠️ 暂无个股数据，请先运行 `python stock_dashboard.py`")
        st.stop()

    # 排序：AI 评分优先
    work_df = STOCKS_DF.copy()
    if CARDS_MAP and "AI评分" not in work_df.columns:
        work_df["AI评分"] = work_df["symbol"].map(lambda x: CARDS_MAP.get(x, {}).get("score"))
    if "AI评分" in work_df.columns and work_df["AI评分"].notna().any():
        work_df = work_df.sort_values("AI评分", ascending=False, na_position="last")

    all_symbols = work_df["symbol"].tolist()

    # ===== 三栏：左(自选/历史/搜索) | 中(分析主体) | 右(评分/风险/新闻/市场) =====
    left, mid, right = st.columns([1, 2.2, 1])

    # ============== 左栏 ==============
    with left:
        st.markdown("#### 👀 自选 & 搜索")
        q = st.text_input("🔍 搜索代码/名称", "")
        if q:
            all_symbols = [s for s in all_symbols if q.upper() in s.upper()]
        st.caption(f"共 {len(all_symbols)} 只")
        # 用 selectbox 选主标的
        if "selected_sym" not in st.session_state:
            st.session_state.selected_sym = all_symbols[0] if all_symbols else None
        sel = st.selectbox("主标的", all_symbols, index=all_symbols.index(st.session_state.selected_sym) if st.session_state.selected_sym in all_symbols else 0)
        st.session_state.selected_sym = sel

        # 自选列表（卡片式）
        st.markdown("##### 📋 自选列表")
        for sym in all_symbols[:15]:
            r = work_df[work_df["symbol"] == sym].iloc[0] if not work_df[work_df["symbol"] == sym].empty else None
            if r is None:
                continue
            chg = safe_float(r.get("涨跌幅"))
            rsi = safe_float(r.get("RSI_14"), 50)
            rsi_color = "#dc2626" if rsi > 70 else ("#16a34a" if rsi < 30 else "#6b7280")
            cls = "active" if sym == sel else ""
            st.markdown(
                f"<div class='stock-row {cls}' style='display:flex;justify-content:space-between;align-items:center;font-size:12px;'>"
                f"<span><b>{sym}</b> <span style='color:{rsi_color};font-size:10px;'>R{rsi:.0f}</span></span>"
                f"<span style='color:{color_for_change(chg)};font-weight:600;'>{fmt_pct(chg, with_sign=False)}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("##### 📜 历史分析存档")
        st.caption("data/cards.json 记录最近一次 AI 对每只股票的判断")
        card = CARDS_MAP.get(sel, {})
        if card:
            st.markdown(
                f"<div style='font-size:12px;line-height:1.7;color:var(--text-dim);'>"
                f"AI 评分: <b style='color:var(--accent);'>{card.get('score', '—')}</b><br>"
                f"操作: <b>{card.get('operation', '—')}</b><br>"
                f"趋势: <b>{card.get('trend', '—')}</b><br>"
                f"判断时间: {DATA.get('cards', {}).get('generated_at', '—') if isinstance(DATA.get('cards'), dict) else '—'}"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("该标的暂无 AI 存档")

    # ============== 中栏 ==============
    with mid:
        if not sel:
            st.info("请选择一只股票")
            return
        r = work_df[work_df["symbol"] == sel].iloc[0]
        sym = sel
        is_hk = sym.endswith(".HK")
        market_label = "港股" if is_hk else "美股"

        # 1) 股票概览
        chg = safe_float(r.get("涨跌幅"))
        price = safe_float(r.get("收盘价"))
        st.markdown(
            f"""
<div class="card">
    <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
            <div style="font-size:22px;font-weight:800;">{sym} <span style='font-size:13px;color:var(--text-dim);font-weight:500;'>{market_label}</span></div>
            <div style="font-size:11px;color:var(--text-dim);">data/stocks.csv · {DATA.get('cards', {}).get('generated_at', '—') if isinstance(DATA.get('cards'), dict) else '—'}</div>
        </div>
        <div style="text-align:right;">
            <div style="font-size:32px;font-weight:800;color:{color_for_change(chg)};">{price:.2f}</div>
            <div style="font-size:16px;font-weight:700;color:{color_for_change(chg)};">{fmt_pct(chg)}</div>
        </div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-top:14px;font-size:12px;">
        <div><span style='color:var(--text-dim);'>RSI</span><br><b>{safe_float(r.get('RSI_14'), 50):.1f}</b></div>
        <div><span style='color:var(--text-dim);'>MACD</span><br><b>{safe_float(r.get('MACD')):.3f}</b></div>
        <div><span style='color:var(--text-dim);'>PE</span><br><b>{r.get('PE_Ratio', 'N/A')}</b></div>
        <div><span style='color:var(--text-dim);'>量比</span><br><b>{r.get('量比状态','—')}</b></div>
        <div><span style='color:var(--text-dim);'>MA20</span><br><b>{safe_float(r.get('MA20')):.2f}</b></div>
        <div><span style='color:var(--text-dim);'>ATR</span><br><b>{safe_float(r.get('ATR')):.2f}</b></div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # 2) AI 摘要
        st.markdown('<div class="section-title">🧠 AI 摘要</div>', unsafe_allow_html=True)
        card = CARDS_MAP.get(sym, {})
        core = card.get("core_view") or r.get("核心观点", "")
        if core:
            st.markdown(
                f"<div class='ai-box'><div class='label'>核心判断</div>{core}</div>",
                unsafe_allow_html=True,
            )
        st_term = card.get("short_term") or r.get("短期建议", "")
        m_term = card.get("mid_term") or r.get("中期建议", "")
        if st_term or m_term:
            c1, c2 = st.columns(2)
            with c1:
                if st_term:
                    st.markdown(f"<div class='ai-box' style='border-left-color:#16a34a;'><div class='label' style='color:#15803d;'>短期 (1-3 天)</div>{st_term}</div>", unsafe_allow_html=True)
            with c2:
                if m_term:
                    st.markdown(f"<div class='ai-box' style='border-left-color:#7c3aed;'><div class='label' style='color:#6d28d9;'>中期 (1-4 周)</div>{m_term}</div>", unsafe_allow_html=True)

        # 3) K 线
        st.markdown(f'<div class="section-title">📈 K 线 · {selected_period}</div>', unsafe_allow_html=True)
        try:
            hist = _fetch_price_history(sym, period=period_map[selected_period], interval="1d")
            if not hist.empty and len(hist) > 5:
                _draw_kline(hist, sym, with_volume=True)
            else:
                st.caption(f"📉 {sym} 数据不足")
        except Exception as e:  # noqa: BLE001
            st.caption(f"⚠️ K线加载失败: {e}")

        # 4) 资金
        st.markdown('<div class="section-title">💰 资金面</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        poc = r.get("资金集中价位")
        conc = r.get("集中度", 0)
        vol = r.get("成交量")
        vol_pcr = r.get("Volume_PCR")
        oi_pcr = r.get("OI_PCR")
        c1.metric("POC 资金集中区", f"{poc:.2f}" if pd.notna(poc) and poc else "N/A")
        c2.metric("集中度", f"{conc:.1f}%" if pd.notna(conc) else "N/A")
        c3.metric("成交量", f"{int(vol):,}" if pd.notna(vol) else "N/A")
        c4.metric("Vol/OI PCR", f"{vol_pcr:.2f} / {oi_pcr:.2f}" if pd.notna(vol_pcr) and pd.notna(oi_pcr) else "N/A")

        # 5) 策略 (sniper + 杠杆)
        st.markdown('<div class="section-title">🎯 交易策略</div>', unsafe_allow_html=True)
        sniper = card.get("sniper", {}) or r.get("sniper", {})
        if sniper:
            c1, c2, c3, c4 = st.columns(4)
            for col, (label, val, color) in zip([c1, c2, c3, c4], [
                ("理想买入", sniper.get("ideal_buy", "—"), "#2563eb"),
                ("二次加仓", sniper.get("second_buy", "—"), "#16a34a"),
                ("止损位", sniper.get("stop_loss", "—"), "#dc2626"),
                ("止盈目标", sniper.get("target", "—"), "#f59e0b"),
            ]):
                with col:
                    st.markdown(f"<div class='card' style='text-align:center;border-top:3px solid {color};padding:10px;'><h4>{label}</h4><div style='font-size:14px;font-weight:700;color:{color};'>{val}</div></div>", unsafe_allow_html=True)

        lev = LEV_MAP.get(sym, {})
        if lev and lev.get("details"):
            st.markdown("##### ⚠️ 杠杆强平监控")
            lev_cols = st.columns(len(lev["details"]))
            for idx, (k, ld) in enumerate(lev["details"].items()):
                atr_mult = ld.get("距强平ATR倍数", 999)
                if isinstance(atr_mult, (int, float)):
                    color = "#dc2626" if atr_mult < 3 else ("#f59e0b" if atr_mult < 6 else "#16a34a")
                else:
                    color = "#9ca3af"
                with lev_cols[idx]:
                    st.markdown(
                        f"<div class='card' style='text-align:center;border-color:{color}40;padding:8px;'>"
                        f"<h4>{k} 强平价</h4>"
                        f"<div style='font-size:14px;font-weight:700;'>${ld.get('强平价', 'N/A')}</div>"
                        f"<div style='font-size:11px;color:{color};font-weight:600;'>{atr_mult}x ATR</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # 催化剂 / 风险
        cats = card.get("catalysts", [])
        rsks = card.get("risks", [])
        if cats or rsks:
            c1, c2 = st.columns(2)
            with c1:
                if cats:
                    st.success("**🚀 利好**: " + " · ".join(cats))
            with c2:
                if rsks:
                    st.error("**⚠️ 风险**: " + " · ".join(rsks))

    # ============== 右栏 ==============
    with right:
        if not sel:
            return
        sym = sel
        r = work_df[work_df["symbol"] == sym].iloc[0]
        card = CARDS_MAP.get(sym, {})

        # 评分
        st.markdown("#### 🎯 AI 评分")
        score = card.get("score") or 0
        op = card.get("operation", "—")
        tr = card.get("trend", "—")
        score_color = "#16a34a" if score >= 70 else ("#f59e0b" if score >= 50 else "#dc2626")
        st.markdown(
            f"<div class='card' style='text-align:center;'>"
            f"<div style='font-size:42px;font-weight:800;color:{score_color};'>{score}</div>"
            f"<div style='font-size:13px;color:var(--text-dim);margin-top:4px;'>操作: <b>{op}</b> · 趋势: <b>{tr}</b></div>"
            f"<div class='gauge-bg' style='margin-top:8px;'><div class='gauge-thumb' style='left:{min(score,100)}%;border-color:{score_color};'></div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        # 抄底/反弹
        bf = safe_float(r.get("抄底评分"), 0)
        rev = r.get("反弹反转信号", "无")
        rev_conf = r.get("反弹反转置信度", "低")
        st.markdown(
            f"<div class='card'>"
            f"<h4>🎯 抄底/反弹</h4>"
            f"<div style='font-size:13px;'>抄底评分: <b>{bf:.0f}</b> 分</div>"
            f"<div style='font-size:13px;'>反弹反转: <b>{rev}</b> ({rev_conf})</div>"
            f"<div style='font-size:11px;color:var(--text-dim);margin-top:4px;'>{r.get('反弹反转描述', '')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # 风险
        st.markdown("#### ⚠️ 风险")
        lev = LEV_MAP.get(sym, {})
        if lev:
            risk_lv = lev.get("综合风险等级", "低")
            risk_color = {"高": "#dc2626", "中": "#f59e0b", "低": "#16a34a"}.get(risk_lv, "#9ca3af")
            st.markdown(
                f"<div class='card' style='border-left:4px solid {risk_color};'>"
                f"<h4>杠杆风险等级</h4>"
                f"<div style='font-size:24px;font-weight:800;color:{risk_color};'>{risk_lv}</div>"
                f"<div style='font-size:11px;color:var(--text-dim);margin-top:4px;'>{lev.get('描述', '')}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # 新闻
        st.markdown("#### 📰 标的新闻")
        if SERPAPI_KEY:
            with st.spinner(f"拉取 {sym} 新闻…"):
                sym_news = U.fetch_stock_news(sym, SERPAPI_KEY, is_hk=sym.endswith(".HK"), top_n=5)
        else:
            sym_news = []
        if not sym_news and isinstance(NEWS_DATA, dict):
            sym_news = NEWS_DATA.get("stocks", {}).get(sym, []) if isinstance(NEWS_DATA.get("stocks"), dict) else []
        if sym_news:
            for n in sym_news[:5]:
                st.markdown(
                    f"<div class='news-card'>"
                    f"<a href='{n.get('link', '#')}' target='_blank' class='title'>{n.get('title','')}</a>"
                    f"<div class='meta'>{n.get('source','')} · {n.get('date','')}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("暂无新闻（请配置 SERPAPI）")

        # 市场上下文
        st.markdown("#### 🌍 市场上下文")
        sectors = card.get("sectors", [])
        if sectors:
            st.markdown("**板块**: " + " · ".join(sectors))
        st.caption(f"VIX {safe_float(MACRO_DF.iloc[0].get('VIX'), 0) if MACRO_DF is not None else 0:.2f} · 10Y {safe_float(MACRO_DF.iloc[0].get('10年期美债收益率'), 0) if MACRO_DF is not None else 0:.2f}%")


# ---------------------------------------------------------------------------
# 页面：跨资产对比（修复版）
# ---------------------------------------------------------------------------
def page_cross_asset():
    st.markdown('<div class="section-title"><span class="accent">📊</span>历史跨资产对比（标普500 / SOX / 10Y美债 / Mag 7）</div>', unsafe_allow_html=True)
    st.caption("✅ 修复：auto_adjust=True 解决 NVDA 拆股异常；10Y 用副轴显示量纲统一；三档视图切换。")

    c1, c2 = st.columns([1, 2])
    with c1:
        years = st.segmented_control("时间区间", ["3年", "5年", "10年", "15年"], default="10年")
        period_str = {"3年": "3y", "5年": "5y", "10年": "10y", "15年": "15y"}[years]
    with c2:
        mode = st.radio(
            "视图模式",
            ["📈 Price", "📊 Return %", "📉 Log Return"],
            horizontal=True,
            help="Price: 绝对价格（10Y 用副轴）; Return %: 累计回报%; Log Return: log(price/price[0])",
        )
        mode_key = "price" if "Price" in mode else ("return" if "Return" in mode else "log")

    with st.spinner("拉取历史数据…"):
        hist_data = U.fetch_history_fixed(period_str)

    if not hist_data:
        st.info("历史数据加载中…")
        return

    fig = go.Figure()
    colors = {
        "标普500": "#1e3a8a", "SOX半导体": "#7c3aed", "10Y美债收益率": "#ea580c",
        "Mag7-MSFT": "#0ea5e9", "Mag7-AAPL": "#6b7280", "Mag7-GOOGL": "#3b82f6",
        "Mag7-AMZN": "#f59e0b", "Mag7-NVDA": "#22c55e", "Mag7-META": "#0d9488",
        "Mag7-TSLA": "#dc2626",
    }

    # === 模式 1: Price (双轴：价格 vs 10Y) ===
    if mode_key == "price":
        # 10Y 单独画在副轴
        if "10Y美债收益率" in hist_data:
            tnx = hist_data["10Y美债收益率"]
            fig.add_trace(go.Scatter(
                x=tnx.index, y=tnx.values, name="10Y 收益率 (%)",
                line=dict(color=colors["10Y美债收益率"], width=2, dash="dot"),
                yaxis="y2",
                hovertemplate="%{y:.2f}%<extra>10Y 美债</extra>",
            ))
        # 价格类：先归一化到 e+18 量级以便在同图
        # 改方案：使用 log 价（量纲统一）展示在同一主轴
        for name, series in hist_data.items():
            if name == "10Y美债收益率":
                continue
            fig.add_trace(go.Scatter(
                x=series.index, y=series.values, name=name,
                line=dict(color=colors.get(name, "#333"), width=1.5),
                hovertemplate="%{y:.2f}<extra>" + name + "</extra>",
            ))
        fig.update_layout(
            yaxis=dict(title="价格 (绝对值，对数轴)", type="log", gridcolor="#e5e7eb"),
            yaxis2=dict(title="10Y 收益率 (%)", overlaying="y", side="right", gridcolor="rgba(0,0,0,0)"),
        )

    # === 模式 2: Return % (累计回报，起点 0%) ===
    elif mode_key == "return":
        for name, series in hist_data.items():
            if name == "10Y美债收益率":
                continue  # 10Y 是水平，不画累计回报
            base = series.iloc[0]
            if base == 0:
                continue
            ret = (series / base - 1.0) * 100
            fig.add_trace(go.Scatter(
                x=ret.index, y=ret.values, name=name,
                line=dict(color=colors.get(name, "#333"), width=1.5),
                hovertemplate="%{y:+.1f}%<extra>" + name + "</extra>",
            ))
        fig.update_layout(yaxis=dict(title="累计回报 (%)", gridcolor="#e5e7eb", zeroline=True, zerolinecolor="#9ca3af", zerolinewidth=1))

    # === 模式 3: Log Return ===
    else:
        for name, series in hist_data.items():
            if name == "10Y美债收益率":
                continue
            base = series.iloc[0]
            if base <= 0:
                continue
            logret = np.log(series / base)
            fig.add_trace(go.Scatter(
                x=logret.index, y=logret.values, name=name,
                line=dict(color=colors.get(name, "#333"), width=1.5),
                hovertemplate="%{y:+.2f}<extra>" + name + "</extra>",
            ))
        fig.update_layout(yaxis=dict(title="Log Return (相对起点)", gridcolor="#e5e7eb", zeroline=True, zerolinecolor="#9ca3af", zerolinewidth=1))

    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        font=dict(color="#6b7280", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        xaxis=dict(gridcolor="#f1f5f9"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # 数据表
    with st.expander("📋 查看原始数据"):
        st.dataframe(pd.DataFrame({n: s.tail(252) for n, s in hist_data.items()}), use_container_width=True, height=300)


# ---------------------------------------------------------------------------
# 页面：新闻中心
# ---------------------------------------------------------------------------
def page_news_center():
    st.markdown('<div class="section-title"><span class="accent">📰</span>新闻与消息中心</div>', unsafe_allow_html=True)

    if not SERPAPI_KEY:
        st.warning("⚠️ 未配置 SERPAPI_KEY。在 GitHub Secrets 添加 SERPAPI，或在 .streamlit/secrets.toml 设置 SERPAPI。")
        st.info("如已配置但仍未抓到，请检查：\n1. Secret 名是否为 `SERPAPI`（大写）\n2. workflow env 是否传入 SERPAPI（看 stock_dashboard.py run 那段的 env）\n3. SerpApi 账户是否还有免费额度（每月 100 次）")
    c1, c2, c3 = st.columns(3)
    if st.button("🔄 立即抓取一次新闻", use_container_width=True):
        with st.spinner("抓取宏观/政策/个股新闻…"):
            symbols = STOCKS_DF["symbol"].tolist() if STOCKS_DF is not None else []
            res = U.fetch_all_news(SERPAPI_KEY, symbols, DATA_DIR / "news.json")
            st.session_state["_news_res"] = res
            st.success(f"抓取完成: 宏观 {len(res['macro'])} / 政策 {len(res['policy'])} / 个股 {sum(len(v) for v in res['stocks'].values())}")

    res = st.session_state.get("_news_res") or NEWS_DATA

    if isinstance(res, dict):
        if "macro" in res or "policy" in res:
            # 新结构（来自 fetch_all_news）
            tab_m, tab_p, tab_s = st.tabs(["🌍 宏观", "🏛️ 政策", "📊 自选股"])
            with tab_m:
                _show_news_list(res.get("macro", []))
            with tab_p:
                _show_news_list(res.get("policy", []))
            with tab_s:
                sym = st.selectbox("选择股票", list(res.get("stocks", {}).keys()) if res.get("stocks") else [])
                _show_news_list(res.get("stocks", {}).get(sym, []) if sym else [])
        else:
            # 旧结构（仅个股）
            st.caption("只检测到个股新闻（无宏观/政策）— 点击上方按钮重新抓取")
            sym = st.selectbox("选择股票", list(res.keys()) if res else [])
            _show_news_list(res.get(sym, []) if sym else [])


def _show_news_list(items: List[Dict[str, Any]]):
    if not items:
        st.info("暂无新闻")
        return
    for n in items:
        st.markdown(
            f"<div class='news-card'>"
            f"<a href='{n.get('link', '#')}' target='_blank' class='title'>{n.get('title','')}</a>"
            f"<div class='meta'>📌 {n.get('source','')} · {n.get('date','')}</div>"
            f"<div style='font-size:12px;color:var(--text-dim);margin-top:4px;'>{n.get('snippet','')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# 页面：数据诊断
# ---------------------------------------------------------------------------
def page_diagnostics():
    st.markdown('<div class="section-title"><span class="accent">⚙️</span>数据诊断 & Secret 检查</div>', unsafe_allow_html=True)

    rows = [
        ("FRED_API", FRED_KEY),
        ("DEEPSEEK_API_KEY", DEEPSEEK_KEY),
        ("SERPAPI", SERPAPI_KEY),
    ]
    for name, v in rows:
        ok = "✅" if v else "❌"
        st.markdown(f"- {ok} **{name}**: {'已配置 (' + str(len(v)) + ' 字符)' if v else '未配置'}")

    st.divider()
    st.markdown("#### 📁 data/ 目录文件状态")
    files = [
        "macro.csv", "stocks.csv", "sox.csv", "sp500.csv",
        "cards.json", "leverage_risk.json", "news.json",
        "report.md", "weekly_report.md", "morning_brief.md", "evening_recap.md",
    ]
    for f in files:
        p = DATA_DIR / f
        if p.exists():
            sz = p.stat().st_size
            mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            st.markdown(f"- ✅ **{f}** — {sz} bytes · updated {mtime}")
        else:
            st.markdown(f"- ❌ **{f}** — 缺失（运行 `python stock_dashboard.py` 生成）")

    st.divider()
    st.markdown("#### 🔥 一键测试 SerpApi")
    if st.button("测试 SerpApi 是否能拉到宏观新闻"):
        if not SERPAPI_KEY:
            st.error("SERPAPI 未配置")
        else:
            with st.spinner("测试中…"):
                test = U.fetch_macro_news(SERPAPI_KEY, top_n=3)
            if test:
                st.success(f"成功拉到 {len(test)} 条")
                for n in test:
                    st.markdown(f"- [{n.get('title','')}]({n.get('link','')})")
            else:
                st.error("未拉到任何结果 — 检查：1) Key 是否有效 2) 引擎是否支持 google_news 3) 额度")


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def _fetch_price_history(symbol: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """统一的 K 线获取：港股优先 akshare，否则 yfinance。"""
    if symbol.endswith(".HK"):
        try:
            import akshare as ak
            code = symbol.replace(".HK", "").zfill(5)
            raw = ak.stock_hk_daily(symbol=code, adjust="qfq")
            rename_map = {"日期": "Date", "开盘": "Open", "收盘": "Close", "最高": "High", "最低": "Low", "成交量": "Volume"}
            raw = raw.rename(columns={k: v for k, v in rename_map.items() if k in raw.columns})
            raw["Date"] = pd.to_datetime(raw["Date"])
            raw = raw.set_index("Date").sort_index()
            days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
            cutoff = raw.index.max() - pd.Timedelta(days=days_map.get(period, 90))
            raw = raw[raw.index >= cutoff]
            return raw[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce").dropna()
        except Exception:  # noqa: BLE001
            pass
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _draw_kline(hist: pd.DataFrame, sym: str, with_volume: bool = True):
    if with_volume:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(
            x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"],
            name=sym, increasing_line_color="#dc2626", decreasing_line_color="#16a34a",
        ), row=1, col=1)
        if len(hist) >= 20:
            ma20 = hist["Close"].rolling(20).mean()
            fig.add_trace(go.Scatter(x=hist.index, y=ma20, line=dict(color="#f59e0b", width=1.2), name="MA20"), row=1, col=1)
        if len(hist) >= 50:
            ma50 = hist["Close"].rolling(50).mean()
            fig.add_trace(go.Scatter(x=hist.index, y=ma50, line=dict(color="#2563eb", width=1.2), name="MA50"), row=1, col=1)
        colors = ["#dc2626" if c > 0 else "#16a34a" for c in hist["Close"].diff().fillna(0)]
        fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], name="Volume", marker_color=colors, opacity=0.5), row=2, col=1)
        fig.update_layout(
            height=420, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="white", xaxis_rangeslider_visible=False, showlegend=False,
            font=dict(color="#6b7280", size=11),
        )
        fig.update_xaxes(gridcolor="#f1f5f9")
        fig.update_yaxes(gridcolor="#f1f5f9")
    else:
        fig = go.Figure(go.Candlestick(
            x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"],
            name=sym, increasing_line_color="#dc2626", decreasing_line_color="#16a34a",
        ))
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
if page == "🏠 Dashboard":
    page_dashboard()
elif page == "🔍 个股深度分析":
    page_stock_deepdive()
elif page == "📊 跨资产对比":
    page_cross_asset()
elif page == "📰 新闻中心":
    page_news_center()
elif page == "⚙️ 数据诊断":
    page_diagnostics()

# ---------------------------------------------------------------------------
# 底部
# ---------------------------------------------------------------------------
st.markdown(
    f"""
<div style="text-align:center;color:var(--text-dim);font-size:11px;padding:30px 0 10px;border-top:1px solid var(--border);margin-top:30px;">
    ⚡ Investment Copilot · Author: Winnie Wang · {datetime.now().strftime('%Y-%m-%d %H:%M')} · 仅供研究参考，不构成投资建议
</div>
""",
    unsafe_allow_html=True,
)
