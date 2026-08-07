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

import bottom_signal as BS  # 市场底部判断模块
import utils as U  # 本地工具模块
import screener  # v3.0 选股三层架构：维科夫事件序列 + 多因子评分
from stock_dashboard import Config  # 自选股列表

# ---------------------------------------------------------------------------
# Streamlit 配置
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Winnie's Daily Stock Dashboard",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("copilot_app")

# ---------------------------------------------------------------------------
# Plotly 图表统一输出
# ---------------------------------------------------------------------------
def _plotly_chart(fig, use_container_width: bool = True, config=None) -> None:
    """统一输出 plotly 图。"""
    st.plotly_chart(fig, use_container_width=use_container_width, config=config)
    _plotly_chart(fig, use_container_width=use_container_width, config=config)


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

    /* v3.0 Bloomberg/DSA 风格：分组小标题 */
    .section-sub {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.6px;
        color: var(--text-dim);
        margin: 20px 0 8px 0;
        padding-bottom: 4px;
        border-bottom: 1px solid var(--border);
        text-transform: uppercase;
    }
    .section-sub .accent { color: var(--accent); }

    /* v3.0 圆环评分（DSA donut） */
    .donut {
        width: 74px;
        height: 74px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        position: relative;
        background: conic-gradient(var(--donut-color, #2563eb) calc(var(--donut-pct, 0) * 1%), #eef2f7 0);
    }
    .donut::before {
        content: "";
        position: absolute;
        inset: 9px;
        background: var(--card, #fff);
        border-radius: 50%;
    }
    .donut .val {
        position: relative;
        font-size: 17px;
        font-weight: 800;
        color: var(--text);
    }

    /* v3.0 左侧历史列表 / 右侧详情卡（Bloomberg 双栏） */
    .list-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 6px 8px;
        margin-bottom: 4px;
        font-size: 12px;
        cursor: pointer;
    }
    .list-card:hover { border-color: var(--accent); }
    .list-card.active { border-color: var(--accent); background: #eff6ff; }
    .detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 16px; font-size: 12px; }
    .detail-grid .k { color: var(--text-dim); }
    .detail-grid .v { font-weight: 700; text-align: right; }

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

    /* ===== 需求1：统一栅格与对齐 =====
       同一行（st.columns）内的卡片采用统一基线高度与一致字号，
       模块尺寸不随文字内容剧烈伸缩；标题/正文/数字字号全部固定。 */
    /* 卡片统一基线高度 + 内边距，避免内容多寡导致高度跳变 */
    .card {
        box-sizing: border-box;
        min-height: 150px;
        display: flex;
        flex-direction: column;
        font-size: 13px;
        line-height: 1.55;
        padding: 14px 16px;
    }
    /* 小信息卡（雷达/状态条等）单独基线，避免被大卡拉伸 */
    .card.compact { min-height: 96px; padding: 10px 12px; }
    .card > h4 { flex: 0 0 auto; margin: 0 0 8px 0; }
    .card .body-fill { flex: 1 1 auto; }
    /* 统一标题/正文字号，杜绝伸缩导致的视觉跳变 */
    .section-title { font-size: 16px; }
    .card h3 { font-size: 14px; color: var(--text); font-weight: 700; }
    .card h4 { font-size: 14px; color: var(--text-dim); font-weight: 700; }
    .stock-row { font-size: 13px; }
    /* 指标数字统一字号 */
    .card .big, .metric-big { font-size: 26px; font-weight: 800; color: var(--text); }
    /* 同行动态高度时，列内卡片底色统一，视觉对齐更整齐 */
    [data-testid="column"] > div { display: flex; }
    [data-testid="column"] > div > div { width: 100%; }

    /* ===== DSA 暗色玻璃态（明日观察位专用） =====
       在浅色页面中嵌入深色玻璃态组件，backdrop-filter 让卡片与下方背景融合；
       配色与参考 DSA 风格保持一致（青色强调 + 三色状态）。*/
    .dsa-glass {
        background: linear-gradient(135deg, rgba(15,23,42,0.92) 0%, rgba(30,41,59,0.92) 100%);
        border: 1px solid rgba(125,180,210,0.20);
        border-radius: 14px;
        padding: 16px 18px;
        color: #e2e8f0;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        box-shadow: 0 4px 18px rgba(2,6,23,0.18);
        margin-bottom: 12px;
    }
    .dsa-glass .title { font-size: 14px; font-weight: 700; color: #67e8f9; margin-bottom: 6px; letter-spacing: 0.3px; }
    .dsa-glass .sub { font-size: 11px; color: #94a3b8; }
    .dsa-glass .num { font-size: 24px; font-weight: 800; color: #f1f5f9; }
    .dsa-glass .acc { color: #67e8f9; }

    /* 状态 pill：ok / caution / danger / neutral */
    .tw-pill {
        display: inline-flex; align-items: center; gap: 5px;
        padding: 3px 10px; border-radius: 999px;
        font-size: 11px; font-weight: 700; letter-spacing: 0.3px;
    }
    .tw-pill::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
    .tw-pill.ok { background: rgba(52,211,153,0.15); color: #34d399; }
    .tw-pill.caution { background: rgba(251,191,36,0.15); color: #fbbf24; }
    .tw-pill.danger { background: rgba(248,113,113,0.15); color: #f87171; }
    .tw-pill.neutral { background: rgba(148,163,184,0.18); color: #94a3b8; }

    /* 明日观察位 4 维度网格 */
    .tw-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .tw-tile {
        background: rgba(15,23,42,0.55);
        border: 1px solid rgba(125,180,210,0.15);
        border-radius: 10px;
        padding: 10px 12px;
        color: #cbd5e1;
    }
    .tw-tile .head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
    .tw-tile .name { font-size: 12px; color: #94a3b8; font-weight: 600; }
    .tw-tile .state { font-size: 13px; font-weight: 700; margin: 2px 0 6px 0; }
    .tw-tile .ev { font-size: 11px; line-height: 1.5; color: #cbd5e1; opacity: 0.88; }
    .tw-tile .meta { font-size: 10px; color: #64748b; margin-top: 4px; }
    /* 状态条强度可视化 */
    .tw-bar { height: 4px; background: rgba(148,163,184,0.18); border-radius: 2px; margin-top: 6px; }
    .tw-bar .fill { height: 100%; border-radius: 2px; }
    .tw-bar .fill.ok { background: #34d399; }
    .tw-bar .fill.caution { background: #fbbf24; }
    .tw-bar .fill.danger { background: #f87171; }
    .tw-bar .fill.neutral { background: #94a3b8; }

    /* 整体策略卡 */
    .tw-overall {
        background: linear-gradient(135deg, rgba(8,47,73,0.85) 0%, rgba(13,71,161,0.85) 100%);
        border: 1px solid rgba(125,180,210,0.25);
        border-radius: 12px;
        padding: 14px 16px;
        color: #f0f9ff;
        margin-bottom: 10px;
    }
    .tw-overall .head { display: flex; align-items: center; justify-content: space-between; }
    .tw-overall .action { font-size: 18px; font-weight: 800; }
    .tw-overall .summary { font-size: 12px; color: #cbd5e1; margin-top: 6px; line-height: 1.6; }
    .tw-overall .ok { color: #34d399; }
    .tw-overall .caution { color: #fbbf24; }
    .tw-overall .danger { color: #f87171; }

    /* 风险/数字高亮 */
    .tw-num-up { color: #f87171; font-weight: 700; }   /* 涨（A股红）*/
    .tw-num-down { color: #34d399; font-weight: 700; } /* 跌（A股绿）*/

    /* ===== 分层架构视觉：结论区(强调) / 证据层(安静) =====
       结论区用浅蓝渐变 + 左侧 accent 边框，强调「先看这里」；
       证据层 Tab 内用既有 .card/.section-sub 安静风格，弱化存在感。 */
    .layer-badge {
        display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: 0.5px;
        color: var(--text-dim); background: #eef2f7; border: 1px solid var(--border);
        padding: 3px 10px; border-radius: 999px; margin: 2px 0 10px 0;
    }
    .layer-badge.concl { color: #0369a1; background: #e0f2fe; border-color: #bae6fd; }
    .conclusion-zone {
        background: linear-gradient(180deg, rgba(37,99,235,0.05) 0%, rgba(255,255,255,0) 100%);
        border: 1px solid rgba(37,99,235,0.18);
        border-left: 4px solid var(--accent);
        border-radius: 14px; padding: 14px 16px; margin-bottom: 18px;
    }
    .conclusion-zone .section-title { margin-top: 6px; }

    /* 实时速览条（结论区集成实时行情） */
    .rt-strip {
        display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
        background: rgba(103,232,249,0.10); border: 1px solid rgba(103,232,249,0.30);
        border-radius: 10px; padding: 8px 14px; margin-bottom: 6px;
    }
    .rt-strip .rt-label { font-size: 13px; font-weight: 700; color: #0369a1; white-space: nowrap; }
    .rt-strip .rt-time { font-size: 11px; color: var(--text-dim); white-space: nowrap; }
    .rt-strip .rt-chips { display: flex; gap: 12px; flex-wrap: wrap; }
    .rt-strip .rt-chip { font-size: 12px; color: var(--text); white-space: nowrap; }
    .rt-strip .rt-na { color: var(--text-dim); }

    @media (max-width: 700px) { .tw-grid { grid-template-columns: 1fr; } }
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 环境变量
# ---------------------------------------------------------------------------
def _get_secret(name: str, default: str = "") -> str:
    """双路读 secret（os.environ + st.secrets）。"""
    val = os.environ.get(name, "")
    if val:
        return val
    if hasattr(st, "secrets"):
        try:
            val = st.secrets.get(name, "")
            if val:
                return str(val)
        except Exception:
            pass
    return default


SERPAPI_KEY = _get_secret("SERPAPI")
DEEPSEEK_KEY = _get_secret("DEEPSEEK_API_KEY")
FRED_KEY = _get_secret("FRED_API")


def _check_lib(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False

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
        "extra_indicators": DATA_DIR / "extra_indicators.json",
        "options_pcr": DATA_DIR / "options_pcr.json",
        "predictions": DATA_DIR / "predictions.json",
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
EXTRA_DATA = DATA.get("extra_indicators") or {}
PCR_DATA = DATA.get("options_pcr") or {}
PREDICTIONS_DATA = DATA.get("predictions") or {}
CARDS_LIST = CARDS_DATA.get("stocks", []) if isinstance(CARDS_DATA, dict) else []
CARDS_MAP = {c.get("symbol"): c for c in CARDS_LIST} if isinstance(CARDS_LIST, list) else {}
LEV_MAP = (DATA.get("leverage") or {}).get("stocks", {}) if isinstance(DATA.get("leverage"), dict) else {}


# ---------------------------------------------------------------------------
# API key 集合（v2.1：多源支持）
# ---------------------------------------------------------------------------
FINNHUB_KEY = _get_secret("FINNHUB_API")
NEWSAPI_KEY = _get_secret("NEWSAPI_KEY")


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
    st.markdown("### 🤖 Investment Copilot")
    st.caption("by Winnie")
    st.divider()
    page = st.radio(
        "导航",
        ["🏠 Dashboard", "🔍 个股深度分析", "📊 跨资产对比", "📖 使用说明"],
        label_visibility="collapsed",
        key="nav_radio",
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
    st.divider()
    if st.button("⚙️ 数据诊断", use_container_width=True):
        st.session_state["page_override"] = "⚙️ 数据诊断"
        st.rerun()

# 诊断模块从侧边栏底部按钮进入（不占用主导航）
if st.session_state.get("page_override"):
    page = st.session_state["page_override"]
    st.session_state["page_override"] = None


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
# ---------------------------------------------------------------------------
# v2.5 智能荐股辅助
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _cached_metrics(symbols_tuple):
    return U.fetch_all_metrics(list(symbols_tuple))


def _bias_emoji(b):
    return "🐂" if b == "看多" else ("🐻" if b == "看空" else "➡️")


def _rec_md(r, sub):
    chg = r.get("chgPct") or 0
    c = "#dc2626" if chg > 0 else ("#16a34a" if chg < 0 else "var(--text-dim)")
    nm = U.STOCK_NAMES.get(r["symbol"], "")
    return (f'<div style="padding:5px 0;border-bottom:1px solid var(--border);font-size:12px;">'
            f'<b>{r["symbol"]}</b> <span style="color:var(--text-dim);font-size:10px;">{nm}</span><br>'
            f'<span style="color:{c};font-weight:600;">{chg:+.2f}%</span> '
            f'<span style="color:var(--text-dim);font-size:10px;">{sub}</span></div>')


def _donut(score: float, size: int = 74, label: str = "评分") -> str:
    """DSA 圆环评分：score 0-100。返回内联 HTML（用于 st.markdown）。"""
    try:
        s = float(score)
    except (TypeError, ValueError):
        s = 0.0
    s = min(100.0, max(0.0, s))
    color = "#dc2626" if s >= 60 else ("#f59e0b" if s >= 40 else "#16a34a")
    return (
        f'<div class="donut" style="--donut-pct:{s:.1f};--donut-color:{color};width:{size}px;height:{size}px;">'
        f'<span class="val" style="font-size:{max(12, size//4)}px;">{s:.0f}</span></div>'
        f'<div style="font-size:10px;color:var(--text-dim);text-align:center;margin-top:4px;">{label}</div>'
    )


# ---------------------------------------------------------------------------
# 明日观察位（v3.2：同花顺 Financial-API + 规则引擎 + DeepSeek 研判）
# ---------------------------------------------------------------------------

def _tw_pill(status: str) -> str:
    """根据 status 返回对应的 .tw-pill HTML 片段。"""
    s = (status or "neutral").lower()
    label = {"ok": "正向", "caution": "关注", "danger": "警示", "neutral": "中性"}.get(s, "中性")
    return f'<span class="tw-pill {s}">{label}</span>'


def _tw_evidence_html(text: str) -> str:
    """统一 evidence 文本渲染（控制行高 + 字号）。"""
    safe = (text or "").replace("<", "&lt;").replace(">", "&gt;")
    return f'<div class="ev">{safe}</div>'


def _render_tomorrow_watch(symbol: str) -> None:
    """
    渲染「明日观察位」模块（DSA 暗色玻璃风格）。
    调用 U.compute_tomorrow_watch 拉取 K 线 + 主力资金 → 渲染四维度。
    提供 AI 研判按钮（DeepSeek / OpenRouter），无 LLM Key 时降级为规则 summary。
    """
    st.markdown(
        '<div class="section-title"><span class="accent">🔮</span>明日观察位 · '
        '<span style="font-size:11px;color:var(--text-dim);font-weight:500;">'
        '规则引擎 + DeepSeek 研判</span></div>',
        unsafe_allow_html=True,
    )

    # 1) 拉取数据：A 股自动尝试主力资金流；其他市场传 None
    capital_flow = None
    if symbol.endswith((".SS", ".SZ")):
        try:
            capital_flow = U.fetch_capital_flow_eastmoney(symbol)
        except Exception:  # noqa: BLE001
            capital_flow = None
    try:
        with st.spinner(f"计算 {symbol} 明日观察位…"):
            watch = U.compute_tomorrow_watch(symbol, capital_flow=capital_flow)
    except Exception as e:  # noqa: BLE001
        st.error(f"明日观察位计算失败：{e}")
        return

    if not watch.get("ok"):
        st.warning(f"⚠️ {watch.get('reason', '数据不足')}")
        return

    raw = watch.get("raw", {})

    # 2) 整体策略卡（顶部强调）
    ov = watch.get("overall", {})
    overall_html = (
        f'<div class="tw-overall">'
        f'<div class="head">'
        f'<div>'
        f'<div style="font-size:11px;color:#7dd3fc;letter-spacing:0.5px;">整体策略 · OVERALL</div>'
        f'<div class="action {ov.get("status","ok")}">{ov.get("action","—")}</div>'
        f'</div>'
        f'<div style="text-align:right;font-size:11px;color:#94a3b8;">'
        f'评分 {ov.get("score",0)}/100<br>信心 {watch.get("support",{}).get("score",50)}%'
        f'</div>'
        f'</div>'
        f'<div class="summary">{ov.get("summary","")}</div>'
        f'<div style="font-size:10px;color:#64748b;margin-top:8px;">'
        f'现价 {raw.get("current_price")} · 日 {raw.get("chg_pct",0):+.2f}% · '
        f'MA5 {raw.get("ma5")} · MA20 {raw.get("ma20")} · RSI {raw.get("rsi")} · '
        f'量比 {raw.get("vol_ratio5")}x</div>'
        f'</div>'
    )
    st.markdown(overall_html, unsafe_allow_html=True)

    # 3) 四维度 2x2 网格
    dims = [
        ("support",  "🛡️ 支撑观察", f'近期低点 {raw.get("support_level")} · 距离 {watch["support"].get("distance_pct","—")}%'),
        ("breakout", "📈 放量信号", f'量比5d {raw.get("vol_ratio5")}x · 量比20d {raw.get("vol_ratio20")}x'),
        ("capital",  "💰 主力资金", ("有数据" if watch["capital"].get("has_data") else "暂无数据")),
        ("risk",     "⚠️ 风险预警", f'ATR% {raw.get("atr_pct")}% · MACD柱 {raw.get("macd_hist")}'),
    ]
    tiles = []
    for key, name, meta in dims:
        d = watch.get(key, {})
        score = d.get("score", 50)
        tiles.append(
            f'<div class="tw-tile">'
            f'<div class="head">'
            f'<div class="name">{name}</div>'
            f'{_tw_pill(d.get("status","neutral"))}'
            f'</div>'
            f'<div class="state" style="color:{"#34d399" if d.get("status")=="ok" else "#fbbf24" if d.get("status")=="caution" else "#f87171" if d.get("status")=="danger" else "#94a3b8"};">{d.get("state","—")}</div>'
            f'{_tw_evidence_html(d.get("evidence",""))}'
            f'<div class="meta">{meta}</div>'
            f'<div class="tw-bar"><div class="fill {d.get("status","neutral")}" style="width:{max(0,min(100,score))}%;"></div></div>'
            f'</div>'
        )
    grid_html = '<div class="dsa-glass"><div class="title">四维度量化研判</div><div class="tw-grid">' + "".join(tiles) + '</div></div>'
    st.markdown(grid_html, unsafe_allow_html=True)

    # 4) AI 研判按钮（DeepSeek / OpenRouter）
    btn_key = f"tw_ai_{symbol}"
    ai_state_key = f"tw_ai_text_{symbol}"
    cols = st.columns([1, 1, 2])
    with cols[0]:
        if st.button("🤖 生成 AI 研判", key=btn_key, use_container_width=True,
                     help="调用 DeepSeek / OpenRouter 基于上述结构化结论生成自然语言研判"):
            with st.spinner("AI 研判中…"):
                text = U.narrate_tomorrow_watch(watch)
            st.session_state[ai_state_key] = text or watch["overall"]["summary"]
    with cols[1]:
        if st.button("🗑️ 清空", key=f"tw_ai_clr_{symbol}", use_container_width=True):
            st.session_state.pop(ai_state_key, None)
            st.rerun()
    with cols[2]:
        st.caption("提示：未配置 DEEPSEEK_API_KEY 时降级展示规则 summary。")

    if st.session_state.get(ai_state_key):
        st.markdown(
            f'<div class="dsa-glass" style="background:linear-gradient(135deg, rgba(8,47,73,0.85) 0%, rgba(15,23,42,0.92) 100%);">'
            f'<div class="title">🧠 AI 研判</div>'
            f'<div style="font-size:13px;line-height:1.85;color:#e2e8f0;">{st.session_state[ai_state_key]}</div>'
            f'<div style="font-size:10px;color:#64748b;margin-top:8px;text-align:right;">非投资建议 · 仅供研究参考</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ===========================================================================
# 新版 Dashboard：第0层·结论区（组合策略 + 今日重点/Top机会/Top风险 + 冲突说明）
#            + 第1层·证据层（4 主题 Tab：宏观 / 结构 / 事件 / 研判）
# 由 app.py 的 page_dashboard 重构而来（分层架构，避免扁平化）。
# ===========================================================================


def _rt_dash_strip() -> None:
    """结论区实时速览：默认观察列表的实时价（自动刷新，多源降级）。

    通过 st.fragment(run_every=30) 注册，仅本 fragment 重跑，页面控件不闪烁。
    集成实时行情到 Dashboard（需求3：不再单独模块），失败容错显示 —。
    """
    from datetime import datetime, timezone, timedelta
    syms = list(_RT_DEFAULT_WATCHLIST)
    try:
        snaps = U.fetch_realtime_snapshots(syms)
    except Exception:  # noqa: BLE001
        snaps = {}
    chips = []
    for s in syms:
        q = snaps.get(s, {})
        if not q.get("ok"):
            chips.append(f"<span class='rt-chip'><b>{s}</b> <span class='rt-na'>—</span></span>")
            continue
        c = color_for_change(safe_float(q.get("pct")))
        chips.append(
            f"<span class='rt-chip'>"
            f"<b>{s}</b> "
            f"<span style='color:{c};font-weight:700;'>{safe_float(q.get('last')):.2f} {fmt_pct(safe_float(q.get('pct')))}</span>"
            f"</span>"
        )
    now = datetime.now(timezone(timedelta(hours=8)))
    st.markdown(
        f"<div class='rt-strip'>"
        f"<span class='rt-label'>📡 实时速览</span>"
        f"<span class='rt-time'>{now:%H:%M:%S} 北京时间 · 腾讯/东财/新浪 自动降级 · ⚠️免费源延迟约15-20分钟</span>"
        f"<div class='rt-chips'>{''.join(chips)}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_conclusion_zone() -> None:
    """第0层·结论区：组合策略 + 今日重点/Top机会/Top风险 + 冲突说明（唯一默认可见）。"""
    try:
        st.markdown('<div class="layer-badge concl">🎯 第 0 层 · 结论区 — 先看这里，再下钻证据</div>', unsafe_allow_html=True)
        st.markdown('<div class="conclusion-zone">', unsafe_allow_html=True)

        # ===== 组合策略 Portfolio Context =====
        dominance = U.compute_portfolio_dominance(STOCKS_DF)
        dominance_emoji = dominance.get("emoji", "💤")
        dominance_label = dominance.get("dominance_label", "空仓")
        if dominance_label == "多头占优":
            dominance_color = "#dc2626"
        elif dominance_label == "空头占优":
            dominance_color = "#16a34a"
        elif dominance_label == "空仓":
            dominance_color = "#9ca3af"
        else:
            dominance_color = "#f59e0b"
        total_n = dominance.get("total", 0)
        long_n = dominance.get("long_count", 0)
        short_n = dominance.get("short_count", 0)
        flat_n = dominance.get("flat_count", 0)
        avg_chg = dominance.get("avg_chg", 0.0)
        long_pct = dominance.get("long_pct", 0.0)
        short_pct = dominance.get("short_pct", 0.0)
        flat_pct = dominance.get("flat_pct", 0.0)
        etf_n = dominance.get("etf_count", 0)
        err = dominance.get("error")
        last_update = "—"
        if STOCKS_DF is not None and not STOCKS_DF.empty and "日期" in STOCKS_DF.columns:
            try:
                last_update = str(STOCKS_DF["日期"].max())[:10]
            except Exception:
                pass

        if err is None:
            # 组合风险指标（Beta / HHI / MaxDD）——尽力而为，失败显示 —
            _beta_txt, _beta_color = "—", "var(--text-dim)"
            _hhi_txt, _hhi_color = "—", "var(--text-dim)"
            _top_sector, _maxdd_txt = "—", "—"
            try:
                _risk = U.compute_portfolio_risk_metrics(STOCKS_DF)
                if _risk.get("beta") is not None:
                    _b = _risk["beta"]
                    _beta_txt = f"{_b:.2f}"
                    _beta_color = "#dc2626" if _b >= 1.2 else ("#f59e0b" if _b >= 0.9 else "#16a34a")
                if _risk.get("hhi") is not None:
                    _h = _risk["hhi"]
                    _hhi_txt = f"{_h:.3f}"
                    _hhi_color = "#dc2626" if _h > 0.25 else ("#f59e0b" if _h > 0.15 else "#16a34a")
                if _risk.get("sector_weights"):
                    _top_sector = next(iter(_risk["sector_weights"]))
                if _risk.get("max_dd") is not None:
                    _maxdd_txt = f"{_risk['max_dd']:.1f}%"
            except Exception:  # noqa: BLE001
                pass
            st.markdown(
                f"""
    <div class="card" style="background:linear-gradient(135deg,#fff 0%,#f8fafc 100%);">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
        <div>
          <h4 style="margin:0 0 4px 0;">🎯 组合策略 Portfolio Context</h4>
          <div style="font-size:11px;color:var(--text-dim);">{total_n}/16 策略持仓 · 更新 {last_update}</div>
        </div>
        <div style="display:flex;gap:12px;align-items:center;">
          <div style="text-align:center;">
            <div style="font-size:10px;color:var(--text-dim);">持仓策略</div>
            <div style="font-size:18px;font-weight:800;color:var(--text);">{total_n}</div>
          </div>
          <div style="text-align:center;">
            <div style="font-size:10px;color:var(--text-dim);">活跃均值</div>
            <div style="font-size:18px;font-weight:800;color:{dominance_color};">{avg_chg:+.2f}%</div>
          </div>
          <div style="text-align:center;">
            <div style="font-size:10px;color:var(--text-dim);">ETF 持仓</div>
            <div style="font-size:18px;font-weight:800;color:var(--text);">{etf_n}</div>
          </div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:24px;margin-top:14px;flex-wrap:wrap;">
        <div style="display:flex;align-items:center;gap:8px;min-width:200px;">
          <span style="font-size:48px;">{dominance_emoji}</span>
          <div>
            <div style="font-size:28px;font-weight:800;color:{dominance_color};">{dominance_label}</div>
            <div style="font-size:11px;color:var(--text-dim);">{U.emoji_for_sentiment(min(100, max(0, 50 + avg_chg*10)))} 风险偏好: {min(100, max(0, 50 + avg_chg*10)):.0f}/100</div>
          </div>
        </div>
        <div style="display:flex;gap:8px;flex:1;flex-wrap:wrap;">
          <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:10px 14px;flex:1;min-width:120px;">
            <div style="font-size:11px;color:#dc2626;font-weight:600;">● 多头</div>
            <div style="font-size:22px;font-weight:800;color:#dc2626;">{long_n}</div>
            <div style="font-size:11px;color:var(--text-dim);">{long_pct:.1f}%</div>
          </div>
          <div style="background:#dcfce7;border:1px solid #bbf7d0;border-radius:8px;padding:10px 14px;flex:1;min-width:120px;">
            <div style="font-size:11px;color:#16a34a;font-weight:600;">● 空头</div>
            <div style="font-size:22px;font-weight:800;color:#16a34a;">{short_n}</div>
            <div style="font-size:11px;color:var(--text-dim);">{short_pct:.1f}%</div>
          </div>
          <div style="background:#f1f5f9;border:1px solid #cbd5e1;border-radius:8px;padding:10px 14px;flex:1;min-width:120px;">
            <div style="font-size:11px;color:#475569;font-weight:600;">● 空仓</div>
            <div style="font-size:22px;font-weight:800;color:#475569;">{flat_n}</div>
            <div style="font-size:11px;color:var(--text-dim);">{flat_pct:.1f}%</div>
          </div>
        </div>
      </div>
      <div style="display:flex;height:10px;border-radius:5px;overflow:hidden;margin-top:12px;background:#e5e7eb;">
        <div style="width:{long_pct}%;background:#dc2626;transition:width 0.4s;"></div>
        <div style="width:{short_pct}%;background:#16a34a;transition:width 0.4s;"></div>
        <div style="width:{flat_pct}%;background:#94a3b8;transition:width 0.4s;"></div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px;font-size:12px;">
        <div style="background:var(--bg2);border-radius:8px;padding:8px 12px;text-align:center;">
          <div style="color:var(--text-dim);font-size:10px;">组合 Beta (vs S&P500)</div>
          <div style="font-size:16px;font-weight:800;color:{_beta_color};">{_beta_txt}</div>
          <div style="font-size:10px;color:var(--text-dim);">市值加权 · 1Y</div>
        </div>
        <div style="background:var(--bg2);border-radius:8px;padding:8px 12px;text-align:center;">
          <div style="color:var(--text-dim);font-size:10px;">行业集中度 HHI</div>
          <div style="font-size:16px;font-weight:800;color:{_hhi_color};">{_hhi_txt}</div>
          <div style="font-size:10px;color:var(--text-dim);">Top: {_top_sector}</div>
        </div>
        <div style="background:var(--bg2);border-radius:8px;padding:8px 12px;text-align:center;">
          <div style="color:var(--text-dim);font-size:10px;">组合最大回撤 MaxDD</div>
          <div style="font-size:16px;font-weight:800;color:#16a34a;">{_maxdd_txt}</div>
          <div style="font-size:10px;color:var(--text-dim);">等权净值 · 1Y</div>
        </div>
      </div>
    </div>
    """,
                unsafe_allow_html=True,
            )
        else:
            st.warning(f"组合策略卡数据缺失: {err}（请先运行 `python stock_dashboard.py` 生成 data/stocks.csv）")

        # ===== 今日重点 + Top 机会 + Top 风险（共享 top_syms / risk_syms 供冲突说明）=====
        st.markdown('<div class="section-title">🎯 今日重点 & 自选扫描</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1.2, 1, 1])

        # 今日重点：基于自选股 + 宏观 + 财报/事件
        with c1:
            st.markdown('<div class="card"><h4>📌 今日重点</h4>', unsafe_allow_html=True)
            focuses = []
            if STOCKS_DF is not None and not STOCKS_DF.empty:
                if "涨跌幅" in STOCKS_DF.columns:
                    top_up = STOCKS_DF.nlargest(1, "涨跌幅").iloc[0]
                    focuses.append(
                        f"🚀 <b>{top_up['symbol']}</b> <span style='color:var(--text-dim);font-size:11px;'>{U.STOCK_NAMES.get(top_up['symbol'],'')}</span> 今日涨 {safe_float(top_up.get('涨跌幅')):.2f}%，关注能否突破 / 短线见顶"
                    )
                    top_dn = STOCKS_DF.nsmallest(1, "涨跌幅").iloc[0]
                    focuses.append(
                        f"🔻 <b>{top_dn['symbol']}</b> <span style='color:var(--text-dim);font-size:11px;'>{U.STOCK_NAMES.get(top_dn['symbol'],'')}</span> 今日跌 {safe_float(top_dn.get('涨跌幅')):.2f}%，关注是否到支撑 / 风险扩大"
                    )
                if "RSI_14" in STOCKS_DF.columns:
                    oversold = STOCKS_DF[STOCKS_DF["RSI_14"] < 30]
                    if not oversold.empty:
                        focuses.append(
                            f"🟢 RSI 超卖: {', '.join(oversold['symbol'].head(3).tolist())}，可能反弹"
                        )
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
        top_syms = []
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
                    top_syms.append(sym)
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
        risk_syms = []
        with c3:
            st.markdown('<div class="card"><h4>🔴 Top 风险</h4>', unsafe_allow_html=True)
            risks = []
            if STOCKS_DF is not None and not STOCKS_DF.empty:
                tmp = STOCKS_DF.copy()
                if "RSI_14" in tmp.columns:
                    for _, r in tmp[tmp["RSI_14"] > 70].head(3).iterrows():
                        risks.append((r["symbol"], f"RSI {safe_float(r.get('RSI_14')):.0f} 超买", "🟠"))
                if "涨跌幅" in tmp.columns:
                    for _, r in tmp.nsmallest(3, "涨跌幅").iterrows():
                        if not any(s == r["symbol"] for s, *_ in risks):
                            risks.append((r["symbol"], f"今日 {fmt_pct(safe_float(r.get('涨跌幅')))}", "🔴"))
                for sym, lev in (LEV_MAP or {}).items():
                    if lev.get("综合风险等级") == "高" and not any(s == sym for s, *_ in risks):
                        risks.append((sym, "杠杆高危", "⚠️"))
                        risk_syms.append(sym)
            for sym, msg, ic in risks[:6]:
                st.markdown(
                    f"<div class='stock-row' style='display:flex;justify-content:space-between;align-items:center;'><span><b>{sym}</b> · {msg}</span><span>{ic}</span></div>",
                    unsafe_allow_html=True,
                )
            if not risks:
                st.caption("当前无显著风险信号（无 RSI>70 / 跌幅领先 / 杠杆高危）")
            st.markdown("</div>", unsafe_allow_html=True)

        # 冲突说明（Top 机会 与 Top 风险 同现）
        _conflict = sorted(set(top_syms) & set(risk_syms))
        if _conflict:
            _ex = "、".join(_conflict)
            st.markdown(
                f'<div class="card" style="min-height:auto;background:#fffbeb;border-left:4px solid #f59e0b;">'
                f'<div style="font-size:12.5px;line-height:1.7;color:#92400e;">'
                f'⚠️ <b>冲突说明：{_ex} 同时出现在「Top 机会」与「Top 风险」。</b><br>'
                f'这并非系统矛盾，而是「<b>机会与风险并存</b>」的典型信号：'
                f'<b>机会面</b>来自其高 AI 评分 / 强势趋势 / 基本面（趋势与结构占优）；'
                f'<b>风险面</b>来自短线超买(RSI&gt;70)、近期回撤或高杠杆（追高回吐 / 强平风险）。'
                f'<br>操作含义：宜用<b>分批介入 + R 倍数纪律</b>（见个股深度分析·交易策略），而非一次性追高；'
                f'机会决定「买不买」，风险决定「买多少、止损在哪」。'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)  # close .conclusion-zone

        # 实时速览（集成实时行情，自动刷新）
        try:
            st.fragment(_rt_dash_strip, run_every=30)()
        except Exception as _e:  # noqa: BLE001
            logger.exception("实时速览 fragment 失败: %s", _e)
            st.caption(f"📡 实时速览暂不可用: {_e}")


    except Exception as _e:  # noqa: BLE001
        logger.exception("结论区渲染异常: %s", _e)
        st.error(f"⚠️ 结论区加载失败: {_e}")
        st.info("提示：请确认 data/stocks.csv 已存在且包含数据。")
def _render_macro_tab() -> None:
    """第1层·证据层 — 宏观：情绪与波动率 + 全球主指数 + 利率与杠杆 + 风险雷达 + 宏观解读。"""
    # ===== 市场情绪与波动率 =====
    st.markdown('<div class="section-sub"><span class="accent">▍</span>市场情绪与波动率</div>', unsafe_allow_html=True)
    fg = U.calculate_fear_greed()
    macro_row = MACRO_DF.iloc[0] if MACRO_DF is not None and not MACRO_DF.empty else None
    sox_row = SOX_DF.iloc[0] if SOX_DF is not None and not SOX_DF.empty else None
    sentiment_score = safe_float(macro_row.get("情绪指数"), 50.0) if macro_row is not None else 50.0
    sentiment_label = str(macro_row.get("情绪标签", "中性")) if macro_row is not None else "中性"

    fg_label = fg.get("label", "中性")
    fg_score = fg.get("score", 50.0)
    fg_color = {"极度贪婪": "#16a34a", "贪婪": "#84cc16", "中性": "#fbbf24", "恐惧": "#f97316", "极度恐惧": "#dc2626"}.get(fg_label, "#9ca3af")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(
            f"""
<div class="card" style="text-align:center;">
    <h4>{U.emoji_for_sentiment(fg_score)} Fear &amp; Greed</h4>
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
    <h4>{U.emoji_for_sentiment(sentiment_score)} 自建情绪指数</h4>
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
        vix_warn = "⚠️ 警戒" if vix > 25 else ("正常" if vix > 0 else "—")
        st.markdown(
            f'<div class="card" style="text-align:center;"><h4>{U.emoji_for_panic(vix)} VIX <span style="font-size:9px;color:var(--text-dim);">隐含波动率</span></h4><div class="big" style="color:{vix_color};">{vix:.2f}</div><div style="font-size:11px;color:var(--text-dim);margin-top:4px;">{vix_warn}</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        vxn_quote = U.fetch_index_quote("^VXN")
        vxn = vxn_quote.get("last", 0.0)
        vxn_color = "#dc2626" if vxn > 30 else ("#f97316" if vxn > 22 else "#16a34a")
        vxn_warn = "😱 警戒" if vxn > 30 else ("⚠️ 偏高" if vxn > 22 else ("正常" if vxn > 0 else "—"))
        st.markdown(
            f'<div class="card" style="text-align:center;"><h4>{U.emoji_for_panic(vxn)} VXN <span style="font-size:9px;color:var(--text-dim);">纳指IV</span></h4><div class="big" style="color:{vxn_color};">{vxn:.2f}</div><div style="font-size:11px;color:var(--text-dim);margin-top:4px;">{vxn_warn}</div></div>',
            unsafe_allow_html=True,
        )
    with c5:
        sox_px = safe_float(sox_row.get("最新价"), 0.0) if sox_row is not None else 0.0
        sox_dd = safe_float(sox_row.get("回撤"), 0.0) if sox_row is not None else 0.0
        sox_bear = bool(sox_row.get("技术性熊市", False)) if sox_row is not None else False
        st.markdown(
            f'<div class="card" style="text-align:center;"><h4>💾 SOX 半导体</h4><div class="big">{sox_px:.0f}</div><div style="font-size:11px;color:var(--text-dim);margin-top:4px;">回撤 {sox_dd:.1f}% {"🐻" if sox_bear else ""}</div></div>',
            unsafe_allow_html=True,
        )
    with c6:
        tnx = safe_float(macro_row.get("10年期美债收益率"), 0.0) if macro_row is not None else 0.0
        st.markdown(
            f'<div class="card" style="text-align:center;"><h4>🏛️ 10Y 美债</h4><div class="big">{tnx:.2f}%</div><div style="font-size:11px;color:var(--text-dim);margin-top:4px;">利率风向标</div></div>',
            unsafe_allow_html=True,
        )

    # ===== 全球主要指数 =====
    st.markdown('<div class="section-sub"><span class="accent">▍</span>全球主要指数</div>', unsafe_allow_html=True)
    world_idx = [
        ("000001.SS", "上证指数", "🇨🇳"),
        ("^HSI", "恒生指数", "🇭🇰"),
        ("^GSPC", "标普 500", "🇺🇸"),
        ("^NDX", "纳指 100", "🇺🇸"),
        ("^DJI", "道指", "🇺🇸"),
    ]
    wc = st.columns(5)
    for col, (sym, name, flag) in zip(wc, world_idx):
        with col:
            q = U.fetch_index_quote(sym)
            if "error" in q or q.get("last", 0) == 0:
                st.markdown(
                    f'<div class="card" style="text-align:center;">'
                    f'<h4>{flag} {name}</h4>'
                    f'<div style="font-size:11px;color:var(--text-dim);margin:14px 0;">—</div>'
                    f'<div style="font-size:10px;color:var(--text-dim);">{sym}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                px = q["last"]
                chg = q["chg_pct"]
                color = "#dc2626" if chg > 0 else ("#16a34a" if chg < 0 else "var(--text-dim)")
                regime_emoji = ""
                try:
                    h = yf.download(sym, period="1mo", progress=False, auto_adjust=True)
                    if not h.empty and len(h) >= 5:
                        if isinstance(h.columns, pd.MultiIndex):
                            h.columns = h.columns.get_level_values(0)
                        chg30 = (h["Close"].iloc[-1] / h["Close"].iloc[0] - 1) * 100
                        regime_emoji = U.emoji_for_market_regime(chg30)
                except Exception:
                    pass
                st.markdown(
                    f'<div class="card" style="text-align:center;">'
                    f'<h4>{flag} {name} {regime_emoji}</h4>'
                    f'<div class="big" style="color:{color};">{px:,.2f}</div>'
                    f'<div style="font-size:13px;font-weight:700;color:{color};margin-top:4px;">{chg:+.2f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ===== 利率与杠杆 =====
    st.markdown('<div class="section-sub"><span class="accent">▍</span>利率与杠杆（FRED）</div>', unsafe_allow_html=True)
    y2c = EXTRA_DATA.get("2y_scorecard") or {}
    debt = EXTRA_DATA.get("us_debt") or {}
    margin = EXTRA_DATA.get("margin_debt") or {}
    nfci = EXTRA_DATA.get("nfci_leverage") or {}

    if not y2c or y2c.get("y2") is None:
        with st.spinner("实时拉取 2Y…"):
            y2c = U.fetch_2y_scorecard()
    if not debt or debt.get("value_trillion") is None:
        with st.spinner("实时拉取 US Debt…"):
            debt = U.fetch_us_debt()
    if not margin or margin.get("value_billion") is None:
        with st.spinner("实时拉取 Margin Debt…"):
            margin = U.fetch_margin_debt()
    if not nfci or nfci.get("value") is None:
        with st.spinner("实时拉取 NFCI…"):
            nfci = U.fetch_nfci_leverage()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        y2 = y2c.get("y2")
        spread = y2c.get("spread_bps")
        signal = y2c.get("signal", "—")
        if y2 is not None and spread is not None:
            spread_color = "#dc2626" if spread < 0 else ("#16a34a" if spread > 75 else "#f59e0b")
            real_y2 = y2c.get("real_y2")
            real_y2_html = (
                f'<div style="font-size:12px;margin-top:4px;">实际利率(2Y): <b>{real_y2:.2f}%</b> '
                f'<span style="font-size:9px;color:var(--text-dim);">≈DGS2−T5YIE</span></div>'
                if real_y2 is not None else
                '<div style="font-size:9px;color:var(--text-dim);margin-top:4px;">实际利率不可用（需 FRED DGS2+T5YIE）</div>'
            )
            st.markdown(
                f'<div class="card" style="text-align:center;">'
                f'<h4>📊 2-Year Scorecard</h4>'
                f'<div style="font-size:13px;color:var(--text-dim);">2Y: <b style="font-size:18px;color:var(--text);">{y2:.2f}%</b> · 10Y: <b style="font-size:18px;color:var(--text);">{y2c.get("y10", 0):.2f}%</b></div>'
                f'<div style="font-size:22px;font-weight:800;color:{spread_color};margin-top:6px;">{spread:.0f} bps</div>'
                f'<div style="font-size:11px;margin-top:4px;">{signal}</div>'
                f'{real_y2_html}'
                f'<div style="font-size:9px;color:var(--text-dim);margin-top:2px;">5d 变化: {y2c.get("spread_5d_chg", "—")} bps · {y2c.get("asof","—")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="card" style="text-align:center;">'
                f'<h4>📊 2-Year Scorecard</h4>'
                f'<div style="font-size:11px;color:var(--text-dim);margin:20px 0;">📡 数据待接入</div>'
                f'<div style="font-size:10px;color:var(--text-dim);">DGS2 / ^TNX</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    with c2:
        debt_val = debt.get("value_trillion")
        if debt_val is not None:
            st.markdown(
                f'<div class="card" style="text-align:center;">'
                f'<h4>🇺🇸 U.S. National Debt</h4>'
                f'<div class="big" style="color:#dc2626;">${debt_val:.2f}T</div>'
                f'<div style="font-size:11px;color:var(--text-dim);margin-top:4px;">同比 +{safe_float(debt.get("yoy_chg_pct", 0), 0):.1f}% · {debt.get("asof","—")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="card" style="text-align:center;">'
                f'<h4>🇺🇸 U.S. National Debt</h4>'
                f'<div style="font-size:11px;color:var(--text-dim);margin:20px 0;">📡 数据待接入</div>'
                f'<div style="font-size:10px;color:var(--text-dim);">GFDEBTN</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    with c3:
        mb_val = margin.get("value_billion")
        if mb_val is not None:
            web_cross = ""
            if margin.get("finra_web_value"):
                web_cross = (
                    f'<div style="font-size:9px;color:var(--text-dim);margin-top:2px;">'
                    f'官网直抓 {margin.get("finra_web_month","")}: ${margin.get("finra_web_value",0):.0f}B（交叉校验）</div>'
                )
            st.markdown(
                f'<div class="card" style="text-align:center;">'
                f'<h4>💳 FINRA Margin Debt</h4>'
                f'<div class="big">${mb_val:.0f}B</div>'
                f'<div style="font-size:11px;color:var(--text-dim);margin-top:4px;">环比 {safe_float(margin.get("mom_chg_pct", 0), 0):+.1f}% · 同比 {safe_float(margin.get("yoy_chg_pct", 0), 0):+.1f}%</div>'
                f'<div style="font-size:11px;margin-top:4px;">{margin.get("signal", "—")}</div>'
                f'{web_cross}'
                f'<div style="font-size:9px;color:var(--text-dim);margin-top:2px;">源: {margin.get("source","FRED MDEBT")} · {margin.get("asof","—")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="card" style="text-align:center;">'
                f'<h4>💳 FINRA Margin Debt</h4>'
                f'<div style="font-size:11px;color:var(--text-dim);margin:20px 0;">📡 数据待接入</div>'
                f'<div style="font-size:10px;color:var(--text-dim);">MDEBT</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    with c4:
        nfci_v = nfci.get("value")
        if nfci_v is not None:
            nfci_chg = nfci.get("chg", 0)
            chg_color = "#16a34a" if nfci_chg < 0 else "#dc2626"
            st.markdown(
                f'<div class="card" style="text-align:center;">'
                f'<h4>🏦 NFCI Leverage</h4>'
                f'<div class="big" style="color:{"#dc2626" if nfci_v > 0.5 else "#16a34a"};">{nfci_v:+.2f}</div>'
                f'<div style="font-size:11px;color:{chg_color};font-weight:600;margin-top:4px;">较上周 {nfci_chg:+.3f}</div>'
                f'<div style="font-size:11px;margin-top:4px;">{nfci.get("signal","—")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="card" style="text-align:center;">'
                f'<h4>🏦 NFCI Leverage</h4>'
                f'<div style="font-size:11px;color:var(--text-dim);margin:20px 0;">📡 数据待接入</div>'
                f'<div style="font-size:10px;color:var(--text-dim);">NFCILEVERAGE</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ===== 宏观风险雷达 =====
    radar = U.compute_macro_risk_radar()
    overall = radar.get("overall", {})
    overall_color = {"green": "#16a34a", "yellow": "#f59e0b", "red": "#dc2626"}.get(overall.get("signal"), "#9ca3af")

    def _radar_group_card(group_key, title, icon):
        g = radar.get(group_key, {})
        if not g:
            return ""
        signal = g.get("signal", "yellow")
        sig_color = {"green": "#16a34a", "yellow": "#f59e0b", "red": "#dc2626"}.get(signal, "#9ca3af")
        sig_label = {"green": "✅ 正常", "yellow": "⚠️ 关注", "red": "🔴 警戒"}.get(signal, "—")
        metrics = g.get("metrics", {}) or {}
        emoji = g.get("emoji", "")
        rows = ""
        for k, v in metrics.items():
            rows += f'<div style="display:flex;justify-content:space-between;font-size:11px;padding:2px 0;"><span style="color:var(--text-dim);">{k}</span><b>{v}</b></div>'
        return (
            f'<div class="card" style="padding:12px 14px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
            f'<h4 style="margin:0;font-size:13px;">{icon} {title}</h4>'
            f'<span style="font-size:14px;">{emoji}</span>'
            f'</div>'
            f'<div style="font-size:11px;color:{sig_color};font-weight:600;margin-bottom:6px;">{sig_label}</div>'
            f'{rows}'
            f'<div style="font-size:10px;color:var(--text-dim);margin-top:6px;border-top:1px solid var(--border);padding-top:4px;">{g.get("note","")}</div>'
            f'</div>'
        )

    st.markdown(
        f'<div class="section-title"><span class="accent">🚨</span>宏观风险雷达 Macro Risk Radar '
        f'<span style="font-size:13px;color:{overall_color};margin-left:8px;">{overall.get("emoji","")} {overall.get("note","")}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.markdown(_radar_group_card("regime", "周期 Regime", "🌊"), unsafe_allow_html=True)
        st.markdown(_radar_group_card("risk", "恐慌 Risk", "😱"), unsafe_allow_html=True)
    with rc2:
        st.markdown(_radar_group_card("rates", "利率 Rates", "🏛️"), unsafe_allow_html=True)
        st.markdown(_radar_group_card("ratios", "比值 Ratios", "⚖️"), unsafe_allow_html=True)
    with rc3:
        st.markdown(_radar_group_card("cross_asset", "跨资产 Cross", "🌐"), unsafe_allow_html=True)
        st.markdown(_radar_group_card("a_share", "A股 A-Share", "🇨🇳"), unsafe_allow_html=True)

    # ===== 宏观数值文字总结 =====
    try:
        _macro_text = U.macro_risk_narrative(radar)
        st.markdown(
            f'<div class="card" style="min-height:auto;background:linear-gradient(135deg,#f8fafc 0%,#eef2f7 100%);border-left:4px solid var(--accent);">'
            f'<div style="font-size:11px;font-weight:700;color:var(--accent);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">📝 宏观解读（基于雷达数值）</div>'
            f'<div style="font-size:12.5px;line-height:1.7;color:var(--text);white-space:pre-line;">{_macro_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("宏观文字总结失败: %s", e)

    # ===== 底部信号灯详情（紧跟宏观风险雷达） =====
    try:
        import bottom_signal as BS
        macro_env = BS.compute_macro_environment_score()
        m_score = macro_env.get("score", 0)
        m_hits = macro_env.get("hits", [])
        m_details = macro_env.get("details", {})
        m_emoji, m_label, m_color = BS.traffic_light(m_score)

        st.markdown(
            f'<div class="section-title"><span class="accent">🚦</span>底部信号灯 Bottom Signal '
            f'<span style="font-size:13px;color:{m_color};margin-left:8px;">{m_emoji} 宏观环境分 {m_score}/2 · {m_label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        bs_c1, bs_c2 = st.columns([1, 2])
        with bs_c1:
            st.markdown(
                f'<div class="card" style="border-left:4px solid {m_color};">'
                f'<div style="font-size:14px;font-weight:700;margin-bottom:8px;">{m_emoji} 宏观环境分 {m_score}/2</div>'
                f'<div style="font-size:11px;color:var(--text-dim);line-height:1.6;">'
                f'{"<br>".join(m_hits) if m_hits else "暂无宏观底部信号"}'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            # 显示各维度详情
            for dim_name, dim_data in m_details.items():
                hit = dim_data.get("命中", False)
                color = "#16a34a" if hit else "#9ca3af"
                icon = "✅" if hit else "◯"
                st.markdown(
                    f'<div style="font-size:11px;color:{color};margin:4px 0;">{icon} {dim_name}</div>',
                    unsafe_allow_html=True,
                )

        with bs_c2:
            if STOCKS_DF is not None and not STOCKS_DF.empty:
                _btm_symbols = STOCKS_DF["symbol"].tolist()[:8]
                _btm_rows = []
                for _bsym in _btm_symbols:
                    try:
                        _pe = None
                        _brow = STOCKS_DF[STOCKS_DF["symbol"] == _bsym]
                        if not _brow.empty:
                            _pe = BS._safe_float(_brow.iloc[0].get("PE_Ratio"))
                        _r = BS.calc_bottom_confidence(_bsym, macro=macro_env, pe=_pe, stocks_df=STOCKS_DF)
                        _btm_rows.append(_r)
                    except Exception:
                        continue
                if _btm_rows:
                    _btm_cols = st.columns(min(len(_btm_rows), 8))
                    for _bc, _br in zip(_btm_cols, _btm_rows):
                        with _bc:
                            st.markdown(
                                f'<div style="text-align:center;padding:8px;border-radius:8px;background:var(--bg2);border-top:3px solid {_br["color"]};">'
                                f'<div style="font-size:12px;font-weight:600;">{_br["symbol"]}</div>'
                                f'<div style="font-size:20px;font-weight:800;color:{_br["color"]};">{_br["traffic_light"]} {_br["bottom_score"]}/4</div>'
                                f'<div style="font-size:10px;color:var(--text-dim);">{_br["label"]}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
    except Exception as e:  # noqa: BLE001
        logger.debug("底部信号灯展示失败: %s", e)


def _render_structure_tab() -> None:
    """第1层·证据层 — 结构：全球市场总览(US/CN/HK) + 全市场热力图 + 维科夫吸筹扫描。"""
    fg = U.calculate_fear_greed()
    fg_label = fg.get("label", "中性")
    fg_score = fg.get("score", 50.0)
    fg_color = {"极度贪婪": "#16a34a", "贪婪": "#84cc16", "中性": "#fbbf24", "恐惧": "#f97316", "极度恐惧": "#dc2626"}.get(fg_label, "#9ca3af")

    # ===== 全球市场三段 =====
    st.markdown('<div class="section-title"><span class="accent">🌍</span>全球市场总览（按市场分段）</div>', unsafe_allow_html=True)
    seg_us, seg_cn, seg_hk = st.tabs(["🇺🇸 美股 (US)", "🇨🇳 A股 (CN)", "🇭🇰 港股 (HK)"])

    # --- 美股 ---
    with seg_us:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="card"><h4>📈 美股主指数</h4>', unsafe_allow_html=True)
            us_indices = [
                ("^GSPC", "标普 500"),
                ("^NDX", "纳指 100"),
                ("^DJI", "道指"),
                ("^RUT", "罗素 2000"),
            ]
            for sym, name in us_indices:
                try:
                    t = yf.Ticker(sym)
                    info = t.fast_info
                    px = float(info.last_price) if info.last_price else 0
                    prev = float(info.previous_close) if info.previous_close else 0
                    chg = (px - prev) / prev * 100 if prev else 0
                    color = "#dc2626" if chg > 0 else ("#16a34a" if chg < 0 else "var(--text-dim)")
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border);">'
                        f'<span>{name}</span>'
                        f'<span><b>{px:.2f}</b> <span style="color:{color};font-weight:600;">{chg:+.2f}%</span></span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                except Exception:
                    st.markdown(f'<div style="color:var(--text-dim);">{name}: —</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card"><h4>📊 美股情绪</h4>', unsafe_allow_html=True)
            try:
                st.markdown(
                    f'<div style="font-size:11px;color:var(--text-dim);">Fear &amp; Greed</div>'
                    f'<div style="font-size:24px;font-weight:800;color:{fg_color};">{fg_score:.0f} · {fg_label}</div>',
                    unsafe_allow_html=True,
                )
            except Exception:
                st.markdown('<div style="color:var(--text-dim);">F&G 待加载</div>', unsafe_allow_html=True)
            try:
                vix_t = yf.Ticker("^VIX")
                vix_info = vix_t.fast_info
                vix_now = float(vix_info.last_price) if vix_info.last_price else 0
                vix_color = "#dc2626" if vix_now > 25 else "#16a34a"
                st.markdown(
                    f'<div style="margin-top:8px;font-size:11px;color:var(--text-dim);">VIX</div>'
                    f'<div style="font-size:22px;font-weight:800;color:{vix_color};">{vix_now:.2f}</div>'
                    f'<div style="font-size:10px;color:var(--text-dim);">{"⚠️ 警戒" if vix_now > 25 else "正常"}</div>',
                    unsafe_allow_html=True,
                )
            except Exception:
                pass
            try:
                tnx_t = yf.Ticker("^TNX")
                tnx_info = tnx_t.fast_info
                tnx_now = float(tnx_info.last_price) if tnx_info.last_price else 0
                st.markdown(
                    f'<div style="margin-top:8px;font-size:11px;color:var(--text-dim);">10Y 美债</div>'
                    f'<div style="font-size:22px;font-weight:800;">{tnx_now:.2f}%</div>',
                    unsafe_allow_html=True,
                )
            except Exception:
                pass
            st.markdown("</div>", unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="card"><h4>🔥 自选股 Top 5 (美股)</h4>', unsafe_allow_html=True)
            if STOCKS_DF is not None and not STOCKS_DF.empty:
                us_df = STOCKS_DF[~STOCKS_DF["symbol"].str.contains(r"\.HK|\.SS|\.SZ", regex=True, na=False)]
                if "涨跌幅" in us_df.columns and not us_df.empty:
                    for _, r in us_df.head(5).iterrows():
                        chg = safe_float(r.get("涨跌幅"))
                        c_chg = "#dc2626" if chg > 0 else ("#16a34a" if chg < 0 else "var(--text-dim)")
                        st.markdown(
                            f'<div style="display:flex;justify-content:space-between;padding:4px 0;font-size:12px;">'
                            f'<span><b>{r["symbol"]}</b> <span style="color:var(--text-dim);font-size:10px;">{U.STOCK_NAMES.get(r["symbol"], "")}</span></span>'
                            f'<span style="color:{c_chg};font-weight:600;">{chg:+.2f}%</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
            st.markdown("</div>", unsafe_allow_html=True)

    # --- A股 ---
    with seg_cn:
        # 不再硬性依赖 akshare：fetch_a_share_overview 内部已多源降级（东财 push2 → 腾讯 gtimg → akshare）
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="card"><h4>📈 A股主指数</h4>', unsafe_allow_html=True)
            with st.spinner("拉取 A股实时行情…"):
                cn_overview = U.fetch_a_share_overview()
            if cn_overview.get("error"):
                st.caption(f"⚠️ {cn_overview['error']}")
            for idx in cn_overview.get("indices", [])[:8]:
                name = idx.get("名称", "")
                px = float(idx.get("最新价", 0) or 0)
                chg = float(idx.get("涨跌幅", 0) or 0)
                color = "#dc2626" if chg > 0 else ("#16a34a" if chg < 0 else "var(--text-dim)")
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--border);">'
                    f'<span>{name}</span>'
                    f'<span><b>{px:.2f}</b> <span style="color:{color};font-weight:600;">{chg:+.2f}%</span></span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="card"><h4>📊 A股情绪</h4>', unsafe_allow_html=True)
                adv = cn_overview.get("advance") or cn_overview.get("上涨")
                dec = cn_overview.get("decline") or cn_overview.get("下降")
                if adv is not None and dec is not None:
                    st.markdown(
                        f'<div style="font-size:11px;color:var(--text-dim);">涨 / 跌家数</div>'
                        f'<div style="font-size:18px;font-weight:800;">'
                        f'<span style="color:#dc2626;">{int(adv)} 涨</span> · '
                        f'<span style="color:#16a34a;">{int(dec)} 跌</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                up_lim = cn_overview.get("涨停")
                dn_lim = cn_overview.get("跌停")
                if up_lim is not None:
                    st.markdown(
                        f'<div style="margin-top:8px;font-size:11px;color:var(--text-dim);">涨停 / 跌停</div>'
                        f'<div style="font-size:18px;font-weight:800;">'
                        f'<span style="color:#dc2626;">{int(up_lim)}</span> · '
                        f'<span style="color:#16a34a;">{int(dn_lim)}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                nb = cn_overview.get("north_flow")
                if nb is not None:
                    nb_color = "#dc2626" if nb > 0 else "#16a34a"
                    st.markdown(
                        f'<div style="margin-top:8px;font-size:11px;color:var(--text-dim);">北向资金 (亿元)</div>'
                        f'<div style="font-size:18px;font-weight:800;color:{nb_color};">{nb:+.2f}</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)
            with c3:
                st.markdown('<div class="card"><h4>🔥 A股 Top 5 (自选股)</h4>', unsafe_allow_html=True)
                if STOCKS_DF is not None and not STOCKS_DF.empty:
                    cn_df = STOCKS_DF[STOCKS_DF["symbol"].str.contains(r"\.SS|\.SZ", regex=True, na=False)]
                    if "涨跌幅" in cn_df.columns and not cn_df.empty:
                        for _, r in cn_df.head(5).iterrows():
                            chg = safe_float(r.get("涨跌幅"))
                            c_chg = "#dc2626" if chg > 0 else ("#16a34a" if chg < 0 else "var(--text-dim)")
                            st.markdown(
                                f'<div style="display:flex;justify-content:space-between;padding:4px 0;font-size:12px;">'
                                f'<span><b>{r["symbol"]}</b> <span style="color:var(--text-dim);font-size:10px;">{U.STOCK_NAMES.get(r["symbol"], "")}</span></span>'
                                f'<span style="color:{c_chg};font-weight:600;">{chg:+.2f}%</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                st.markdown("</div>", unsafe_allow_html=True)

    # --- 港股 ---
    with seg_hk:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="card"><h4>📈 港股主指数</h4>', unsafe_allow_html=True)
            hk_indices = [
                ("^HSI", "恒生指数"),
                ("^HSCE", "国企指数"),
                ("^HSTECH", "恒生科技"),
                ("^HSCCI", "恒生地产"),
            ]
            for sym, name in hk_indices:
                try:
                    t = yf.Ticker(sym)
                    info = t.fast_info
                    px = float(info.last_price) if info.last_price else 0
                    prev = float(info.previous_close) if info.previous_close else 0
                    chg = (px - prev) / prev * 100 if prev else 0
                    color = "#dc2626" if chg > 0 else ("#16a34a" if chg < 0 else "var(--text-dim)")
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--border);">'
                        f'<span>{name}</span>'
                        f'<span><b>{px:.2f}</b> <span style="color:{color};font-weight:600;">{chg:+.2f}%</span></span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                except Exception:
                    st.markdown(f'<div style="color:var(--text-dim);">{name}: —</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card"><h4>📊 港股情绪</h4>', unsafe_allow_html=True)
            try:
                hsi_t = yf.Ticker("^HSI")
                hsi_px = float(hsi_t.fast_info.last_price or 0)
                hsi_chg = float(hsi_t.fast_info.last_price or 0) / float(hsi_t.fast_info.previous_close or 1) - 1
                hsi_color = "#dc2626" if hsi_chg > 0 else "#16a34a"
                st.markdown(
                    f'<div style="font-size:11px;color:var(--text-dim);">恒指动量 (近 5 日)</div>'
                    f'<div style="font-size:24px;font-weight:800;color:{hsi_color};">{hsi_chg*100:+.2f}%</div>',
                    unsafe_allow_html=True,
                )
            except Exception:
                pass
            try:
                hsi_hist = yf.download("^HSI", period="3mo", progress=False)
                if not hsi_hist.empty:
                    ret = hsi_hist["Close"].pct_change().dropna()
                    if len(ret) >= 20:
                        vol_20d = float(ret.tail(20).std() * (252**0.5) * 100)
                        st.markdown(
                            f'<div style="margin-top:8px;font-size:11px;color:var(--text-dim);">20 日年化波动</div>'
                            f'<div style="font-size:22px;font-weight:800;">{vol_20d:.1f}%</div>',
                            unsafe_allow_html=True,
                        )
            except Exception:
                pass
            st.markdown("</div>", unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="card"><h4>🔥 港股 Top 5 (自选股)</h4>', unsafe_allow_html=True)
            if STOCKS_DF is not None and not STOCKS_DF.empty:
                hk_df = STOCKS_DF[STOCKS_DF["symbol"].str.contains(r"\.HK", regex=True, na=False)]
                if "涨跌幅" in hk_df.columns and not hk_df.empty:
                    for _, r in hk_df.head(5).iterrows():
                        chg = safe_float(r.get("涨跌幅"))
                        c_chg = "#dc2626" if chg > 0 else ("#16a34a" if chg < 0 else "var(--text-dim)")
                        st.markdown(
                            f'<div style="display:flex;justify-content:space-between;padding:4px 0;font-size:12px;">'
                            f'<span><b>{r["symbol"]}</b> <span style="color:var(--text-dim);font-size:10px;">{U.STOCK_NAMES.get(r["symbol"], "")}</span></span>'
                            f'<span style="color:{c_chg};font-weight:600;">{chg:+.2f}%</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
            st.markdown("</div>", unsafe_allow_html=True)

    # ===== 全市场热力图 =====
    st.markdown('<div class="section-title"><span class="accent">🔥</span>全市场热力图</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
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
    with c3:
        st.caption("🇨🇳 A股全市场 (按申万行业 / 当日涨跌幅)")
        with st.spinner("拉取A股热力图…"):
            a_hm = U.build_heatmap_data(U.A_SHARE_HEATMAP_TICKERS)
        _draw_heatmap(a_hm, "sector", "symbol", "change_pct", "weight")

    # ===== 维科夫吸筹扫描 =====
    st.markdown('<div class="section-title"><span class="accent">🧬</span>维科夫吸筹扫描 <span style="font-size:11px;color:var(--text-dim);">纯计算 · 无 LLM</span></div>', unsafe_allow_html=True)
    wyc_syms = [s for s in (STOCKS_DF["symbol"].tolist() if STOCKS_DF is not None and not STOCKS_DF.empty else []) if not s.endswith((".HK", ".SS", ".SZ"))][:8]
    if wyc_syms:
        wyc_rows = []
        for wsym in wyc_syms:
            try:
                wh = _fetch_price_history(wsym, period="6mo", interval="1d")
                if wh is None or wh.empty or len(wh) < 120:
                    continue
                w = screener.detect_wyckoff_events(wh)
                if w.get("ok"):
                    wyc_rows.append((wsym, w))
            except Exception:  # noqa: BLE001
                continue
        wyc_rows.sort(key=lambda x: -x[1]["confidence"])
        if wyc_rows:
            cols = st.columns(len(wyc_rows))
            for col, (wsym, w) in zip(cols, wyc_rows):
                conf = w["confidence"]
                conf_color = "#16a34a" if conf >= 0.71 else ("#f59e0b" if conf >= 0.43 else ("#3b82f6" if conf >= 0.14 else "#9ca3af"))
                with col:
                    st.markdown(
                        f'<div class="card" style="text-align:center;border-top:3px solid {conf_color};">'
                        f'<h4>{wsym}</h4>'
                        f'<div style="display:flex;justify-content:center;margin:6px 0;">{_donut(conf * 100, size=62, label="吸筹置信度")}</div>'
                        f'<div style="font-size:11px;color:var(--text-dim);">{w["stage"]}</div>'
                        f'<div style="font-size:10px;color:var(--text-dim);margin-top:2px;">{w["event_count"]}/7 事件 · {w["phase"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            with st.expander("📋 事件明细"):
                for wsym, w in wyc_rows:
                    evs = " → ".join(e["event"] for e in w["events"]) or "无"
                    st.markdown(f"**{wsym}** ({w['stage']}) 置信度 {w['confidence']} · 事件序列: {evs}")
                    st.caption(w["summary"])

            _top_w = wyc_rows[0][1]
            _exp = screener.explain_wyckoff(_top_w)
            st.markdown(
                f'<div class="card" style="min-height:auto;background:linear-gradient(135deg,#f0f9ff 0%,#e0f2fe 100%);border-left:4px solid #0284c7;">'
                f'<div style="font-size:11px;font-weight:700;color:#0369a1;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">'
                f'🧠 维科夫吸筹 · AI 解读（以最高置信 {wyc_rows[0][0]} 为例）</div>'
                f'<div style="font-size:12.5px;line-height:1.7;color:#0c4a6e;">'
                f'<b>置信度含义：</b>{_exp["confidence_meaning"]}<br>'
                f'<b>当前阶段：</b>{_exp["stage_meaning"]}<br>'
                f'<b>事件序列：</b>{_exp["sequence_meaning"]}'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            if st.button("🤖 生成 AI 研判（DeepSeek · 仅解释结构化数据）", key="wyc_ai_dash", use_container_width=True):
                with st.spinner("调用 DeepSeek 生成研判…"):
                    _wh_top = _fetch_price_history(wyc_rows[0][0], period="6mo", interval="1d")
                    _mf_top = screener.score_multi_factor(_wh_top) if (_wh_top is not None and not _wh_top.empty) else {"score": "—", "bias": "—", "threshold_pass": False, "factors": {}}
                    _prompt = screener.build_screener_narrative_prompt(
                        _top_w, _mf_top, wyc_rows[0][0], U.STOCK_NAMES.get(wyc_rows[0][0], ""),
                    )
                    _ai = U._call_llm(
                        messages=[
                            {"role": "system", "content": "你是量化研究助理，输出简体中文，只解释给定数据，不编造价格/消息/目标价。"},
                            {"role": "user", "content": _prompt},
                        ],
                        prefer="deepseek",
                    )
                if _ai:
                    st.success(_ai)
                else:
                    st.info("未配置 DEEPSEEK_API_KEY（或调用失败），已降级展示上方结构化解读。配置方式：Streamlit Cloud → Secrets 增加 DEEPSEEK_API_KEY。")
            st.caption("💡 指引：" + _exp["guidance"].replace("\n", " "))
        else:
            st.caption("K线数据不足，无法扫描（需要至少 120 根日K）")
    else:
        st.caption("暂无美股标的可扫描")

    # ===== 市场底部信号灯（与维科夫并列，不互相修改）=====
    st.markdown('<div class="section-title"><span class="accent">🚦</span>市场底部信号灯 <span style="font-size:11px;color:var(--text-dim);">宏观环境分(全市场统一) + 个股结构分(逐股不同) · 纯计算</span></div>', unsafe_allow_html=True)
    try:
        # 宏观环境分（只算一次）
        _policy = U.fetch_policy_news_free(top_n=30)
        _macro = BS.compute_macro_environment_score(policy_news=_policy)
        macro_score = _macro["score"]
        _m_emoji, _m_label, _m_color = BS.traffic_light(macro_score)
        st.markdown(
            f'<div class="card" style="border-left:4px solid {_m_color};">'
            f'<div style="font-size:14px;font-weight:700;margin-bottom:6px;">'
            f'{_m_emoji} 宏观环境分 · {macro_score}/2 · {_m_label}</div>'
            f'<div style="font-size:11px;color:var(--text-dim);line-height:1.6;">'
            f'监管恐慌: {"命中" if _macro["details"].get("监管恐慌",{}).get("命中") else "未命中"} '
            f'(政策新闻恐慌={_macro["details"].get("监管恐慌",{}).get("政策新闻恐慌")})<br>'
            f'杠杆去化: {"命中" if _macro["details"].get("杠杆去化",{}).get("命中") else "未命中"} '
            f'({_macro["details"].get("杠杆去化",{}).get("离散度描述", "")})'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # 个股底部确信度（取 watchlist 前 6 只）
        _btm_symbols = [s for s in (STOCKS_DF["symbol"].tolist() if STOCKS_DF is not None and not STOCKS_DF.empty else [])][:6]
        if _btm_symbols:
            _btm_rows = []
            for _bsym in _btm_symbols:
                try:
                    _pe = None
                    if STOCKS_DF is not None and not STOCKS_DF.empty:
                        _brow = STOCKS_DF[STOCKS_DF["symbol"] == _bsym]
                        if not _brow.empty:
                            _pe = BS._safe_float(_brow.iloc[0].get("PE_Ratio"))
                    _r = BS.calc_bottom_confidence(
                        _bsym,
                        macro=_macro,
                        pe=_pe,
                        stocks_df=STOCKS_DF,
                    )
                    _btm_rows.append(_r)
                except Exception:
                    continue
            if _btm_rows:
                _btm_cols = st.columns(min(len(_btm_rows), 6))
                for _bc, _br in zip(_btm_cols, _btm_rows):
                    with _bc:
                        st.markdown(
                            f'<div class="card" style="text-align:center;border-top:3px solid {_br["color"]};">'
                            f'<h4>{_br["symbol"]}</h4>'
                            f'<div style="font-size:28px;font-weight:800;color:{_br["color"]};">{_br["traffic_light"]} {_br["bottom_score"]}/4</div>'
                            f'<div style="font-size:11px;color:var(--text-dim);margin-top:4px;">{_br["label"]}</div>'
                            f'<div style="font-size:10px;color:var(--text-dim);margin-top:2px;">'
                            f'宏观{_br["macro_score"]} + 结构{_br["individual_score"]}'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )
                with st.expander("📋 底部信号灯明细"):
                    for _br in _btm_rows:
                        st.markdown(f"**{_br['symbol']}** {_br['traffic_light']} {_br['bottom_score']}/4 — {_br['label']}")
                        _ind = _br.get("individual_details", {})
                        _mh = _br.get("macro_hits", [])
                        _ih = _br.get("individual_hits", [])
                        st.caption(
                            f"宏观命中: {'; '.join(_mh) if _mh else '无'} · "
                            f"个股命中: {'; '.join(_ih) if _ih else '无'}"
                        )
    except Exception as _e:  # noqa: BLE001
        st.caption(f"底部信号灯暂不可用: {_e}")


def _render_events_tab() -> None:
    """第1层·证据层 — 事件：经济日程 + FedWatch。"""
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
        st.markdown("</div>", unsafe_allow_html=True)


def _render_research_tab() -> None:
    """第1层·证据层 — 研判：Morning Brief / Evening Recap + 智能荐股。"""
    # ===== Morning Brief / Evening Recap =====
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

    # ===== 智能荐股 =====
    st.markdown('<div class="section-title"><span class="accent">📊</span>智能荐股（技术面 · 估值 · 动量 · 杠杆止损）</div>', unsafe_allow_html=True)
    st.caption("综合 技术面(MA20/60 趋势) + 估值(PE) + 动量(当日) + 杠杆止损位，量化初筛三类建议。综合评分≥60 为买入信号。⚠️ 仅供参考，不构成投资建议。")
    try:
        syms = list(Config().stocks)
        with st.spinner("正在计算智能荐股（拉取行情 / PE）…"):
            metrics = _cached_metrics(tuple(syms))
        rec = U.recommend_stocks(metrics)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### ⚡ 日内做T候选（高波动）")
            if rec["intraday_t"]:
                for r in rec["intraday_t"]:
                    lo = r["last"] * (1 - r["atrPct"] / 100) if r["last"] else 0
                    up = r["last"] * (1 + r["atrPct"] / 100) if r["last"] else 0
                    st.markdown(_rec_md(r, f"波动 {r['atrPct']}% · T区间 ↓{lo:.2f} ↑{up:.2f}"), unsafe_allow_html=True)
            else:
                st.caption("暂无高波动标的（需 ATR≥2.5%）")
        with c2:
            st.markdown("##### 📈 中期持股推荐")
            for r in rec["midterm_hold"]:
                pe = r["pe"] if r["pe"] is not None else "—"
                st.markdown(_rec_md(r, f"评分 {r['comp']} {_bias_emoji(r['bias'])} · PE {pe} · 止损 {r['stop']}"), unsafe_allow_html=True)
        with c3:
            st.markdown("##### ✅ 综合买入信号")
            if rec["buy"]:
                for r in rec["buy"]:
                    pe = r["pe"] if r["pe"] is not None else "—"
                    st.markdown(_rec_md(r, f"评分 {r['comp']} {_bias_emoji(r['bias'])} · PE {pe}"), unsafe_allow_html=True)
            else:
                st.caption("当前无综合评分≥60 的标的")
    except Exception as e:  # noqa: BLE001
        st.warning(f"⚠️ 智能荐股计算失败: {e}")

    # ===== 3 年回测验证（Phase 2 ③） =====
    st.markdown('<div class="section-title"><span class="accent">📉</span>3 年历史回测验证（维科夫 / 多因子 / R倍数）</div>', unsafe_allow_html=True)
    st.caption("对历史信号做滚动回测，检验策略是否经得起统计检验。运行 `python backtest.py` 更新数据。")
    try:
        import backtest as BT
        report = BT.load_report()
        results = report.get("results", {})
        if not results:
            st.info("回测数据尚未生成。运行 `python backtest.py` 后自动展示。")
        else:
            rows = []
            for sym, r in results.items():
                if "error" in r:
                    continue
                w = r.get("wyckoff", {})
                m = r.get("multifactor", {})
                rm = r.get("r_multiple", {})
                g = m.get("gte60", {}) or {}
                l = m.get("lt60", {}) or {}
                rows.append({
                    "标的": sym,
                    "维科夫信号数": w.get("signals", 0),
                    "维科夫20日胜率": f"{w.get('win_rate_20d', '—')}%",
                    "维科夫20日均收": f"{w.get('avg_ret_20d', '—')}%",
                    "多因子≥60胜率": f"{g.get('win_rate_20d', '—')}%",
                    "多因子<60胜率": f"{l.get('win_rate_20d', '—')}%",
                    "R倍数收益": f"{rm.get('r_total_return_pct', '—')}%",
                    "Buy&Hold": f"{rm.get('bh_total_return_pct', '—')}%",
                    "R超额": f"{rm.get('r_excess_pct', '—')}pp",
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption(
                    "解读：① 多因子 ≥60 vs <60 的胜率差是「60分买入线」有效性的直接证据；"
                    "② R倍数超额为负说明固定止盈规则在牛市跑输 Buy&Hold；"
                    "③ 维科夫信号胜率 >55% 才值得纳入初筛。数据为免费源日线，仅供参考。"
                )
    except Exception as e:  # noqa: BLE001
        logger.debug("回测展示失败: %s", e)

    # ===== 阈值扫描校准（40-60 分） =====
    st.markdown('<div class="section-title"><span class="accent">🎚️</span>多因子阈值扫描校准（40-60 分）</div>', unsafe_allow_html=True)
    st.caption("全量 43 只 3 年滚动回测，验证 60 分买入线是否最优。运行 `python threshold_scan.py` 更新数据。")
    try:
        import threshold_scan as TS
        ts_report = TS.load_report()
        by_th = ts_report.get("by_threshold", {})
        rec = ts_report.get("recommendation", {})
        if by_th:
            t_rows = []
            for t, a in by_th.items():
                plr = a.get("profit_loss_ratio")
                t_rows.append({
                    "阈值": f"≥{t}",
                    "信号数": a.get("signals", 0),
                    "样本充足": "✅" if a.get("sample_sufficient") else "⚠️不足",
                    "20日胜率": f"{a.get('win_rate', 0):.1f}%",
                    "平均收益": f"{a.get('avg_return', 0):+.2f}%",
                    "中位收益": f"{a.get('median_return', 0):+.2f}%",
                    "盈亏比": f"{plr:.2f}" if plr else "—",
                    "均回撤": f"{a.get('avg_mdd', 0):.2f}%",
                })
            st.dataframe(pd.DataFrame(t_rows), use_container_width=True, hide_index=True)
            best = rec.get("best_threshold")
            if best:
                st.success(f"🎯 推荐阈值：**≥{best} 分**（{rec.get('note', '')}）")
            st.caption(
                "解读：全量聚合下 60 分仍是最优（胜率61.4%/均收+6.56%/盈亏比1.76，样本1223充足）；"
                "但单只高波动科技股（如 NVDA/MSFT）存在异质性——60 分阈值对它们可能失效，建议叠加底部信号灯二次过滤。"
                "详见 THRESHOLD_SCAN_REPORT.md。"
            )
        else:
            st.info("阈值扫描数据尚未生成。运行 `python threshold_scan.py` 后自动展示。")
    except Exception as e:  # noqa: BLE001
        logger.debug("阈值扫描展示失败: %s", e)


def _render_ai_traders_tab() -> None:
    """🤖 AI炒手对战：KIMI vs DeepSeek 净值对比、持仓、交易日志、胜率统计。"""
    st.markdown('<div class="section-title"><span class="accent">🤖</span>AI 炒手对战 AI Trader Battle</div>', unsafe_allow_html=True)

    import ai_traders as AT

    # 加载两个模型的数据
    models = {}
    for mid in AT.MODEL_IDS:
        try:
            port = AT.load_portfolio(mid)
            nav_path = AT._trader_dir(mid) / "nav_history.csv"
            trades_path = AT._trader_dir(mid) / "trades.jsonl"
            nav_df = None
            if nav_path.exists():
                import pandas as pd
                nav_df = pd.read_csv(nav_path)
            trades = AT.load_trades(mid)
            models[mid] = {"portfolio": port, "nav": nav_df, "trades": trades}
        except Exception as e:  # noqa: BLE001
            logger.debug("加载 %s 数据失败: %s", mid, e)
            models[mid] = None

    # 如果没有任何数据，显示提示
    if not any(models.values()):
        st.info("🤖 AI 炒手数据尚未生成。首次运行 `python ai_traders.py` 后会自动创建。")
        return

    # 顶部：净值对比
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("#### 📈 净值曲线对比（归一化）")
        nav_data = []
        for mid, data in models.items():
            if data and data["nav"] is not None and not data["nav"].empty:
                ndf = data["nav"].copy()
                ndf["model"] = mid.upper()
                nav_data.append(ndf)
        if nav_data:
            import pandas as pd
            import plotly.graph_objects as go
            nav_all = pd.concat(nav_data, ignore_index=True)
            fig = go.Figure()
            for mid in AT.MODEL_IDS:
                sub = nav_all[nav_all["model"] == mid.upper()]
                if not sub.empty:
                    # 归一化到起始值 = 100
                    base = sub["nav"].iloc[0]
                    sub = sub.copy()
                    sub["norm"] = sub["nav"] / base * 100
                    fig.add_trace(go.Scatter(
                        x=sub["date"], y=sub["norm"],
                        mode="lines+markers", name=mid.upper(),
                        line=dict(width=2),
                    ))
            fig.update_layout(
                height=350, margin=dict(l=40, r=20, t=30, b=40),
                xaxis_title="日期", yaxis_title="净值 (起点=100)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            _plotly_chart(fig, use_container_width=True)
        else:
            st.caption("暂无净值历史")

    with c2:
        st.markdown("#### 💼 当前持仓")
        for mid, data in models.items():
            if not data:
                continue
            port = data["portfolio"]
            positions = port.get("positions", {})
            cash = port.get("cash", AT.INITIAL_CASH)
            pos_value = sum(p.get("qty", 0) * p.get("avg_cost", 0) for p in positions.values())
            nav = cash + pos_value
            st.markdown(
                f'<div style="padding:10px;border-radius:8px;background:var(--bg2);margin-bottom:8px;">'
                f'<div style="font-size:13px;font-weight:700;">{mid.upper()}</div>'
                f'<div style="font-size:11px;color:var(--text-dim);">净值: ${nav:,.2f}</div>'
                f'<div style="font-size:11px;color:var(--text-dim);">现金: ${cash:,.2f}</div>'
                f'<div style="font-size:10px;color:var(--text-dim);margin-top:4px;">持仓 {len(positions)} 只</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # 中部：持仓明细表格
    st.markdown("#### 📋 持仓明细")
    for mid, data in models.items():
        if not data:
            continue
        positions = data["portfolio"].get("positions", {})
        if positions:
            rows = []
            for sym, pos in positions.items():
                rows.append({
                    "模型": mid.upper(), "代码": sym,
                    "数量": pos.get("qty", 0), "成本价": f"${pos.get('avg_cost', 0):.2f}",
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # 下部：交易日志时间线
    st.markdown("#### 📝 交易日志")
    for mid, data in models.items():
        if not data or not data["trades"]:
            continue
        with st.expander(f"{mid.upper()} 交易记录 ({len(data['trades'])} 笔)"):
            for t in data["trades"][-20:]:  # 最近20笔
                action_emoji = "🟢" if t.get("action") == "BUY" else "🔴"
                st.markdown(
                    f"{action_emoji} **{t.get('date', '')}** {t.get('action', '')} "
                    f"**{t.get('symbol', '')}** {t.get('qty', 0)}股 @ ${t.get('price', 0):.2f} "
                    f"| 评分{t.get('score_at_trade', 'N/A')} | 来源: {t.get('source', '')}"
                )
                if t.get("reasoning"):
                    st.caption(f"理由: {t['reasoning']}")

    # 胜率对比（读取 threshold_validation_report.json）
    st.markdown("#### 📊 候选池 vs 模型自选 胜率对比")
    report_path = DATA_DIR / "threshold_validation_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            rows = []
            for mid, r in report.get("models", {}).items():
                if "error" in r:
                    continue
                gte = r.get("gte60", {})
                lt = r.get("lt60", {})
                rows.append({
                    "模型": mid.upper(),
                    "≥60分交易": gte.get("count", 0),
                    "≥60平均收益": f"{gte.get('avg_return', 0):.2f}%",
                    "≥60胜率": f"{gte.get('win_rate', 0):.1f}%",
                    "<60分交易": lt.get("count", 0),
                    "<60平均收益": f"{lt.get('avg_return', 0):.2f}%",
                    "<60胜率": f"{lt.get('win_rate', 0):.1f}%",
                })
            if rows:
                import pandas as pd
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        except Exception as e:  # noqa: BLE001
            logger.debug("胜率对比展示失败: %s", e)
    else:
        st.caption("回测报告尚未生成。运行 `python backtest_threshold.py` 后自动更新。")


def page_dashboard():
    st.markdown('<div class="section-title"><span class="accent">🏠</span>市场全景 Dashboard</div>', unsafe_allow_html=True)

    # ===== 第0层 · 结论区（唯一默认可见）=====
    _render_conclusion_zone()

    # ===== 第1层 · 证据层（5 主题 Tab）=====
    st.markdown('<div class="layer-badge">第 1 层 · 证据层 — 按主题组织，点击展开</div>', unsafe_allow_html=True)
    tab_macro, tab_struct, tab_event, tab_research, tab_ai = st.tabs(
        ["📊 宏观", "🧩 结构", "📅 事件", "🧠 研判", "🤖 AI炒手"]
    )
    with tab_macro:
        _render_macro_tab()
    with tab_struct:
        _render_structure_tab()
    with tab_event:
        _render_events_tab()
    with tab_research:
        _render_research_tab()
    with tab_ai:
        _render_ai_traders_tab()


def _draw_heatmap(df: pd.DataFrame, sector_col: str, sym_col: str, color_col: str, size_col: str):
    """绘制 plotly treemap 热力图。颜色范围按当日收益率 5%-95% 分位自适应。"""
    if df is None or df.empty:
        st.info("热力图数据为空")
        return
    try:
        # 动态色域：按该市场当日收益率分布的 5%-95% 分位数（机构惯例，避免普涨普跌日颜色失真）
        vals = pd.to_numeric(df[color_col], errors="coerce").dropna()
        if len(vals) >= 5:
            lo = float(vals.quantile(0.05))
            hi = float(vals.quantile(0.95))
            # 防止零/过窄区间导致渲染异常；绝对值对称化，中心对齐 0
            bound = max(abs(lo), abs(hi), 0.5)
            rc = (-bound, bound)
        else:
            rc = (-3, 3)
        fig = px.treemap(
            df,
            path=[px.Constant("全市场"), sector_col, sym_col],
            values=size_col,
            color=color_col,
            color_continuous_scale=["#16a34a", "#84cc16", "#f1f5f9", "#fbbf24", "#dc2626"],
            color_continuous_midpoint=0,
            range_color=rc,
            custom_data=[color_col, "price"],
        )
        fig.update_traces(
            textinfo="label+text",
            texttemplate="<b>%{label}</b><br>%{customdata[0]:.2f}%",
            hovertemplate="<b>%{label}</b><br>涨跌: %{customdata[0]:.2f}%<br>价格: %{customdata[1]:.2f}<extra></extra>",
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=380, paper_bgcolor="rgba(0,0,0,0)")
        _plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
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
        # 第一层：按市场分类（港股 / A股 / 美股）
        hk_syms = [s for s in all_symbols if s.endswith(".HK")]
        a_syms = [s for s in all_symbols if s.endswith((".SS", ".SZ"))]
        us_syms = [s for s in all_symbols if not s.endswith((".HK", ".SS", ".SZ"))]
        market_caps = []
        if us_syms:
            market_caps.append("🇺🇸 美股")
        if hk_syms:
            market_caps.append("🇭🇰 港股")
        if a_syms:
            market_caps.append("🇨🇳 A股")
        market = st.radio("市场分类", market_caps, horizontal=True, label_visibility="collapsed", key="market_radio")
        if "美股" in market:
            market_syms = us_syms
        elif "港股" in market:
            market_syms = hk_syms
        else:
            market_syms = a_syms

        # 第二层：在所选市场内搜索 / 选择（自选列表直点，替代下拉）
        q = st.text_input("🔍 搜索代码/名称", "")
        if q:
            market_syms = [s for s in market_syms if q.upper() in s.upper()]
        st.caption(f"共 {len(market_syms)} 只")
        if "selected_sym" not in st.session_state or st.session_state.selected_sym not in market_syms:
            st.session_state.selected_sym = market_syms[0] if market_syms else None

        # 自选列表（卡片式，点击即查看 —— Bloomberg 左侧列表风格）
        st.markdown("##### 📋 自选列表")
        for sym in market_syms[:15]:
            r = work_df[work_df["symbol"] == sym].iloc[0] if not work_df[work_df["symbol"] == sym].empty else None
            if r is None:
                continue
            chg = safe_float(r.get("涨跌幅"))
            rsi = safe_float(r.get("RSI_14"), 50)
            rsi_color = "#dc2626" if rsi > 70 else ("#16a34a" if rsi < 30 else "#6b7280")
            cls = "active" if sym == st.session_state.selected_sym else ""
            if st.button(
                f"{sym}  {U.STOCK_NAMES.get(sym, '')}  R{rsi:.0f}  {fmt_pct(chg, with_sign=False)}",
                key=f"pick_{sym}",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state.selected_sym = sym
                st.rerun()
            # 高亮当前选中（用 CSS 类在按钮下方不可行，改为在按钮前输出标记）
            if cls == "active":
                st.markdown(f"<div style='font-size:9px;color:var(--accent);margin:-4px 0 2px 2px;'>▼ {sym} 正在分析</div>", unsafe_allow_html=True)
        sel = st.session_state.selected_sym

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
        _sub = work_df[work_df["symbol"] == sel]
        if _sub.empty:
            st.warning("⚠️ 未找到该标的的行情记录（数据可能未生成），请刷新数据后重试。")
            return
        r = _sub.iloc[0]
        sym = sel
        is_hk = sym.endswith(".HK")
        market_label = "港股" if is_hk else ("A股" if sym.endswith((".SS", ".SZ")) else "美股")

        # ===== 固定顶部：概览 + AI摘要（始终可见，不随 Tab 切换）=====
        chg = safe_float(r.get("涨跌幅"))
        price = safe_float(r.get("收盘价"))
        _price_src = "stocks.csv"
        _ts = r.get("日期", "—")
        _delayed_note = ""  # 免费行情源延迟提示
        try:
            if sym.endswith((".SS", ".SZ")):
                _acode = sym.split(".")[0]
                _aq = U.fetch_a_share_quote(_acode)
                if _aq and _aq.get("最新价"):
                    price = float(_aq["最新价"])
                    chg = float(_aq.get("涨跌幅", chg))
                    _price_src = "东方财富"
                    _ts = datetime.now().strftime("%H:%M")
            else:
                _rq = U.fetch_realtime_quote(sym)
                if _rq.get("ok") and _rq.get("last"):
                    price = float(_rq["last"])
                    chg = float(_rq.get("pct", chg))
                    _price_src = _rq.get("source", "行情")
                    _ts = datetime.now().strftime("%H:%M")
            # 免费聚合源行情有延迟，标注以免误判为交易所实时
            if _price_src != "stocks.csv":
                _delayed_note = " ⚠️免费源·延迟约15-20分钟"
        except Exception:  # noqa: BLE001
            pass
        st.markdown(
            f"""
<div class="card">
    <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
            <div style="font-size:22px;font-weight:800;">{sym} <span style='font-size:13px;color:var(--text-dim);font-weight:500;'>{market_label}</span></div>
            <div style="font-size:11px;color:var(--text-dim);">{_price_src} · {_ts}{_delayed_note}</div>
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

        # AI 摘要（固定顶部）
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

        # ===== 第2层·钻取层（3 主题 Subtab）=====
        t1, t2, t3 = st.tabs(["📈 K线技术面", "🎯 策略与风险", "🧠 AI研判"])

        # ---------- Subtab 1：K线技术面（含实时盘口/分时/成交明细）----------
        with t1:
            # 实时分时（当日分钟级）
            st.markdown('<div class="section-title">📈 当日分时（实时）</div>', unsafe_allow_html=True)
            try:
                trend = U.fetch_intraday_trend(sym)
                if trend is not None and not trend.empty and "price" in trend.columns:
                    fig = go.Figure()
                    base = float(trend["price"].iloc[0])
                    series = (trend["price"] / base - 1.0) * 100.0 if base else trend["price"]
                    col = "#dc2626" if series.iloc[-1] >= 0 else "#16a34a"
                    fig.add_trace(go.Scatter(x=trend["time"], y=trend["price"], mode="lines",
                                            name="分时", line=dict(color=col, width=1.6)))
                    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      font=dict(color="#9ca3af", size=11),
                                      xaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
                                      yaxis=dict(gridcolor="rgba(0,0,0,0.06)", title="价格"),
                                      hovermode="x unified")
                    _plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.caption("🕐 暂无分时数据（非交易时段，或该市场无免费分时源）")
            except Exception as e:  # noqa: BLE001
                st.caption(f"⚠️ 分时加载失败: {e}")

            # K 线
            st.markdown(f'<div class="section-title">📈 K 线 · {selected_period}</div>', unsafe_allow_html=True)
            try:
                hist = _fetch_price_history(sym, period=period_map[selected_period], interval="1d")
                if not hist.empty and len(hist) > 5:
                    _draw_kline(hist, sym, with_volume=True)
                else:
                    st.caption(f"📉 {sym} 数据不足")
            except Exception as e:  # noqa: BLE001
                st.caption(f"⚠️ K线加载失败: {e}")

            # 实时盘口（五档）+ 成交明细（新维度：富途/东财）
            is_cn_hk = sym.endswith((".SS", ".SZ", ".HK"))
            if is_cn_hk:
                st.markdown('<div class="section-title">📊 实时盘口 & 成交明细</div>', unsafe_allow_html=True)
                oc1, oc2 = st.columns(2)
                with oc1:
                    st.markdown("##### 🔢 五档盘口")
                    try:
                        book = U.fetch_order_book(sym)
                        if book.get("ok"):
                            bids = book.get("bids", [])
                            asks = book.get("asks", [])
                            rows = ""
                            for i in range(min(5, len(asks))):
                                p, v = asks[4 - i]
                                rows += (f'<div style="display:flex;justify-content:space-between;font-size:12px;padding:2px 0;">'
                                         f'<span style="color:#16a34a;font-weight:600;">卖{5-i}</span>'
                                         f'<span>{p:.2f}</span><span style="color:var(--text-dim);">{_rt_fmt_volume(v)}</span></div>')
                            rows += '<div style="height:1px;background:var(--border);margin:4px 0;"></div>'
                            for i in range(min(5, len(bids))):
                                p, v = bids[i]
                                rows += (f'<div style="display:flex;justify-content:space-between;font-size:12px;padding:2px 0;">'
                                         f'<span style="color:#dc2626;font-weight:600;">买{i+1}</span>'
                                         f'<span>{p:.2f}</span><span style="color:var(--text-dim);">{_rt_fmt_volume(v)}</span></div>')
                            st.markdown(
                                f'<div class="card" style="font-size:12px;">{rows}'
                                f'<div style="font-size:10px;color:var(--text-dim);margin-top:6px;">源: {book.get("source","")}</div></div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.caption("📡 盘口暂不可用（非交易时段或源校验未通过）")
                    except Exception as e:  # noqa: BLE001
                        st.caption(f"⚠️ 盘口加载失败: {e}")
                with oc2:
                    st.markdown("##### 🧾 当日成交明细（逐笔）")
                    try:
                        tick = U.fetch_tick_detail(sym, count=20)
                        if tick.get("ok"):
                            ticks = tick.get("ticks", [])
                            drows = ""
                            for t in ticks[:20]:
                                dirc = t.get("direction", "NEUTRAL")
                                dc = "#dc2626" if dirc == "BUY" else ("#16a34a" if dirc == "SELL" else "#9ca3af")
                                dl = {"BUY": "买", "SELL": "卖", "NEUTRAL": "—"}.get(dirc, "—")
                                drows += (f'<div style="display:flex;justify-content:space-between;font-size:11px;padding:1px 0;">'
                                          f'<span style="color:var(--text-dim);">{t.get("time","")[-8:]}</span>'
                                          f'<span>{t.get("price",0):.2f}</span>'
                                          f'<span style="color:var(--text-dim);">{_rt_fmt_volume(t.get("volume",0))}</span>'
                                          f'<span style="color:{dc};font-weight:600;">{dl}</span></div>')
                            st.markdown(
                                f'<div class="card" style="font-size:11px;max-height:220px;overflow:auto;">{drows}'
                                f'<div style="font-size:10px;color:var(--text-dim);margin-top:6px;">源: {tick.get("source","")}</div></div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.caption("📡 成交明细暂不可用（非交易时段或源校验未通过）")
                    except Exception as e:  # noqa: BLE001
                        st.caption(f"⚠️ 成交明细加载失败: {e}")
            else:
                st.caption("ℹ️ 美股无免费盘口/逐笔数据源（盘口与成交明细仅 A股/港股）。")

            # 资金面
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

            # 维科夫吸筹结构（技术面）
            st.markdown("#### 🧬 维科夫吸筹结构")
            try:
                wh2 = _fetch_price_history(sym, period="6mo", interval="1d")
                if wh2 is not None and not wh2.empty and len(wh2) >= 120:
                    w = screener.detect_wyckoff_events(wh2)
                    if w.get("ok"):
                        conf = w["confidence"]
                        conf_color = "#16a34a" if conf >= 0.71 else ("#f59e0b" if conf >= 0.43 else ("#3b82f6" if conf >= 0.14 else "#9ca3af"))
                        st.markdown(
                            f'<div class="card" style="text-align:center;border-top:3px solid {conf_color};">'
                            f'<div style="display:flex;justify-content:center;">{_donut(conf * 100, size=62, label="置信度")}</div>'
                            f'<div style="font-size:13px;font-weight:700;margin-top:4px;">{w["stage"]}</div>'
                            f'<div style="font-size:11px;color:var(--text-dim);margin-top:2px;">{w["event_count"]}/7 事件 · {w["phase"]}</div>'
                            f'<div style="font-size:11px;margin-top:6px;text-align:left;color:var(--text);">{w["summary"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        if w["events"]:
                            st.caption("事件序列: " + " → ".join(e["event"] for e in w["events"]))
                        _exp2 = screener.explain_wyckoff(w)
                        st.markdown(
                            f'<div style="font-size:11.5px;line-height:1.6;color:var(--text-dim);margin-top:6px;padding:6px 8px;background:#f0f9ff;border-radius:6px;border-left:3px solid #0284c7;">'
                            f'<b style="color:#0369a1;">AI 解读：</b>{_exp2["confidence_meaning"]}<br>'
                            f'{_exp2["stage_meaning"]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption(f"维科夫检测不可用：{w.get('error','')}")
                else:
                    st.caption("K线数据不足 120 根，无法检测吸筹结构")
            except Exception as e:  # noqa: BLE001
                st.caption(f"维科夫检测失败: {e}")

        # ---------- Subtab 2：策略与风险 ----------
        with t2:
            # 交易策略（sniper + R 倍数分批减仓 + 杠杆）
            st.markdown('<div class="section-title">🎯 交易策略</div>', unsafe_allow_html=True)
            sniper = card.get("sniper", {}) or r.get("sniper", {})
            if sniper:
                c1, c2, c3, c4 = st.columns(4)
                for col, (label, val, color) in zip([c1, c2, c3, c4], [
                    ("理想买入", sniper.get("ideal_buy", "—"), "#2563eb"),
                    ("二次加仓", sniper.get("second_buy", "—"), "#16a34a"),
                    ("止损位", sniper.get("stop_loss", "—"), "#dc2626"),
                    ("止盈逻辑", "分批减仓", "#f59e0b"),
                ]):
                    with col:
                        st.markdown(f"<div class='card' style='text-align:center;border-top:3px solid {color};padding:10px;'><h4>{label}</h4><div style='font-size:14px;font-weight:700;color:{color};'>{val}</div></div>", unsafe_allow_html=True)

                try:
                    import risk as _risk
                    _buy = _risk._num(sniper.get("ideal_buy") or r.get("收盘价"))
                    _stop = _risk._num(sniper.get("stop_loss"))
                    if _buy is not None and _stop is not None and _stop < _buy:
                        _atr_v = safe_float(r.get("ATR"))
                        plan = _risk.r_multiple_plan(_buy, _stop, current_price=price, atr=_atr_v or None)
                        if plan.get("ok"):
                            st.markdown("##### 📐 R 倍数分批止盈（垫厚利润）")
                            st.caption("原逻辑：单一止盈目标一次性了结，容易卖飞或回吐。新逻辑：以入场风险 R 为单位，分三批兑现，边涨边垫高止损。")
                            _s1, _s2, _s3 = plan["stages"]
                            _trail = _s3.get("price")
                            _ma20 = safe_float(r.get("MA20"))
                            _trail_note = f"（≈ MA20 {_ma20:.2f}）" if _ma20 and _trail and _ma20 > _trail else ""
                            _row = (
                                f"<div class='card' style='padding:10px 12px;font-size:12px;'>"
                                f"<div style='display:flex;justify-content:space-between;margin-bottom:6px;'>"
                                f"<span>入场风险 R</span><b>{plan['risk_pct']:.1f}% <span style='color:var(--text-dim);font-size:10px;'>(${plan['risk_r']:.2f})</span></b></div>"
                                f"<div style='display:flex;justify-content:space-between;margin-bottom:6px;color:var(--text-dim);'>"
                                f"<span>当前盈亏</span><b style='color:{color_for_change(plan.get('current_pnl_pct', 0))};'>{plan.get('current_pnl_pct', 0):+.2f}% ({plan.get('current_pnl_R', 0):+.2f}R)</b></div>"
                                f"<div style='display:flex;justify-content:space-between;margin-bottom:6px;'>"
                                f"<span>① +1R ({_s1['price_pct']:+.1f}%)</span><b style='color:#2563eb;'>{_s1['price']:.2f} · 减仓 1/3，止损上移至成本价</b></div>"
                                f"<div style='display:flex;justify-content:space-between;margin-bottom:6px;'>"
                                f"<span>② +2R ({_s2['price_pct']:+.1f}%)</span><b style='color:#16a34a;'>{_s2['price']:.2f} · 再减 1/3，止损上移至 +1R</b></div>"
                                f"<div style='display:flex;justify-content:space-between;margin-bottom:6px;'>"
                                f"<span>③ 剩余 1/3</span><b style='color:#f59e0b;'>{_s3['action']}</b></div>"
                                f"<div style='display:flex;justify-content:space-between;margin-bottom:6px;background:#fffbeb;padding:4px 6px;border-radius:6px;'>"
                                f"<span style='color:#92400e;'>清仓触发价</span><b style='color:#b45309;'>{_trail:.2f}{_trail_note} · 回撤至此或跌破 MA20 即清仓</b></div>"
                                f"<div style='font-size:11px;color:var(--accent);font-weight:600;margin-top:4px;'>▶ {plan.get('current_stage','')}</div>"
                                f"</div>"
                            )
                            st.markdown(_row, unsafe_allow_html=True)
                            st.caption("具体减仓价位点已标注：① 到 +1R 价减 1/3 并保本；② 到 +2R 价再减 1/3 并锁定 +1R；③ 剩余移动止损，回撤至清仓触发价或破 MA20 出局。")
                except Exception as e:  # noqa: BLE001
                    logger.debug("R 倍数计划生成失败: %s", e)

            # 明日观察位（规则引擎 + AI 研判）
            _render_tomorrow_watch(sym)

            # 杠杆强平监控
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

        # ---------- Subtab 3：AI研判（新闻解读 + 下周走势 + 公告/龙虎榜）----------
        with t3:
            # 0) 宏观 + 政策新闻
            macro_news_items = (NEWS_DATA.get("macro", []) if isinstance(NEWS_DATA, dict) else []) or []
            policy_news_items = (NEWS_DATA.get("policy", []) if isinstance(NEWS_DATA, dict) else []) or []
            if not macro_news_items and not policy_news_items:
                with st.spinner("实时拉取宏观/政策新闻…"):
                    try:
                        macro_news_items = U.fetch_yahoo_rss(query="Federal Reserve")
                        policy_news_items = U.fetch_policy_news_free(top_n=6)
                    except Exception:  # noqa: BLE001
                        pass
            if macro_news_items or policy_news_items:
                m1, m2 = st.columns(2)
                with m1:
                    st.markdown("##### 🌍 宏观新闻")
                    for n in (macro_news_items or [])[:5]:
                        st.markdown(
                            f"<div class='news-card' style='padding:6px 10px;'>"
                            f"<a href='{n.get('link','#')}' target='_blank' class='title' style='font-size:12px;'>{n.get('title','')}</a>"
                            f"<div class='meta' style='font-size:10px;'>{n.get('source','')} · {n.get('date','')}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                with m2:
                    st.markdown("##### 🏛️ 政策新闻")
                    for n in (policy_news_items or [])[:5]:
                        st.markdown(
                            f"<div class='news-card' style='padding:6px 10px;'>"
                            f"<a href='{n.get('link','#')}' target='_blank' class='title' style='font-size:12px;'>{n.get('title','')}</a>"
                            f"<div class='meta' style='font-size:10px;'>{n.get('source','')} · {n.get('date','')}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                st.divider()

            # 1) 个股新闻 + 近2日启发式解读
            sym_news = []
            if isinstance(NEWS_DATA, dict):
                sym_news = (NEWS_DATA.get("stocks", {}) or {}).get(sel, []) if isinstance(NEWS_DATA.get("stocks"), dict) else []
            if not sym_news:
                with st.spinner(f"实时拉取 {sel} 新闻…"):
                    if sel.endswith((".SS", ".SZ")):
                        _em = U.fetch_eastmoney_stock_news(sel)
                        sym_news = list(_em or [])
                        if not sym_news:
                            sym_news = U.fetch_10jqka_news(top_n=5) or U.fetch_xueqiu_news(top_n=5)
                    else:
                        sym_news = U.fetch_yahoo_rss(ticker=sel)
                        if not sym_news and not sel.endswith(".HK"):
                            sym_news = U.fetch_stocktwits(sel, limit=8)

            interp = U.interpret_news(sel, sym_news, within_days=2)
            n_l, n_r = st.columns([1, 1])
            with n_l:
                st.markdown("##### 🗞️ 新闻列表（近2日优先）")
                if sym_news:
                    for n in sym_news[:6]:
                        st.markdown(
                            f"<div class='news-card'>"
                            f"<a href='{n.get('link', '#')}' target='_blank' class='title'>{n.get('title','')}</a>"
                            f"<div class='meta'>{n.get('source','')} · {n.get('date','')}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("暂无新闻（SerpApi/免费源 均未拉到）")
            with n_r:
                st.markdown("##### 🔍 近2日新闻解读（启发式）")
                tone_color = {"看多": "#16a34a", "看空": "#dc2626", "中性": "#f59e0b", "信息不足": "#9ca3af"}.get(interp["tone"], "#9ca3af")
                st.markdown(
                    f"<div class='ai-box' style='border-left-color:{tone_color};'>"
                    f"<div class='label' style='color:{tone_color};'>综合情绪：{interp['tone']}（净分 {interp['score']:+.2f}）</div>"
                    f"{interp['summary']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if interp["positives"]:
                    st.success("**偏多**: " + " · ".join(interp["positives"][:3]))
                if interp["negatives"]:
                    st.error("**偏空**: " + " · ".join(interp["negatives"][:3]))
                st.caption("⚠️ 以上为基于标题关键词的机器启发式解读，非投顾建议。")

            # 2) 下周走势预测
            st.markdown("##### 🔮 下周走势预测")
            pred_block = (PREDICTIONS_DATA.get("stocks", {}) or {}).get(sel, {}) if isinstance(PREDICTIONS_DATA, dict) else {}
            if pred_block and pred_block.get("prediction"):
                pred_md = pred_block["prediction"]
                st.markdown(
                    f'<div class="ai-box" style="border-left-color:#7c3aed;font-size:13px;line-height:1.7;">{pred_md.replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True,
                )
                tech = pred_block.get("technical", {})
                if tech:
                    st.caption(
                        f"基线技术: ${tech.get('close', 0):.2f} · "
                        f"RSI {tech.get('rsi', 0):.0f} · "
                        f"ATR {tech.get('atr', 0):.2f}"
                    )
            else:
                st.caption("暂无预测（运行 `python stock_dashboard.py --predictions` 生成）")

            # 3) 公告 & 龙虎榜（新维度：巨潮官方公告 + 东财龙虎榜，A股）
            if sym.endswith((".SS", ".SZ")):
                st.markdown("##### 📋 个股公告（巨潮资讯网·官方）")
                try:
                    anns = U.fetch_cninfo_stock_announcements(sym, top_n=6)
                    if anns:
                        for a in anns[:6]:
                            st.markdown(
                                f"<div class='news-card'><a href='{a.get('link','#')}' target='_blank' class='title' style='font-size:12px;'>{a.get('title','')}</a>"
                                f"<div class='meta' style='font-size:10px;'>{a.get('source','')} · {a.get('date','')}</div></div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("暂无公告")
                except Exception as e:  # noqa: BLE001
                    st.caption(f"⚠️ 公告加载失败: {e}")
            else:
                st.caption("ℹ️ 个股公告为 A股专属（巨潮官方源）。")

            st.markdown("##### 🐯 龙虎榜（全市场·上一交易日）")
            try:
                dt_list = U.fetch_dragon_tiger(top_n=15)
                if dt_list:
                    dt_rows = "".join(
                        f"<div style='display:flex;justify-content:space-between;font-size:12px;padding:3px 0;border-bottom:1px solid var(--border);'>"
                        f"<span><b>{d.get('name','')}</b> <span style='color:var(--text-dim);font-size:10px;'>{d.get('code','')}</span></span>"
                        f"<span style='color:{color_for_change(safe_float(d.get('pct')))};font-weight:600;'>{safe_float(d.get('pct')):+.2f}%</span>"
                        f"<span style='color:var(--text-dim);'>净买 {_rt_fmt_volume(d.get('net_buy') or 0)}</span>"
                        f"</div>"
                        for d in dt_list[:15]
                    )
                    st.markdown(f"<div class='card' style='font-size:12px;'>{dt_rows}</div>", unsafe_allow_html=True)
                else:
                    st.caption("暂无龙虎榜数据")
            except Exception as e:  # noqa: BLE001
                st.caption(f"⚠️ 龙虎榜加载失败: {e}")

    with right:
        if not sel:
            return
        sym = sel
        _sub = work_df[work_df["symbol"] == sym]
        if _sub.empty:
            return
        r = _sub.iloc[0]
        card = CARDS_MAP.get(sym, {})

        # 评分（DSA 圆环）
        st.markdown("#### 🎯 AI 评分")
        try:
            score = float(card.get("score") or 0)
        except (TypeError, ValueError):
            score = 0.0
        op = card.get("operation", "—")
        tr = card.get("trend", "—")
        st.markdown(
            f"<div class='card' style='text-align:center;'>"
            f"<div style='display:flex;justify-content:center;'>{_donut(score, size=86, label='综合评分')}</div>"
            f"<div style='font-size:13px;color:var(--text-dim);margin-top:8px;'>操作: <b>{op}</b> · 趋势: <b>{tr}</b></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # v3.0 多因子评分（选股三层架构 · 第二层，纯计算）
        st.markdown("#### 📐 多因子评分")
        try:
            wh = _fetch_price_history(sym, period="6mo", interval="1d")
            if wh is not None and not wh.empty and len(wh) >= 30:
                mf = screener.score_multi_factor(wh)
                if mf.get("ok"):
                    f = mf["factors"]
                    pass_ = mf["threshold_pass"]
                    _mf_rows = (
                        f'<div class="card" style="padding:10px 12px;font-size:12px;">'
                        f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;">'
                        f'<span>综合分（60 阈值）</span><b style="color:{"#16a34a" if pass_ else "#dc2626"};">{mf["score"]} · {"通过" if pass_ else "未过"}</b></div>'
                        f'<div class="detail-grid">'
                        f'<span class="k">均线 tech</span><span class="v">{f["tech"]}</span>'
                        f'<span class="k">POC 偏离</span><span class="v">{f["poc_dev_pct"]:+.1f}% ({f["poc"]})</span>'
                        f'<span class="k">量能 vol</span><span class="v">{f["vol"]}</span>'
                        f'<span class="k">相对强度 rs</span><span class="v">{f["rs"]}</span>'
                        f'<span class="k">板块联动</span><span class="v">{f["sector"]}</span>'
                        f'<span class="k">龙头强度</span><span class="v">{f["leader"]}</span>'
                        f'</div></div>'
                    )
                    st.markdown(_mf_rows, unsafe_allow_html=True)
                else:
                    st.caption(f"多因子评分不可用：{mf.get('error','')}")
            else:
                st.caption("K线数据不足 30 根，无法评分")
        except Exception as e:  # noqa: BLE001
            st.caption(f"多因子评分失败: {e}")

        # v3.0 维科夫吸筹结构（选股三层架构 · 第一层）
        st.markdown("#### 🧬 维科夫吸筹")
        try:
            wh2 = _fetch_price_history(sym, period="6mo", interval="1d")
            if wh2 is not None and not wh2.empty and len(wh2) >= 120:
                w = screener.detect_wyckoff_events(wh2)
                if w.get("ok"):
                    conf = w["confidence"]
                    conf_color = "#16a34a" if conf >= 0.71 else ("#f59e0b" if conf >= 0.43 else ("#3b82f6" if conf >= 0.14 else "#9ca3af"))
                    st.markdown(
                        f'<div class="card" style="text-align:center;border-top:3px solid {conf_color};">'
                        f'<div style="display:flex;justify-content:center;">{_donut(conf * 100, size=62, label="置信度")}</div>'
                        f'<div style="font-size:13px;font-weight:700;margin-top:4px;">{w["stage"]}</div>'
                        f'<div style="font-size:11px;color:var(--text-dim);margin-top:2px;">{w["event_count"]}/7 事件 · {w["phase"]}</div>'
                        f'<div style="font-size:11px;margin-top:6px;text-align:left;color:var(--text);">{w["summary"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if w["events"]:
                        st.caption("事件序列: " + " → ".join(e["event"] for e in w["events"]))
                    # 需求6：内嵌 AI 解读（置信度 + 事件序列含义 + 操作指引）
                    _exp2 = screener.explain_wyckoff(w)
                    st.markdown(
                        f'<div style="font-size:11.5px;line-height:1.6;color:var(--text-dim);margin-top:6px;padding:6px 8px;background:#f0f9ff;border-radius:6px;border-left:3px solid #0284c7;">'
                        f'<b style="color:#0369a1;">AI 解读：</b>{_exp2["confidence_meaning"]}<br>'
                        f'{_exp2["stage_meaning"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button("🤖 生成 AI 研判（DeepSeek）", key=f"wyc_ai_{sym}", use_container_width=True):
                        with st.spinner("调用 DeepSeek…"):
                            _mf2 = screener.score_multi_factor(wh2) if (wh2 is not None and not wh2.empty) else {"score": "—", "bias": "—", "threshold_pass": False, "factors": {}}
                            _prompt2 = screener.build_screener_narrative_prompt(w, _mf2, sym, U.STOCK_NAMES.get(sym, ""))
                            _ai2 = U._call_llm(
                                messages=[
                                    {"role": "system", "content": "你是量化研究助理，输出简体中文，只解释给定数据，不编造价格/消息/目标价。"},
                                    {"role": "user", "content": _prompt2},
                                ],
                                prefer="deepseek",
                            )
                        if _ai2:
                            st.success(_ai2)
                        else:
                            st.info("未配置 DEEPSEEK_API_KEY（或调用失败），已降级展示上方结构化解读。")
                else:
                    st.caption(f"维科夫检测不可用：{w.get('error','')}")
            else:
                st.caption("K线数据不足 120 根，无法检测吸筹结构")
        except Exception as e:  # noqa: BLE001
            st.caption(f"维科夫检测失败: {e}")
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

        # 新闻列表与「近2日解读」已移至三栏下方的全宽区（见本函数末尾）

        # ===== v2.1 新增：Vol/OI PCR（来自 yfinance 期权）=====
        st.markdown("#### 📊 期权 Vol/OI PCR")
        pcr = PCR_DATA.get(sym, {}) if isinstance(PCR_DATA, dict) else {}
        if pcr and (pcr.get("vol_pcr") is not None or pcr.get("oi_pcr") is not None):
            vol_pcr = pcr.get("vol_pcr")
            oi_pcr = pcr.get("oi_pcr")
            # 颜色：PCR > 1.0 = 看空；< 0.7 = 看多
            def pcr_color(v):
                if v is None: return "#9ca3af"
                if v > 1.0: return "#16a34a"  # 偏高（看空保护）
                if v < 0.7: return "#dc2626"  # 偏低（看多情绪）
                return "#f59e0b"
            def pcr_fmt(v):
                # 修复：f-string 格式说明符不能与条件表达式直接组合
                # （原写法 `{v:.2f if v is not None else "—"}` 运行时抛 ValueError）
                try:
                    return f"{v:.2f}" if v is not None else "—"
                except (TypeError, ValueError):
                    return "—"
            st.markdown(
                f'<div class="card" style="font-size:13px;">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
                f'<span>到期日</span><b>{pcr.get("expiry","—")}</b></div>'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
                f'<span>Vol PCR</span><b style="color:{pcr_color(vol_pcr)};font-size:18px;">{pcr_fmt(vol_pcr)}</b></div>'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
                f'<span>OI PCR</span><b style="color:{pcr_color(oi_pcr)};font-size:18px;">{pcr_fmt(oi_pcr)}</b></div>'
                f'<div style="display:flex;justify-content:space-between;font-size:11px;color:var(--text-dim);">'
                f'<span>Call/Put Vol</span><span>{pcr.get("call_volume", 0):,} / {pcr.get("put_volume", 0):,}</span></div>'
                f'<div style="display:flex;justify-content:space-between;font-size:11px;color:var(--text-dim);">'
                f'<span>Call/Put OI</span><span>{pcr.get("call_oi", 0):,} / {pcr.get("put_oi", 0):,}</span></div>'
                f'<div style="display:flex;justify-content:space-between;font-size:11px;color:var(--text-dim);margin-top:4px;">'
                f'<span>IV (C/P)</span><span>{pcr.get("iv_call", "—")} / {pcr.get("iv_put", "—")}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            # 解读
            v = pcr.get("vol_pcr", 0) or 0
            if v > 1.2:
                st.caption("📉 Vol PCR > 1.2：Put 成交活跃，看空/对冲情绪重")
            elif v > 0.8:
                st.caption("➡️ 0.8-1.2：多空均衡")
            else:
                st.caption("📈 Vol PCR < 0.8：Call 主导，市场偏多")
        else:
            # 按错误类型区分提示：数据源不支持 / 无到期日 / 链为空 / 临时失败 / 数据未生成
            pcr_err = (pcr or {}).get("error", "")
            if pcr_err == "market_not_supported":
                st.caption("ℹ️ 该标的市场暂无期权链数据（当前数据源不覆盖港股/A股个股期权），Vol/OI PCR 仅美股标的有。")
            elif pcr_err in ("no_expiry", "empty_chain"):
                st.caption("ℹ️ 该标的近期无可用期权数据（无到期日或期权链为空）。")
            elif pcr_err:
                st.caption(f"⚠️ 期权数据获取失败：{pcr_err}（可稍后重试）")
            else:
                st.caption("暂无期权数据（由每日数据管线生成 data/options_pcr.json）")

        # 下周走势预测 + 新闻解读 + 公告/龙虎榜 已并入中栏「🧠 AI研判」Subtab

        # 市场上下文
        st.markdown("#### 🌍 市场上下文")
        sectors = card.get("sectors", [])
        if sectors:
            st.markdown("**板块**: " + " · ".join(sectors))
        st.caption(f"VIX {safe_float(MACRO_DF.iloc[0].get('VIX'), 0) if MACRO_DF is not None and not MACRO_DF.empty else 0:.2f} · 10Y {safe_float(MACRO_DF.iloc[0].get('10年期美债收益率'), 0) if MACRO_DF is not None and not MACRO_DF.empty else 0:.2f}%")

    


# 页面：跨资产对比（修复版）
# ---------------------------------------------------------------------------
def page_cross_asset():
    st.markdown('<div class="section-title"><span class="accent">📊</span>历史跨资产对比（标普500 / SOX / 10Y美债 / Mag 7）</div>', unsafe_allow_html=True)

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
    _plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # 数据表
    with st.expander("📋 查看原始数据"):
        st.dataframe(pd.DataFrame({n: s.tail(252) for n, s in hist_data.items()}), use_container_width=True, height=300)

    # ===== v2.4 新增：VIX vs VXN 恐慌指数对比（与上图时间维度一致）=====
    st.markdown('<div class="section-title"><span class="accent">😱</span>恐慌指数对比：VIX (标普) vs VXN (纳指)</div>', unsafe_allow_html=True)
    st.caption("✅ 与上图用同一个时间区间 + auto_adjust=True；用于观察美债暴增/股市急跌时恐慌情绪联动。")

    with st.spinner("拉取 VIX / VXN 历史…"):
        panic_data = U.fetch_panic_history(period_str)

    if panic_data:
        # 找两条曲线的公共时间区间
        all_idx = None
        for s in panic_data.values():
            all_idx = s.index if all_idx is None else all_idx.intersection(s.index)
        if all_idx is not None and len(all_idx) > 0:
            common_start, common_end = all_idx.min(), all_idx.max()

            fig2 = go.Figure()
            # VIX 主轴
            if "VIX" in panic_data:
                vix_s = panic_data["VIX"].loc[common_start:common_end]
                fig2.add_trace(go.Scatter(
                    x=vix_s.index, y=vix_s.values, name="VIX (标普恐慌)",
                    line=dict(color="#f59e0b", width=2),
                    hovertemplate="%{y:.2f}<extra>VIX</extra>",
                ))
            # VXN 副轴 (VXN 通常比 VIX 高 1-5 点)
            if "VXN" in panic_data:
                vxn_s = panic_data["VXN"].loc[common_start:common_end]
                fig2.add_trace(go.Scatter(
                    x=vxn_s.index, y=vxn_s.values, name="VXN (纳指恐慌)",
                    line=dict(color="#dc2626", width=2, dash="dot"),
                    yaxis="y2",
                    hovertemplate="%{y:.2f}<extra>VXN</extra>",
                ))
            # 警戒带：VIX>30 阴影
            fig2.add_hrect(y0=30, y1=80, line_width=0, fillcolor="rgba(220,38,38,0.06)", layer="below")
            fig2.add_hline(y=30, line=dict(color="#dc2626", width=1, dash="dash"), annotation_text="恐慌阈值 30", annotation_position="top left")

            fig2.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="white",
                font=dict(color="#6b7280", size=11),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
                xaxis=dict(gridcolor="#f1f5f9"),
                yaxis=dict(title="VIX (标普恐慌)", gridcolor="#e5e7eb"),
                yaxis2=dict(title="VXN (纳指恐慌)", overlaying="y", side="right", gridcolor="rgba(0,0,0,0)"),
                hovermode="x unified",
            )
            _plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

            # 关键统计
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if "VIX" in panic_data:
                    last_vix = float(panic_data["VIX"].iloc[-1])
                    st.metric("VIX 最新", f"{last_vix:.2f}", delta=f"{U.emoji_for_panic(last_vix)}")
            with c2:
                if "VXN" in panic_data:
                    last_vxn = float(panic_data["VXN"].iloc[-1])
                    st.metric("VXN 最新", f"{last_vxn:.2f}", delta=f"{U.emoji_for_panic(last_vxn)}")
            with c3:
                if "VIX" in panic_data and "VXN" in panic_data:
                    spread = float(panic_data["VXN"].iloc[-1]) - float(panic_data["VIX"].iloc[-1])
                    st.metric("VXN-VIX 利差", f"{spread:+.2f}", help="纳指波动溢价")
            with c4:
                if "VIX" in panic_data:
                    import numpy as _np
                    arr = panic_data["VIX"].dropna().values
                    pct_days = float((arr > 30).sum() / max(1, len(arr)) * 100)
                    st.metric("VIX>30 占比", f"{pct_days:.1f}%", help=f"区间内 {int((arr>30).sum())} / {len(arr)} 个交易日")

            with st.expander("📋 查看 VIX/VXN 原始数据"):
                st.dataframe(pd.DataFrame({n: s.loc[common_start:common_end] for n, s in panic_data.items()}), use_container_width=True, height=260)
        else:
            st.info("VIX/VXN 历史数据为空")
    else:
        st.info("VIX/VXN 拉取失败，可稍后重试")

    # ===== v2.4 备选：宏观风险联动（美债规模 vs 杠杆 vs 恐慌指数 副图）=====
    with st.expander("➕ 进阶: 美债规模 / 杠杆 联动（需 FRED_API）"):
        if not _get_secret("FRED_API"):
            st.caption("⚠️ 未配置 FRED_API，跳过美债/杠杆联动图。可在 Streamlit Cloud Secrets 填 `FRED_API` 后启用。")
        else:
            st.caption("✅ 已配置 FRED_API，可叠加：U.S. National Debt (GFDEBTN)、FINRA Margin Debt (MDEBT)、NFCI Leverage。")
            try:
                linkage = U.fetch_fred_linkage_series(days=400)
                if linkage.get("ok") and len(linkage.get("dates", [])) >= 2:
                    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
                    fig3.add_trace(
                        go.Scatter(x=linkage["dates"], y=linkage["debt_pct"],
                                   name="美债规模 GFDEBTN 变化%", line=dict(color="#1e3a8a", width=2)),
                        secondary_y=False,
                    )
                    fig3.add_trace(
                        go.Scatter(x=linkage["dates"], y=linkage["margin_pct"],
                                   name="融资余额 MDEBT 变化%", line=dict(color="#ea580c", width=2)),
                        secondary_y=False,
                    )
                    fig3.add_trace(
                        go.Scatter(x=linkage["dates"], y=linkage["nfci"],
                                   name="NFCI 杠杆指数 (右轴)", line=dict(color="#7c3aed", width=2, dash="dot")),
                        secondary_y=True,
                    )
                    fig3.update_layout(
                        title="美债规模 vs 杠杆 联动（近 400 交易日，债务/融资=相对起点变化%，NFCI=原值）",
                        height=380, margin=dict(l=10, r=10, t=46, b=10),
                        legend=dict(orientation="h", y=1.08),
                        hovermode="x unified",
                    )
                    fig3.update_yaxes(title_text="变化 %（起点=0）", secondary_y=False)
                    fig3.update_yaxes(title_text="NFCI 杠杆指数", secondary_y=True)
                    _plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.caption(f"⚠️ 联动图拉取失败：{linkage.get('error', '未知错误')}")
            except Exception as e:  # noqa: BLE001
                st.caption(f"⚠️ 联动图渲染失败：{e}")


# ---------------------------------------------------------------------------
# 页面：新闻中心
# ---------------------------------------------------------------------------
def page_news_center():
    st.markdown('<div class="section-title"><span class="accent">📰</span>新闻与消息中心</div>', unsafe_allow_html=True)

    # ===== v2.1：多源状态面板 =====
    sources_status = []
    if SERPAPI_KEY: sources_status.append(("SerpApi (付费)", "✅", "宏观/政策/个股"))
    else: sources_status.append(("SerpApi (付费)", "❌", "需 SERPAPI secret"))
    if FINNHUB_KEY: sources_status.append(("Finnhub", "✅", "美股个股新闻"))
    else: sources_status.append(("Finnhub", "⚠️ 可选", "60 calls/min 免费"))
    if NEWSAPI_KEY: sources_status.append(("NewsAPI", "✅", "宏观新闻"))
    else: sources_status.append(("NewsAPI", "⚠️ 可选", "100/day 免费"))
    sources_status.append(("Yahoo Finance RSS", "✅", "始终免费"))
    sources_status.append(("Stocktwits API", "✅", "始终免费（散户情绪）"))
    sources_status.append(("东方财富网", "✅", "始终免费（中文宏观）"))

    # 新闻按「影响股价」相关性过滤后，再归类为 宏观 / 政策 / 自选股（过滤逻辑见 utils.fetch_all_news_multi_source）

    if st.button("🔄 立即抓取一次新闻（多源）", use_container_width=True):
        with st.spinner("多源抓取中…"):
            symbols = STOCKS_DF["symbol"].tolist() if STOCKS_DF is not None else []
            res = U.fetch_all_news_multi_source(
                symbols=symbols,
                serpapi_key=SERPAPI_KEY,
                finnhub_key=FINNHUB_KEY,
                newsapi_key=NEWSAPI_KEY,
                out_path=DATA_DIR / "news.json",
            )
            st.session_state["_news_res"] = res
            used = ", ".join(res.get("sources_used", [])) or "无"
            st.success(
                f"✅ 抓取完成 · 使用源: {used} · "
                f"宏观 {len(res.get('macro', []))} / 政策 {len(res.get('policy', []))} / "
                f"个股 {sum(1 for v in res.get('stocks', {}).values() if v)}/{len(symbols)}"
            )

    res = st.session_state.get("_news_res") or NEWS_DATA

    if isinstance(res, dict):
        if "macro" in res or "policy" in res:
            tab_m, tab_p, tab_s = st.tabs(["🌍 宏观", "🏛️ 政策", "📊 自选股"])
            with tab_m:
                _show_news_list(res.get("macro", []))
            with tab_p:
                _show_news_list(res.get("policy", []))
            with tab_s:
                sym = st.selectbox("选择股票", list(res.get("stocks", {}).keys()) if res.get("stocks") else [])
                _show_news_list(res.get("stocks", {}).get(sym, []) if sym else [])
        else:
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
        ("FRED_API", FRED_KEY, "利率/债务/杠杆卡 (2Y·实际利率·Debt·Margin·NFCI)"),
        ("DEEPSEEK_API_KEY", DEEPSEEK_KEY, "AI 报告 / 下周预测 (DeepSeek)"),
        ("OPENROUTER_API_KEY", _get_secret("OPENROUTER_API_KEY"), "AI 报告 OpenRouter 兜底 (Claude 3.5 Sonnet)"),
        ("SERPAPI", SERPAPI_KEY, "新闻增强源（100/月免费，超限自动降级 RSS）"),
        ("FINNHUB_API", FINNHUB_KEY, "美股个股新闻（60/min 免费）"),
        ("NEWSAPI_KEY", NEWSAPI_KEY, "宏观新闻（100/day 免费）"),
        ("HITHINK_FINANCE_API_KEY", _get_secret("HITHINK_FINANCE_API_KEY"),
         "同花顺 Financial-API 实时行情 / K线（未配置时个股深度页 watch 走 yfinance fallback）"),
    ]
    for name, v, desc in rows:
        ok = "✅" if v else "❌"
        st.markdown(f"- {ok} **{name}**: {'已配置 (' + str(len(v)) + ' 字符)' if v else '未配置'} — {desc}")

    st.markdown("#### 🐍 Python 库状态")
    libs = [
        ("akshare", U.akshare_available(), "A股数据全景 / K线 / 涨跌榜"),
        ("fredapi", _check_lib("fredapi"), "FRED 4 个指标 (DGS2/GFDEBTN/MDEBT/NFCILEVERAGE)"),
        ("openai", _check_lib("openai"), "DeepSeek / OpenRouter 调用"),
        ("feedparser", _check_lib("feedparser"), "RSS 新闻源"),
    ]
    for name, ok, desc in libs:
        st.markdown(f"- {'✅' if ok else '❌'} **{name}**: {'已安装' if ok else '未安装 — pip install ' + name} — {desc}")

    st.divider()
    st.markdown("#### 📁 data/ 目录文件状态")
    files = [
        "macro.csv", "stocks.csv", "sox.csv", "sp500.csv",
        "cards.json", "leverage_risk.json", "news.json",
        "report.md", "weekly_report.md", "morning_brief.md", "evening_recap.md",
        "extra_indicators.json (CI --extras)", "options_pcr.json (CI --extras)", "predictions.json (CI --predictions)",
    ]
    for f in files:
        p = DATA_DIR / f
        if p.exists():
            sz = p.stat().st_size
            mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            st.markdown(f"- ✅ **{f}** — {sz} bytes · updated {mtime}")
        else:
            tag = "(CI --extras)" if "extras" in f else ("(CI --predictions)" if "predictions" in f else "")
            st.markdown(f"- ❌ **{f}** — 缺失（CI 需执行 `python stock_dashboard.py --extras` / `--predictions` 子命令生成；主流程 run() 不生成这三个文件）")

    st.divider()
    st.markdown("#### 🛠️ 一键测试各项数据源")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🧪 测试 SerpApi", use_container_width=True):
            if not SERPAPI_KEY:
                st.error("SERPAPI 未配置")
            else:
                with st.spinner("测试中…"):
                    test = U.fetch_macro_news(SERPAPI_KEY, top_n=3)
                if test:
                    st.success(f"成功拉到 {len(test)} 条")
                else:
                    st.error("未拉到任何结果")
    with c2:
        if st.button("🧪 测试 Yahoo RSS (免费)", use_container_width=True):
            with st.spinner("测试中…"):
                test = U.fetch_yahoo_rss(query="Federal Reserve")
            if test:
                st.success(f"成功拉到 {len(test)} 条")
            else:
                st.error("未拉到（可能在中国大陆被墙）")
    with c3:
        if st.button("🧪 测试 Stocktwits (免费)", use_container_width=True):
            with st.spinner("测试中…"):
                test = U.fetch_stocktwits("AAPL")
            if test:
                st.success(f"成功拉到 {len(test)} 条")
            else:
                st.error("未拉到")

    st.divider()
    st.markdown("#### 💡 专业免费数据源（已调研 · 按优先级）")
    st.markdown("""
| 数据源 | 免费额度 | 用途 | 说明 |
|---|---|---|---|
| **FRED** | 120 req/min | 利率 / 债务 / 杠杆 / 通胀 | 已接入，最稳定 |
| **东方财富 push2** | 无限制 | A股/港股实时价 + 日K + 全球新闻 | 已接入（A股降级主源） |
| **Finnhub** | 60 req/min | 美股个股新闻 / 财报 | 注册即用：finnhub.io/register |
| **Tiingo** | 无硬限（低频） | 美股/ETF 历史 + 基本面 | 免费 token：tiingo.com |
| **Alpha Vantage** | 25 req/day | 个股期权链 PCR / 技术指标 | 已用 HISTORICAL_OPTIONS |
| **Twelve Data** | 800 req/day | 实时行情 + K线 + 技术指标 | 免费注册即用 |
| **Polygon.io** | 5 req/min | 美股历史 / 聚合行情 | 免费 tier 含历史数据 |
| **NewsAPI** | 100/day | 宏观新闻 | newsapi.org/register |
| **SerpApi** | 100/月 | 泛搜索新闻（易超限） | 超限后自动降级到 RSS 免费源 |
""")
    st.caption("SerpApi 免费额度(100/月)极易耗尽：代码已加错误检测 + Google News RSS / Yahoo RSS / 东财 / 同花顺 / 雪球多级免费兜底，未配置或超限时新闻板块不空。")


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
    # 技能建议①：K线最多显示最近 500 根（防大周期卡顿；指标在截断后计算）
    if hist is not None and len(hist) > 500:
        hist = hist.tail(500)
    if with_volume:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(
            x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"],
            name=sym,
            # 技能建议③：色盲双编码 — 涨=实心红填充 / 跌=空心绿（仅描边）
            increasing_line_color="#dc2626",
            decreasing_line_color="#16a34a",
            decreasing_fillcolor="rgba(0,0,0,0)",
        ), row=1, col=1)
        if len(hist) >= 20:
            ma20 = hist["Close"].rolling(20).mean()
            fig.add_trace(go.Scatter(x=hist.index, y=ma20, line=dict(color="#f59e0b", width=1.2), name="MA20"), row=1, col=1)
        if len(hist) >= 50:
            ma50 = hist["Close"].rolling(50).mean()
            fig.add_trace(go.Scatter(x=hist.index, y=ma50, line=dict(color="#2563eb", width=1.2), name="MA50"), row=1, col=1)
        colors = ["#dc2626" if c > 0 else "#16a34a" for c in hist["Close"].diff().fillna(0)]
        # 技能建议②：成交量柱 40% 透明度
        fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], name="Volume", marker_color=colors, opacity=0.4), row=2, col=1)
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
            name=sym,
            increasing_line_color="#dc2626",
            decreasing_line_color="#16a34a",
            decreasing_fillcolor="rgba(0,0,0,0)",
        ))
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False)
    _plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# 实时行情公共工具（被 Dashboard 结论区实时速览 + 个股深度实时盘口复用）
# 说明：不再设独立「实时行情监控」页面（需求3：实时数据并入 Dashboard 与个股深度页）。
# ---------------------------------------------------------------------------
_RT_DEFAULT_WATCHLIST = ["688809.SS", "300408.SZ", "0700.HK", "07709.HK", "NVDA", "TSLA"]


def _rt_normalize_input(raw: str) -> Optional[str]:
    """访客输入 → 内部代码：
      600519→600519.SS、688809→688809.SS、000001→000001.SZ、300750→300750.SZ、
      0700→0700.HK、00700→00700.HK、7709→07709.HK、NVDA→NVDA；.SH→.SS；原样透传带后缀代码。"""
    s = (raw or "").strip().upper().replace(" ", "")
    if not s:
        return None
    if s.endswith(".SH"):
        s = s[:-3] + ".SS"
    if s.endswith((".SS", ".SZ", ".HK")):
        return s
    if s.isdigit():
        if len(s) == 6:
            return s + (".SS" if s[0] in "69" else ".SZ")
        if len(s) == 5:
            return s.zfill(5) + ".HK"  # 7709 → 07709.HK
        if len(s) == 4:
            return s + ".HK"           # 0700 → 0700.HK
        return None
    return s  # 美股


def _rt_fmt_volume(v) -> str:
    """成交量/成交额 → 万/亿 中文缩写。"""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    if x is None or x != x:
        return "—"
    ax = abs(x)
    if ax >= 1e8:
        return f"{x / 1e8:.2f}亿"
    if ax >= 1e4:
        return f"{x / 1e4:.2f}万"
    return f"{x:.0f}"


def _rt_market_label(sym: str) -> str:
    if sym.endswith(".HK"):
        return "🇭🇰"
    if sym.endswith((".SS", ".SZ")):
        return "🇨🇳"
    return "🇺🇸"


# ---------------------------------------------------------------------------
# 使用说明（面向非专业用户）
# ---------------------------------------------------------------------------
def page_usage_guide() -> None:
    """📖 使用说明：通俗使用指南 + AI分析额度警示 + 功能现状 + 数据层构成。"""
    st.markdown('<div class="layer-badge concl">📖 使用说明 — 写给不太懂投资的你</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="conclusion-zone" style="font-size:13.5px;line-height:1.9;color:var(--text);">'
        '本系统是一个 <b>投资研究辅助工具</b>：把行情、新闻、宏观、技术指标汇总到一个面板里，'
        '帮你快速看清「今天市场怎么了、哪些股票值得关注、风险在哪」。<br>'
        '请记住：<b>它只做信息和数据整理，不构成任何投资建议</b>，最终买卖决策由你自己负责。'
        '</div>',
        unsafe_allow_html=True,
    )

    # 1) 怎么用
    st.markdown("### 🧭 一、三步上手（不用懂术语也能看）", unsafe_allow_html=True)
    _guide = [
        ("🏠 Dashboard（总览）", "打开默认就是这个页。先看最上面的<b>结论区</b>（今天该关注什么、机会、风险），"
                                 "再点下面的标签（宏观 / 结构 / 事件 / 研判）看细节。"),
        ("🔍 个股深度分析", "在左侧输入股票代码（例如美股 <code>AAPL</code>、港股 <code>0700.HK</code>、A股 <code>600519.SH</code>），"
                            "就能看到它的 K线、买卖策略、风险位、新闻解读。"),
        ("📊 跨资产对比", "把几只股票放在一起比强弱、比估值，适合纠结「买哪只」的时候。"),
    ]
    for title, desc in _guide:
        st.markdown(
            f'<div class="card" style="background:var(--card);border:1px solid var(--border);'
            f'border-left:4px solid var(--accent);padding:14px 16px;margin:10px 0;border-radius:12px;">'
            f'<h4 style="margin:0 0 6px;color:var(--text);">{title}</h4>'
            f'<div style="color:var(--text-dim);font-size:13px;line-height:1.8;">{desc}</div></div>',
            unsafe_allow_html=True,
        )
    st.caption("💡 小提示：本系统遵循「红涨绿跌」的中国习惯。数据由 GitHub 每日自动更新；点左侧「🔄 强制刷新数据」可手动清缓存。")

    # 2) AI分析 额度警示
    st.markdown("### ⚠️ 二、重要提醒：请谨慎点击「🤖 AI分析」按钮", unsafe_allow_html=True)
    st.markdown(
        '<div style="background:#fff7ed;border:1px solid #fed7aa;border-left:4px solid #f59e0b;'
        'border-radius:12px;padding:16px 18px;margin:10px 0;font-size:13.5px;line-height:1.9;color:#7c2d12;">'
        '系统里凡带 <b>「🤖 生成 AI 研判」</b> 字样的按钮（个股深度分析、维科夫吸筹扫描等），'
        '每点一次都会调用大模型（DeepSeek / OpenRouter），<b>消耗真实的 API 额度（token）</b>，是有成本的。<br><br>'
        '📌 <b>请遵守以下约定：</b><br>'
        '① <b>非必要请勿点击</b> AI 分析按钮；<br>'
        '② 大多数时候，页面上的「结论区 / 结构化指标 / 规则解读」已经足够你做判断，<b>不必点 AI</b>；<br>'
        '③ 若没有配置大模型 Key，系统会自动降级为「规则总结」，<b>不消耗额度</b>——这时点了也不会多花钱，但也没必要。<br><br>'
        '简单说：<b>能看结构化结论就别点 AI，把额度留给真正需要的时候。</b>'
        '</div>',
        unsafe_allow_html=True,
    )

    # 3) 功能现状说明（港股/A股缺AI评分、K线非实时）
    st.markdown("### 📋 三、功能现状说明（已知限制与改进方向）", unsafe_allow_html=True)

    with st.expander("❓ 为什么港股 / A股 板块缺少 AI 评分及相关数据？", expanded=True):
        st.markdown(
            '<div style="font-size:13px;line-height:1.9;color:var(--text);">'
            '<b>现象：</b>在「结论区 → Top机会(Top 5)」与个股深度分析中，<b>港股(.HK)、A股(.SS/.SZ) 往往没有 AI 评分</b>，'
            '且部分相关计算（个股期权 PCR、杠杆强平价、AI 决策卡片）也缺失或不完整。<br><br>'
            '<b>根本原因：</b>AI 评分来自每日数据管线 <code>stock_dashboard.py</code> 调用大模型，'
            '但只对「<b>成功取到技术数据的标的</b>」生成卡片。而港股 / A股的<b>日线历史数据唯一免费源是 akshare（东方财富系）</b>'
            '（见 <code>stock_dashboard.py</code> 的 <code>fetch_hk / fetch_a_share</code>）。'
            '在云端 CI / Streamlit Cloud（美国节点）上，akshare 常因<b>未安装</b>或<b>中国数据端点不可达</b>而失败，'
            '回退到 yfinance 对 <code>.HK/.SS/.SZ</code> 多返回空 → 这些标的在内存里是 <code>None</code> → '
            '<b>既不写入 stocks.csv，也不生成 AI 卡片</b>。于是整块 AI 评分及相关数据都缺了。<br><br>'
            '<b>改进方向：</b><br>'
            '① 在 CI 中预装 akshare，并配置可访问中国数据源的网络 / 代理；<br>'
            '② 为港股 / A股 接入<b>富途 OpenD</b> 或<b>同花顺 Financial-API</b>（需本地运行或 API Key）作为技术数据源；<br>'
            '③ 对港股 / A股 单独批次调用大模型生成 AI 卡片（不依赖美股同批次）。'
            '</div>',
            unsafe_allow_html=True,
        )

    with st.expander("❓ 为什么 K线图不是实时数据？", expanded=True):
        st.markdown(
            '<div style="font-size:13px;line-height:1.9;color:var(--text);">'
            '<b>现象：</b>个股深度分析的 K线图，最新一根蜡烛是<b>上一交易日的收盘</b>，盘中不跳动。<br><br>'
            '<b>根本原因：</b>主 K线（<code>_draw_kline</code>）使用的是<b>日K（日线）</b>数据'
            '（yfinance <code>interval=\"1d\"</code> 或 akshare 日线），本质是 T+1 的历史收盘，'
            '天然不含盘中实时。实时盘中走势目前只在「📈 K线技术面 → 当日分时」里以折线呈现'
            '（<code>fetch_intraday_trend</code>，美股 / A股 / 港股均有，港股 / A股依赖东财 / 腾讯分时源）。<br><br>'
            '<b>改进方向：</b><br>'
            '① 接入<b>富途 OpenD</b> / <b>同花顺</b> 的分钟级行情，补全实时蜡烛；<br>'
            '② 将「当日分时」升级为<b>实时刷新蜡烛图</b>（需要分钟级 tick 数据源，通常为付费或本地通道）。'
            '</div>',
            unsafe_allow_html=True,
        )

    # 5) 数据层构成清单
    st.markdown("### 🧱 五、本系统的数据层由哪些模块构成？", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:12.5px;line-height:1.85;color:var(--text-dim);margin-bottom:8px;">'
        '下面按职责列出全部数据模块（函数名在 <code>utils.py</code> / <code>stock_dashboard.py</code> 中可查）。'
        '</div>',
        unsafe_allow_html=True,
    )
    _layers = [
        ("① 宏观数据层", "利率 / 杠杆 / 波动率 / 期权情绪",
         "fetch_fred_linkage_series、fetch_us_debt、fetch_nfci_leverage、fetch_margin_debt（含 FINRA Margin Debt）、"
         "fetch_2y_scorecard、fetch_options_pcr / fetch_all_pcr"),
        ("② 新闻舆情层", "宏观 / 政策 / 个股 / 公告 / 龙虎榜",
         "fetch_macro_news、fetch_policy_news（含免费版）、fetch_all_news、fetch_google_news_rss、"
         "fetch_stock_news、fetch_eastmoney_stock_news、fetch_cninfo_market_announcements、"
         "fetch_cninfo_stock_announcements、fetch_dragon_tiger、fetch_10jqka_news、fetch_xueqiu_news、interpret_news"),
        ("③ 行情与实时层", "实时报价 / 速览 / 分时 / 盘口 / 逐笔",
         "fetch_realtime_quote（东财→腾讯→新浪多源）、fetch_realtime_snapshots、fetch_intraday_trend、"
         "fetch_order_book（五档盘口）、fetch_tick_detail（逐笔成交）"),
        ("④ A股数据层", "指数 / 热力图 / K线 / 个股报价 / 资金流",
         "fetch_a_share_overview（东财→腾讯→akshare 降级）、fetch_a_share_heatmap_data、fetch_a_share_kline、"
         "fetch_a_share_quote、fetch_a_share_top_movers、fetch_capital_flow_eastmoney（主力资金）"),
        ("⑤ 港股数据层", "港股日线 / 实时报价",
         "stock_dashboard.fetch_hk（akshare）、fetch_realtime_quote 的港股分支（东财→腾讯→新浪）"),
        ("⑥ 美股 / ETF 数据层", "历史 K线 / 新闻 / 指数",
         "fetch_history_fixed（yfinance）、fetch_yahoo_rss、fetch_finnhub_news、fetch_stocktwits、fetch_index_quote"),
        ("⑦ 技术指标与评分层", "组合 / 宏观风险 / 选股 / 明日观察位",
         "compute_portfolio_dominance、compute_macro_risk_radar、screener（维科夫 + 多因子）、"
         "BottomFishingEngine、ReversalAnalyzer、compute_tomorrow_watch、risk.r_multiple_plan（R倍数止盈）"),
        ("⑧ AI 研判层", "大模型自然语言生成（DeepSeek / OpenRouter）",
         "_call_llm、narrate_tomorrow_watch、build_prediction_prompt、build_morning_brief_prompt、"
         "render_morning_brief、generate_overview、generate_cards（AI 评分卡片）"),
        ("⑨ 数据管线 / 持久化层", "每日 CI 生成静态数据文件",
         "stock_dashboard.collect_all / _fetch_stock_parallel / _persist；产物：stocks.csv、macro.csv、"
         "sox.csv、cards.json、leverage_risk.json、morning_brief.md、evening_recap.md"),
        ("⑩ 外部增强层（可选）", "本地 / Key 增强，缺失不影响主流程",
         "fetch_realtime_via_futu（富途 OpenD，需本地进程）、fetch_realtime_via_ths / utils_ths.py（同花顺，需 API Key）"),
    ]
    for name, duty, fns in _layers:
        st.markdown(
            f'<div style="background:var(--card);border:1px solid var(--border);border-radius:10px;'
            f'padding:12px 14px;margin:8px 0;">'
            f'<div style="font-weight:700;color:var(--text);font-size:13.5px;">{name}'
            f'<span style="font-weight:400;color:var(--accent);font-size:12px;margin-left:8px;">· {duty}</span></div>'
            f'<div style="color:var(--text-dim);font-size:12px;line-height:1.7;margin-top:4px;">'
            f'<code style="font-size:11px;">{fns}</code></div></div>',
            unsafe_allow_html=True,
        )
    st.caption("⚠️ 投资有风险，本系统所有数据与 AI 结论仅供参考，不构成投资建议。")


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
elif page == "📖 使用说明":
    page_usage_guide()
elif page == "⚙️ 数据诊断":
    page_diagnostics()

# ---------------------------------------------------------------------------
# 底部
# ---------------------------------------------------------------------------
st.markdown(
    f"""
<div style="text-align:center;color:var(--text-dim);font-size:11px;padding:30px 0 10px;border-top:1px solid var(--border);margin-top:30px;">
    ⚡ Investment Copilot ·
    <svg width="20" height="20" viewBox="0 0 32 32" style="vertical-align:middle;margin:0 3px;">
      <circle cx="11" cy="9" r="4" fill="#e0a93b"/><circle cx="21" cy="9" r="4" fill="#e0a93b"/>
      <circle cx="16" cy="17" r="9" fill="#f0c060"/>
      <ellipse cx="16" cy="20" rx="4.5" ry="3.5" fill="#d99a3a"/>
      <circle cx="13" cy="15" r="1.1" fill="#3a2a12"/><circle cx="19" cy="15" r="1.1" fill="#3a2a12"/>
      <circle cx="16" cy="19" r="1" fill="#3a2a12"/>
    </svg>
    Author: Winnie Wang · {datetime.now().strftime('%Y-%m-%d %H:%M')} · 仅供研究参考，不构成投资建议
</div>
""",
    unsafe_allow_html=True,
)
