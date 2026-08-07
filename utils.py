"""
utils.py — 投资分析工作台 (Investment Copilot) 工具模块

提供以下能力：
  1. SerpApi 多源新闻抓取（宏观 / 政策 / 个股），自动去重
  2. 自建 Fear & Greed 指数（5 因子模型：VIX、市场宽度、动量、安全资产、垃圾债利差）
  3. FedWatch 降息概率（用 SOFR/联邦基金期货反推下次会议概率）
  4. Economic Calendar（财报 / FOMC / CPI / PPI / NFP 抓取 + 静态兜底）
  5. 美股 / 港股热力图数据源（标普 500 + 纳指 100 + 恒生指数 + 国企指数）
  6. Morning Brief / Evening Recap 提示词模板
  7. OpenRouter / DeepSeek 双 LLM 接入

设计原则：
  - 所有外部调用都做超时 + 异常兜底，单点失败不能让 Streamlit 崩溃
  - 所有数据可被 Streamlit @st.cache_data 装饰（提供 hash 函数）
  - 不依赖付费 API（除 SERPAPI_KEY），所有 fallback 优先用 yfinance 免费数据
  - **双路读 Key**：支持 os.environ（GitHub Actions / 本地）和 st.secrets（Streamlit Cloud）
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Streamlit 软依赖：stock_dashboard.py 是非 web 跑（GitHub Actions / 本地）
# 不能强制 import streamlit；这里用 try/except 软加载
# ---------------------------------------------------------------------------
try:
    import streamlit as st  # noqa: F401

    _HAS_STREAMLIT = True
except ImportError:
    st = None  # type: ignore
    _HAS_STREAMLIT = False


def _get_secret(name: str, default: str = "") -> str:
    """
    双路读 secret（任何 API key 通用）：
      1) os.environ[name]（GitHub Actions / 本地 .env）
      2) st.secrets[name]（Streamlit Cloud 的 secrets）
    """
    val = os.environ.get(name, "")
    if val:
        return val
    if _HAS_STREAMLIT and hasattr(st, "secrets"):
        try:
            val = st.secrets.get(name, "")  # type: ignore[union-attr]
            if val:
                return str(val)
        except Exception:  # noqa: BLE001
            pass
    return default


logger = logging.getLogger("copilot_utils")

# ---------------------------------------------------------------------------
# 1. SerpApi 新闻抓取（多源 + 去重）
# ---------------------------------------------------------------------------

# 宏观 / 政策 / 大类资产 — 用 Google News（更全、更适合美股市场）
MACRO_QUERIES = [
    "Federal Reserve FOMC rate decision",
    "CPI inflation report today",
    "PPI producer price index release",
    "nonfarm payrolls NFP jobs report",
    "US Treasury yield 10 year",
    "geopolitical risk market",
    "China policy stimulus economy",
    "Hong Kong stock market policy",
    "oil price OPEC",
    "AI chip export control",
]

# 个股关键词（每只股票搜一两次，避免一次搜出太杂）
STOCK_QUERY_TEMPLATES = [
    "{sym} stock news",
    "{sym} earnings guidance",
]

CN_STOCK_QUERY_TEMPLATES = [
    "{sym} 股票 财报",
    "{sym} 公司 公告",
]


def _serpapi_search(
    query: str,
    api_key: str,
    engine: str = "google_news",
    num: int = 5,
    gl: str = "us",
    hl: str = "en",
    timeout: int = 12,
) -> List[Dict[str, Any]]:
    """
    单次 SerpApi 调用的统一入口。
    engine 支持：
      - "google_news"  英文新闻（默认）
      - "baidu_news"   中文新闻
    返回: [{title, link, source, date, snippet}]，失败返回 []
    """
    if not api_key:
        return []
    try:
        from serpapi.google_search import GoogleSearch  # type: ignore
        params: Dict[str, Any] = {
            "engine": engine,
            "q": query,
            "api_key": api_key,
            "num": num,
        }
        if engine == "google_news":
            params["gl"] = gl
            params["hl"] = hl
        result = GoogleSearch(params).get_dict()
        # SerpApi 免费额度耗尽/被限流时返回 error 字段而非 news_results：
        #   {"error": "Google News API is not available for this query..."} 或
        #   {"search_metadata": ..., "error": "403 Forbidden: rate limit"}
        if result.get("error"):
            logger.warning("SerpApi[%s] '%s' 返回错误: %s", engine, query, str(result.get("error"))[:200])
            return []
        if result.get("search_information", {}).get("organic_results_state") == "Fully empty":
            return []
        news = result.get("news_results") or []
        out: List[Dict[str, Any]] = []
        for n in news:
            out.append(
                {
                    "title": n.get("title", "").strip(),
                    "link": n.get("link", "").strip(),
                    "source": n.get("source", "") or n.get("publisher", ""),
                    "date": n.get("date", ""),
                    "snippet": n.get("snippet", ""),
                }
            )
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning("SerpApi[%s] '%s' 失败: %s", engine, query, e)
        return []


def _dedup_news(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 link/title 哈希去重，保留先出现的。"""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for it in items:
        key = (it.get("link") or it.get("title", "")).strip().lower()
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        if h in seen or not it.get("title"):
            continue
        seen.add(h)
        out.append(it)
    return out


def fetch_macro_news(api_key: str, top_n: int = 12) -> List[Dict[str, Any]]:
    """宏观新闻（Fed/CPI/就业/地缘 等）— 全部英文 google_news。"""
    pool: List[Dict[str, Any]] = []
    for q in MACRO_QUERIES:
        pool.extend(_serpapi_search(q, api_key, engine="google_news", num=3))
        time.sleep(0.15)  # 避免 SerpApi 限流
    pool = _dedup_news(pool)
    # 简单按日期排序（SerpApi date 格式 "3 hours ago" / "Yesterday" 不一定能 parse）
    return pool[:top_n]


def fetch_policy_news(api_key: str, top_n: int = 8) -> List[Dict[str, Any]]:
    """政策新闻（SEC/Treasury/White House/中国政策）。"""
    queries = [
        "SEC regulation announcement",
        "US Treasury Secretary policy",
        "White House executive order economy",
        "China PBOC policy",
        "Hong Kong monetary authority",
        "EU regulation financial market",
    ]
    pool: List[Dict[str, Any]] = []
    for q in queries:
        pool.extend(_serpapi_search(q, api_key, engine="google_news", num=3))
        time.sleep(0.15)
    return _dedup_news(pool)[:top_n]


def fetch_google_news_rss(
    query: str, hl: str = "zh-CN", gl: str = "CN", num: int = 6
) -> List[Dict[str, Any]]:
    """Google News RSS 搜索（完全免费、无 key）。按关键词抓最新财经/政策新闻。"""
    from urllib.parse import quote_plus

    ceid = {"zh-CN": "CN:zh-Hans", "en-US": "US:en"}.get(hl, "CN:zh-Hans")
    url = (
        "https://news.google.com/rss/search?q="
        + quote_plus(query)
        + f"&hl={hl}&gl={gl}&ceid={ceid}"
    )
    return fetch_rss_feed(url, source_name=f"GoogleNews[{hl}]", top_n=num)


# 政策板块免费源关键词（中英两组，覆盖 中国/美国 政策面）
POLICY_QUERIES_ZH = [
    "中国 央行 货币政策",
    "中国 证监会 监管",
    "国务院 经济政策",
    "财政部 财政政策",
    "降准 降息",
    "关税 反制",
]
POLICY_QUERIES_EN = [
    "China PBOC monetary policy",
    "China CSRC regulation",
    "US Treasury tariff trade policy",
    "Federal Reserve policy rate",
    "SEC regulation financial markets",
    "trade policy tariff China US",
]


def fetch_policy_news_free(top_n: int = 8) -> List[Dict[str, Any]]:
    """政策新闻免费源：Google News RSS 按中英政策关键词搜索（无需任何 API key）。"""
    pool: List[Dict[str, Any]] = []
    for q in POLICY_QUERIES_ZH:
        pool.extend(fetch_google_news_rss(q, hl="zh-CN", gl="CN", num=3))
        time.sleep(0.1)
    for q in POLICY_QUERIES_EN:
        pool.extend(fetch_google_news_rss(q, hl="en-US", gl="US", num=3))
        time.sleep(0.1)
    pool = _dedup_news(pool)
    return pool[:top_n]


def fetch_stock_news(
    symbol: str,
    api_key: str,
    is_hk: bool = False,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """个股新闻：美股 → google_news；港股 → baidu_news。"""
    sym_clean = symbol.replace(".HK", "").zfill(5) if is_hk else symbol
    if is_hk:
        templates = CN_STOCK_QUERY_TEMPLATES
        engine = "baidu_news"
    else:
        templates = STOCK_QUERY_TEMPLATES
        engine = "google_news"
    pool: List[Dict[str, Any]] = []
    for tpl in templates:
        q = tpl.format(sym=sym_clean)
        pool.extend(_serpapi_search(q, api_key, engine=engine, num=3))
        time.sleep(0.1)
    return _dedup_news(pool)[:top_n]


def fetch_all_news(
    api_key: str,
    symbols: List[str],
    out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    一键抓全：宏观 + 政策 + 每只个股。结果形如:
    {
      "macro": [...],
      "policy": [...],
      "stocks": { "AAPL": [...], "0700.HK": [...], ... },
      "generated_at": "2026-07-31 17:05",
    }
    """
    if not api_key:
        return {"macro": [], "policy": [], "stocks": {}, "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M")}

    payload: Dict[str, Any] = {
        "macro": fetch_macro_news(api_key, top_n=12),
        "policy": fetch_policy_news(api_key, top_n=8),
        "stocks": {},
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    for sym in symbols:
        is_hk = sym.endswith(".HK")
        try:
            payload["stocks"][sym] = fetch_stock_news(sym, api_key, is_hk=is_hk, top_n=5)
        except Exception as e:  # noqa: BLE001
            logger.warning("%s 新闻失败: %s", sym, e)
            payload["stocks"][sym] = []

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# 2. Fear & Greed 指数（5 因子自建版，参考 CNN 原始模型）
# ---------------------------------------------------------------------------
# 5 因子（每因子 0-100 标准化，0=极度恐惧，100=极度贪婪）：
#   A. 市场动量       SPY 125 日均线偏离度
#   B. 市场强度       标普站上 50 日均线的成分股占比（用 SPY vs SPY 200MA 近似）
#   C. 宽度           SPY 50/200MA 距离
#   D. VIX            芝加哥期权交易所波动率
#   E. 安全资产需求   黄金 / 标普 20 日比值
# 综合 = 0.25*A + 0.20*B + 0.20*C + 0.25*D + 0.10*E


def _safe_last(series: Optional[pd.Series], default: float = 0.0) -> float:
    try:
        if series is None or series.empty:
            return default
        v = series.iloc[-1]
        return float(v) if pd.notna(v) else default
    except Exception:  # noqa: BLE001
        return default


def _normalize(value: float, low: float, high: float) -> float:
    """将任意值线性映射到 [0, 100]，超出端点裁剪。"""
    if high == low:
        return 50.0
    out = (value - low) / (high - low) * 100.0
    return float(max(0.0, min(100.0, out)))


def calculate_fear_greed() -> Dict[str, Any]:
    """
    自建 Fear & Greed 指数（0-100，越高越贪婪）。
    """
    try:
        spy = yf.download("SPY", period="1y", progress=False, auto_adjust=True)
        vix_df = yf.download("^VIX", period="3mo", progress=False)
        gld_df = yf.download("GLD", period="3mo", progress=False, auto_adjust=True)
    except Exception as e:  # noqa: BLE001
        logger.warning("F&G 数据抓取失败: %s", e)
        return {"score": 50.0, "label": "中性", "factors": {}, "error": str(e)}

    if spy.empty or vix_df.empty:
        return {"score": 50.0, "label": "中性", "factors": {}, "error": "SPY/VIX 数据为空"}

    spy_close = spy["Close"].squeeze() if isinstance(spy["Close"], pd.DataFrame) else spy["Close"]
    vix_close = vix_df["Close"].squeeze() if isinstance(vix_df["Close"], pd.DataFrame) else vix_df["Close"]

    # A. 动量：当前价相对 125 日均线偏离度，越高越贪婪
    ma125 = spy_close.rolling(125).mean().iloc[-1]
    momentum_pct = (float(spy_close.iloc[-1]) - float(ma125)) / float(ma125) * 100 if ma125 else 0.0
    a_score = _normalize(momentum_pct, low=-15.0, high=15.0)

    # B. 强度：SPY 在 50 日均线之上多少
    ma50 = spy_close.rolling(50).mean().iloc[-1]
    above_ma50_pct = (float(spy_close.iloc[-1]) - float(ma50)) / float(ma50) * 100 if ma50 else 0.0
    b_score = _normalize(above_ma50_pct, low=-10.0, high=10.0)

    # C. 宽度：50MA vs 200MA 距离（黄金交叉深度）
    ma200 = spy_close.rolling(200).mean().iloc[-1] if len(spy_close) >= 200 else ma50
    width_pct = (float(ma50) - float(ma200)) / float(ma200) * 100 if ma200 else 0.0
    c_score = _normalize(width_pct, low=-10.0, high=10.0)

    # D. VIX：低 = 贪婪
    vix_val = _safe_last(vix_close, 20.0)
    d_score = _normalize(vix_val, low=10.0, high=40.0)  # 反向
    d_score = 100.0 - d_score

    # E. 安全资产需求：GLD 涨跌越强 → 越恐惧
    if gld_df is not None and not gld_df.empty:
        gld_close = gld_df["Close"].squeeze() if isinstance(gld_df["Close"], pd.DataFrame) else gld_df["Close"]
        gld_20ret = (float(gld_close.iloc[-1]) / float(gld_close.iloc[-20]) - 1.0) * 100 if len(gld_close) >= 20 else 0.0
    else:
        gld_20ret = 0.0
    e_score = _normalize(gld_20ret, low=10.0, high=-10.0)  # 反向
    e_score = 100.0 - e_score

    composite = round(0.30 * a_score + 0.20 * b_score + 0.15 * c_score + 0.25 * d_score + 0.10 * e_score, 1)

    if composite >= 75:
        label = "极度贪婪"
    elif composite >= 60:
        label = "贪婪"
    elif composite >= 45:
        label = "中性"
    elif composite >= 25:
        label = "恐惧"
    else:
        label = "极度恐惧"

    return {
        "score": composite,
        "label": label,
        "factors": {
            "动量": round(a_score, 1),
            "强度": round(b_score, 1),
            "宽度": round(c_score, 1),
            "VIX反向": round(d_score, 1),
            "避险需求反向": round(e_score, 1),
        },
    }


# ---------------------------------------------------------------------------
# 3. FedWatch（用 SOFR/联邦基金期货反推会议降息概率）
# ---------------------------------------------------------------------------

# 已知 2026 年 FOMC 会议日期（静态兜底，滚动更新可由 macro 抓取覆盖）
# 依据：美联储官网 fomccalendars.htm（2026-08-04 核验）
FOMC_2026_DATES = [
    "2026-01-27", "2026-01-28",
    "2026-03-17", "2026-03-18",
    "2026-04-28", "2026-04-29",
    "2026-06-16", "2026-06-17",
    "2026-07-28", "2026-07-29",
    "2026-09-15", "2026-09-16",
    "2026-10-27", "2026-10-28",
    "2026-12-08", "2026-12-09",
]


def _normalize_close_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    把 yf.download 的输出压成 1D Close Series。
    处理 MultiIndex、单行 DataFrame、squeeze 后退化等所有边界情况。
    返回 None 表示无法提取。
    """
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        return None
    close = df["Close"]
    # 如果 yfinance 返回单行单列 DataFrame，squeeze 会变成 scalar；要先取列
    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 1:
            close = close.iloc[:, 0]
        else:
            close = close.iloc[:, 0]  # 取第一列
    if not isinstance(close, pd.Series):
        return None
    close = close.dropna()
    if close.empty:
        return None
    return close


def next_fomc_meeting(asof: Optional[Any] = None) -> str:
    """
    动态计算「下一场 FOMC 会议」的起始日期（杜绝硬编码过期日期的低级错误）。

    输入：FOMC_2026_DATES 为每场会议的两日配对（[start, end, start, end, ...]）。
    返回：第一场 'start' >= 今天（或 asof）的会议起始日；若全部已过期，返回最后一场。
    """
    try:
        from datetime import date as _date
        ref = asof.date() if hasattr(asof, "date") else (_date.today() if asof is None else _date.fromisoformat(str(asof)[:10]))
    except Exception:  # noqa: BLE001
        from datetime import date as _date
        ref = _date.today()
    future = [d for d in FOMC_2026_DATES if _safe_parse_date(d) is not None and _safe_parse_date(d) >= ref]
    if future:
        return future[0]
    # 全部过期：返回最后一场的起始日（仍比返回过去日期更有信息量）
    return FOMC_2026_DATES[0] if FOMC_2026_DATES else "TBD"


def _safe_parse_date(s: str):
    try:
        from datetime import date as _date
        return _date.fromisoformat(str(s)[:10])
    except Exception:  # noqa: BLE001
        return None


def calc_fedwatch_from_futures() -> Dict[str, Any]:
    """
    用 SR3 (3 月 SOFR 期货) 反推市场隐含的下次会议利率区间概率。

    简化模型：
      - 当前有效联邦基金利率假设为 5.33% (2024 末水平，会随时间变化)
      - SR3 = 100 - 隐含利率
      - 隐含利率 = (当前 SR3 - 目标 SR3) / 12 + 当前 FFR
    """
    try:
        sr3 = yf.download("SR3=F", period="6mo", progress=False, auto_adjust=False)
        if sr3 is None or sr3.empty:
            sr3 = yf.download("ZQ=F", period="6mo", progress=False, auto_adjust=False)
        close = _normalize_close_series(sr3) if sr3 is not None else None
        if close is None:
            return {"error": "无法获取 SOFR/FF 期货 (SR3=F / ZQ=F 都为空)", "meetings": []}
        # 关键修复：必须转成 float，不能让 .iloc 在 float64 上被调用
        last_val = float(close.iloc[-1])
        if pd.isna(last_val) or last_val <= 0 or last_val >= 100:
            return {"error": f"SOFR 期货价格异常: {last_val}", "meetings": []}
        implied_rate = 100.0 - last_val
    except Exception as e:  # noqa: BLE001
        logger.warning("FedWatch 计算失败: %s", e)
        return {"error": str(e), "meetings": []}

    current_ffr = 3.625  # 兜底；2026-08 联邦基金目标区间 3.50%-3.75% 中点（理想情况用 FRED DFF）
    next_meeting = next_fomc_meeting()  # 动态计算下一场会议（不再写死 2026-01-27 等过期日期）
    cut_bps = round((implied_rate - current_ffr) * 100)
    if cut_bps <= 0:
        prob_cut = 5.0
        prob_hold = 90.0
        prob_hike = 5.0
        verdict = "市场预期维持利率不变"
    else:
        # 简化：cut 概率 ∝ 隐含降息幅度
        prob_cut = min(95.0, 30.0 + cut_bps * 1.2)
        prob_hold = max(0.0, 100.0 - prob_cut - 5.0)
        prob_hike = 5.0
        verdict = f"市场预期下次会议降息约 {cut_bps}bp"

    return {
        "implied_rate": round(implied_rate, 2),
        "current_ffr": current_ffr,
        "cut_bps": cut_bps,
        "next_meeting": next_meeting,
        "prob_cut": round(prob_cut, 1),
        "prob_hold": round(prob_hold, 1),
        "prob_hike": round(prob_hike, 1),
        "verdict": verdict,
        "source": "CME SOFR/FF 期货 (SR3=F / ZQ=F)",
        "asof": str(close.index[-1].date()) if hasattr(close.index[-1], "date") else "",
    }


# ---------------------------------------------------------------------------
# 4. Economic Calendar（财报 / 宏观日程）
# ---------------------------------------------------------------------------

# 静态兜底：未来 30 天已知的重磅事件
# 日期核验：AM Best/UniCredit 2026-08 经济日历、美联储官网（2026-08-04）
STATIC_CALENDAR = [
    {"date": "2026-08-07", "time": "20:30", "event": "美国 7 月非农就业 (NFP)", "importance": "🔴 高", "type": "宏观"},
    {"date": "2026-08-05", "time": "未定", "event": "苹果 (AAPL) 财报", "importance": "🟠 中", "type": "财报"},
    {"date": "2026-08-12", "time": "20:30", "event": "美国 7 月 CPI 同比", "importance": "🔴 高", "type": "宏观"},
    {"date": "2026-08-13", "time": "20:30", "event": "美国 7 月 PPI 同比", "importance": "🟡 中", "type": "宏观"},
    {"date": "2026-08-19", "time": "02:00", "event": "FOMC 会议纪要公布 (7/28-29 会议)", "importance": "🟠 中", "type": "宏观"},
    {"date": "2026-08-28", "time": "20:30", "event": "美联储主席 Warsh 讲话 (Jackson Hole 8/27-29)", "importance": "🔴 高", "type": "宏观"},
    {"date": "2026-08-28", "time": "20:30", "event": "美国 7 月 PCE 物价指数", "importance": "🔴 高", "type": "宏观"},
    {"date": "2026-08-28", "time": "未定", "event": "英伟达 (NVDA) 财报", "importance": "🔴 高", "type": "财报"},
    {"date": "2026-08-29", "time": "未定", "event": "中芯国际 / 港股科技股 财报", "importance": "🟠 中", "type": "财报"},
]


def fetch_economic_calendar(api_key: Optional[str] = None) -> List[Dict[str, str]]:
    """
    经济日历：优先 SerpApi 抓"earnings calendar this week" / "FOMC schedule 2026"，
    抓不到 / 失败 / 无 key → 用 STATIC_CALENDAR 兜底。
    """
    if not api_key:
        return STATIC_CALENDAR
    try:
        from serpapi.google_search import GoogleSearch  # type: ignore
        pool: List[Dict[str, str]] = []
        queries = [
            "FOMC meeting dates 2026 schedule",
            "earnings calendar this week AAPL NVDA MSFT",
            "CPI release date August 2026",
            "nonfarm payrolls August 2026 release",
        ]
        for q in queries:
            res = GoogleSearch(
                {"engine": "google_news", "q": q, "api_key": api_key, "num": 3}
            ).get_dict()
            for n in res.get("news_results", []) or []:
                pool.append(
                    {
                        "date": n.get("date", "近期"),
                        "time": "—",
                        "event": n.get("title", ""),
                        "importance": "🟠 中",
                        "type": "日程",
                    }
                )
            time.sleep(0.15)
        if not pool:
            return STATIC_CALENDAR
        # 合并静态表（高重要性已知事件）
        return STATIC_CALENDAR + pool[:6]
    except Exception as e:  # noqa: BLE001
        logger.warning("经济日历抓取失败: %s", e)
        return STATIC_CALENDAR


# ---------------------------------------------------------------------------
# 5. 美股 / 港股热力图数据
# ---------------------------------------------------------------------------

# 标普 500 + 纳指 100 主要成分股（按市值精选 60 只 + 行业代表，覆盖 11 个 GICS 板块）
US_HEATMAP_TICKERS = {
    "科技": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CRM", "AMD", "INTC", "QCOM", "ADBE"],
    "通信": ["NFLX", "DIS", "T", "VZ", "TMUS"],
    "消费": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "WMT", "TGT", "COST"],
    "金融": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "V", "MA", "AXP"],
    "医疗": ["UNH", "LLY", "PFE", "JNJ", "ABBV", "MRK", "TMO", "ABT"],
    "工业": ["BA", "CAT", "GE", "HON", "LMT", "RTX", "DE"],
    "能源": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "公用": ["NEE", "DUK", "SO"],
    "材料": ["LIN", "FCX", "NEM"],
    "地产": ["PLD", "AMT", "CCI"],
    "半导体": ["TSM", "ASML", "MU", "MRVL", "LRCX", "AMAT", "KLAC"],
}

# 港股：恒生指数 + 恒生科技 + 国企指数主要成分（精简 30 只）
HK_HEATMAP_TICKERS = {
    "互联网/科技": ["0700.HK", "9988.HK", "3690.HK", "1810.HK", "9618.HK", "1024.HK", "2382.HK"],
    "金融": ["0005.HK", "0945.HK", "1299.HK", "0388.HK", "1398.HK", "2628.HK", "3988.HK"],
    "地产": ["0016.HK", "1117.HK", "1109.HK", "1997.HK", "0001.HK", "0012.HK"],
    "消费": ["0941.HK", "0288.HK", "1876.HK", "2319.HK", "9992.HK", "2020.HK"],
    "能源/材料": ["0883.HK", "0857.HK", "0386.HK", "2899.HK"],
    "医药": ["1093.HK", "2269.HK", "1177.HK", "1801.HK"],
    "汽车/制造": ["1211.HK", "9818.HK", "0175.HK", "2382.HK", "0267.HK"],
}

# A 股全市场热力图（按申万一级行业，主要成分股 / 龙头）
A_SHARE_HEATMAP_TICKERS: Dict[str, List[str]] = {
    "食品饮料": ["600519.SS", "000858.SZ", "603288.SS"],
    "银行/保险": ["601318.SS", "600036.SS", "601166.SS"],
    "新能源": ["300750.SZ", "002594.SZ", "601012.SS"],
    "半导体/电子": ["688981.SS", "002475.SZ", "603501.SS", "688041.SS"],
    "医药": ["600276.SS", "300760.SZ"],
    "汽车": ["601127.SS", "600104.SS"],
    "家电": ["000333.SZ", "000651.SZ"],
    "周期/材料": ["600900.SS", "601899.SS", "600585.SS"],
    "能源": ["600028.SS", "601857.SS"],
    "通信": ["600941.SS"],
    "计算机": ["002415.SZ"],
    "地产": ["600048.SS"],
}


def build_heatmap_data(
    universe: Dict[str, List[str]],
    period: str = "1d",
) -> pd.DataFrame:
    """
    拉取一组 ticker 的当日涨跌幅，并组装成 [sector, symbol, change%, mkt_cap_proxy]。
    返回 DataFrame，便于 plotly treemap。
    """
    rows: List[Dict[str, Any]] = []
    tickers = [t for arr in universe.values() for t in arr]
    seen: set = set()
    uniq = [t for t in tickers if not (t in seen or seen.add(t))]

    for sector, arr in universe.items():
        for sym in arr:
            try:
                t = yf.Ticker(sym)
                hist = t.history(period="5d", auto_adjust=True)
                if hist is None or hist.empty or len(hist) < 2:
                    raise ValueError("empty")
                last = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2])
                chg = (last - prev) / prev * 100.0
                # 用近 50 日均成交额粗略代理市值权重（treemap 显示大小用）
                avg_vol_amt = float((hist["Close"] * hist["Volume"]).iloc[-50:].mean()) if len(hist) >= 50 else float((hist["Close"] * hist["Volume"]).mean())
                rows.append(
                    {
                        "sector": sector,
                        "symbol": sym.replace(".HK", "").replace(".SS", "").replace(".SZ", ""),
                        "symbol_full": sym,
                        "change_pct": round(chg, 2),
                        "weight": max(1e6, avg_vol_amt),  # treemap size
                        "price": last,
                    }
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("热力图 %s 失败: %s", sym, e)
                rows.append(
                    {
                        "sector": sector,
                        "symbol": sym.replace(".HK", "").replace(".SS", "").replace(".SZ", ""),
                        "symbol_full": sym,
                        "change_pct": 0.0,
                        "weight": 1e6,
                        "price": 0.0,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Morning Brief / Evening Recap 提示词模板
# ---------------------------------------------------------------------------

MORNING_BRIEF_PROMPT = """你是华尔街资深卖方策略师，风格类似 Goldman / Morgan Stanley 的 Morning Strategy Note。
请基于以下数据写一份「盘前早报 (Morning Brief)」，目标读者是专业投资者。

【日期】{date} (亚洲盘后 / 美东盘前)
【市场状态】{market_status}
【F&G 指数】{fear_greed_score}（{fear_greed_label}）
【情绪指数】{sentiment_score}（{sentiment_label}）
【FedWatch】{fedwatch_verdict}（降息概率 {fedwatch_cut}% / 维持 {fedwatch_hold}% / 加息 {fedwatch_hike}%）

【昨日美股收盘】
- 标普500: {spx_chg}%
- 纳斯达克: {ndx_chg}%
- 道琼斯: {dji_chg}%
- SOX 半导体: {sox_chg}%

【今晨宏观新闻 Top 5】
{macro_news}

【今晨政策新闻 Top 3】
{policy_news}

【今日重要日程】
{calendar}

请按以下结构输出（每个 section 用 === 分隔，500-800 字）：

=== 1. 隔夜发生了什么 ===
总结昨晚美股/海外市场最重要的 2-3 件事 + 直接驱动

=== 2. 今天的核心主题 ===
用一句话定位今天的交易主线（例：通胀数据前多空观望 / 财报季开局情绪谨慎）

=== 3. 关键观察 & 风险 ===
列 3-4 个今天必须盯的指标 / 价位 / 事件

=== 4. 交易思路 ===
给出 1-2 个可执行的方向（板块/风格），不做具体股票点位推荐
"""

EVENING_RECAP_PROMPT = """你是华尔街资深卖方策略师，请基于以下数据写一份「盘后总结 (Evening Recap)」。
风格类似 Goldman Sachs EOD Note，专业、克制、有观点。

【日期】{date} (美东收盘)
【市场状态】{market_status}
【F&G 指数】{fear_greed_score}（{fear_greed_label}）

【今日美股收盘】
- 标普500: {spx_chg}% (收 {spx_close})
- 纳斯达克: {ndx_chg}% (收 {ndx_close})
- 道琼斯: {dji_chg}% (收 {dji_close})
- SOX 半导体: {sox_chch}% (回撤 {sox_drawdown}%)
- VIX: {vix} ({vix_chg}%)

【自选股今日表现】
{watchlist_perf}

【今日宏观新闻 Top 5】
{macro_news}

【今日重要事件回顾】
{calendar}

【财报 / 个股大新闻】
{stock_news}

请按以下结构输出（800-1200 字）：

=== 1. 今天发生了什么 ===
3-5 句话总结今日行情的核心叙事

=== 2. 为什么涨 / 为什么跌 ===
归因：宏观数据？资金面？个股事件？技术面突破？

=== 3. 最大的 driver ===
挑出今天对市场影响最大的一件事（数据 / 政策 / 个股）

=== 4. 板块轮动 ===
资金今天从哪到哪？

=== 5. 未来 3 天风险 ===
列 2-3 个潜在催化剂 / 风险点，给出观察指标

要求：专业但不夸张，所有判断用"可能""倾向于"措辞，不做绝对预测。
"""


def _call_llm(
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.65,
    max_tokens: int = 1600,
    prefer: str = "deepseek",  # "deepseek" | "openrouter" | "auto"
) -> Optional[str]:
    """
    通用 LLM 调用层：优先 DeepSeek，OpenRouter 兜底。
    从双路 secret 同时读取两个 key：
      - DEEPSEEK_API_KEY
      - OPENROUTER_API_KEY
    """
    deepseek_key = _get_secret("DEEPSEEK_API_KEY")
    openrouter_key = _get_secret("OPENROUTER_API_KEY")

    # 决定调用顺序
    if prefer == "openrouter":
        order = [("openrouter", openrouter_key), ("deepseek", deepseek_key)]
    elif prefer == "deepseek":
        order = [("deepseek", deepseek_key), ("openrouter", openrouter_key)]
    else:  # auto
        order = [("deepseek", deepseek_key), ("openrouter", openrouter_key)]

    for provider, key in order:
        if not key:
            continue
        try:
            from openai import OpenAI
            if provider == "deepseek":
                client = OpenAI(api_key=key, base_url="https://api.deepseek.com/v1")
                model = "deepseek-chat"
            else:  # openrouter
                client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
                # 默认模型：Claude 3.5 Sonnet（性价比最高的卖方报告模型）
                model = "anthropic/claude-3.5-sonnet"
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content
            if content:
                return content
        except Exception as e:  # noqa: BLE001
            logger.warning("%s 调用失败: %s — 切换下一个 provider", provider, e)
            continue
    return None


def render_morning_brief(
    api_key: str = "",
    context: Dict[str, Any] = None,
    *,
    prefer: str = "deepseek",
) -> Optional[str]:
    """调用 LLM 生成 Morning Brief（DeepSeek / OpenRouter 二选一）。"""
    if context is None:
        return None
    prompt = MORNING_BRIEF_PROMPT.format(**context)
    return _call_llm(
        messages=[
            {"role": "system", "content": "你是专业卖方策略师，输出必须是简体中文，分析风格克制专业。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.65,
        max_tokens=1600,
        prefer=prefer,
    )


def render_evening_recap(
    api_key: str = "",
    context: Dict[str, Any] = None,
    *,
    prefer: str = "deepseek",
) -> Optional[str]:
    """调用 LLM 生成 Evening Recap。"""
    if context is None:
        return None
    prompt = EVENING_RECAP_PROMPT.format(**context)
    return _call_llm(
        messages=[
            {"role": "system", "content": "你是专业卖方策略师，输出必须是简体中文，分析风格克制专业。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.65,
        max_tokens=2000,
        prefer=prefer,
    )


# ---------------------------------------------------------------------------
# 7. 历史跨资产对比（修复 NVDA 拆股 + 10Y 量纲）
# ---------------------------------------------------------------------------

HISTORY_TICKERS = {
    "标普500": "^GSPC",
    "SOX半导体": "^SOX",
    "10Y美债收益率": "^TNX",
    "Mag7-MSFT": "MSFT",
    "Mag7-AAPL": "AAPL",
    "Mag7-GOOGL": "GOOGL",
    "Mag7-AMZN": "AMZN",
    "Mag7-NVDA": "NVDA",
    "Mag7-META": "META",
    "Mag7-TSLA": "TSLA",
}


def fetch_history_fixed(period: str = "10y") -> Dict[str, pd.Series]:
    """
    修复版历史数据拉取：
      - auto_adjust=True（关键！解决 NVDA 拆股后复权问题）
      - 10Y 用 (^TNX) 是收益率(%)，不能跟价格一起归一化；调用方按 mode 分别处理
      - 兜底：拉取失败 → 用上一交易日数据
    """
    out: Dict[str, pd.Series] = {}
    for name, sym in HISTORY_TICKERS.items():
        try:
            df = yf.download(sym, period=period, progress=False, auto_adjust=True)
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close = df["Close"].dropna()
            if close.empty:
                continue
            close.name = name
            out[name] = close
        except Exception as e:  # noqa: BLE001
            logger.warning("历史 %s 拉取失败: %s", sym, e)
    return out


# ---------------------------------------------------------------------------
# 8. 简易市场状态判断（开盘/收盘/午休）
# ---------------------------------------------------------------------------

def market_status_now() -> Dict[str, str]:
    """
    返回 US / HK / CN 当前市场状态（Open/Closed/Pre/Post）。
    基于 UTC 时间 + 简单时段判断（不区分节假日，但足够用于 Dashboard 状态条）。
    """
    now_utc = datetime.utcnow()
    weekday = now_utc.weekday()  # 0=Mon
    if weekday >= 5:
        return {"us": "Closed (周末)", "hk": "Closed (周末)", "cn": "Closed (周末)"}

    h = now_utc.hour
    m = now_utc.minute
    minutes = h * 60 + m

    # 美股 (UTC): 夏令时 13:30-20:00; 冬令时 14:30-21:00 (7-10月用夏令时)
    is_summer = 3 <= now_utc.month <= 10
    us_open = 13 * 60 + 30 if is_summer else 14 * 60 + 30
    us_close = 20 * 60 if is_summer else 21 * 60

    # 港股 (UTC): 夏令时 01:30-08:00; 冬令时 02:30-09:00
    hk_open = 1 * 60 + 30 if is_summer else 2 * 60 + 30
    hk_close = 8 * 60 if is_summer else 9 * 60

    def state(open_m: int, close_m: int) -> str:
        if minutes < open_m - 30:
            return "Pre-Market"
        if minutes < open_m:
            return "Pre-Open"
        if minutes < close_m:
            return "Open"
        if minutes < close_m + 30:
            return "Just Closed"
        return "Closed"

    return {
        "us": state(us_open, us_close),
        "hk": state(hk_open, hk_close),
        "cn": "Closed" if minutes < 1 * 60 + 30 or minutes > 7 * 60 else "Open",  # A 股 UTC 01:30-07:00
        "now_utc": now_utc.strftime("%Y-%m-%d %H:%M UTC"),
    }


# ---------------------------------------------------------------------------
# 9. 工具函数：本地化 hash（供 streamlit cache_data 用）
# ---------------------------------------------------------------------------

def hash_for_cache(obj: Any) -> str:
    """给 streamlit @st.cache_data 提供稳定 hash。"""
    return hashlib.md5(repr(obj).encode("utf-8")).hexdigest()


# ===========================================================================
# v2.1 新增模块（在末尾追加，老代码不动）
# ===========================================================================

# ---------------------------------------------------------------------------
# 10. 2-Year Real-Time Scorecard（2Y 国债实时评分卡）
# ---------------------------------------------------------------------------
# 2Y 国债是 Fed 利率预期最敏感的代理，与 10Y 利差 (2s10s) 是衰退先行指标。
# 倒挂 (<0) → 衰退预警；走陡 → 复苏信号。

def fetch_2y_scorecard() -> Dict[str, Any]:
    """
    2-Year Real-Time Scorecard:
      - 2Y 当前收益率 (FRED DGS2)
      - 2Y 实际利率 ≈ DGS2 − T5YIE（5 年期盈亏平衡通胀，近似替代 2Y 通胀预期，
        因 FRED 无长期 T2YIE 序列；界面须标注"近似"）
      - 10Y 收益率
      - 2s10s 利差
      - 利差走势（5 日变化）
    """
    out: Dict[str, Any] = {
        "y2": None, "y10": None, "real_y2": None, "real_y2_note": "近似（DGS2−T5YIE）",
        "spread_bps": None, "spread_5d_chg": None, "signal": "—", "asof": "",
    }
    try:
        # 10Y 用 ^TNX（百分比 → /100 转成收益率）
        tnx_df = yf.download("^TNX", period="6mo", progress=False)
        tnx_close = _normalize_close_series(tnx_df)
        if tnx_close is not None and len(tnx_close) >= 5:
            y10_now = float(tnx_close.iloc[-1]) / 100.0
            y10_5d_ago = float(tnx_close.iloc[-5]) / 100.0
            out["y10"] = round(y10_now * 100, 3)  # 显示为 %
            out["y10_5d_chg_bps"] = round((y10_now - y10_5d_ago) * 10000, 1)

        # 2Y 用 FRED DGS2 (2-Year Treasury Constant Maturity Rate)
        try:
            from fredapi import Fred
            fred_key = _get_secret("FRED_API")
            if fred_key:
                fred = Fred(api_key=fred_key)
                dgs2 = fred.get_series("DGS2", observation_start=(datetime.now() - timedelta(days=30)))
                if dgs2 is not None and len(dgs2) > 0:
                    dgs2 = dgs2.dropna()
                    out["y2"] = round(float(dgs2.iloc[-1]), 3)
                    if out["y10"] is not None:
                        out["spread_bps"] = round((out["y10"] - out["y2"]) * 100, 1)
                    if len(dgs2) >= 5:
                        out["spread_5d_chg"] = round((float(dgs2.iloc[-1]) - float(dgs2.iloc[-5])) * 100, 1)
                    out["asof"] = str(dgs2.index[-1].date())
                # 2Y 实际利率 ≈ DGS2 − T5YIE（5Y 盈亏平衡通胀）
                t5yie = fred.get_series("T5YIE", observation_start=(datetime.now() - timedelta(days=30)))
                if t5yie is not None and len(t5yie) > 0:
                    t5yie = t5yie.dropna()
                    out["real_y2"] = round(float(dgs2.iloc[-1]) - float(t5yie.iloc[-1]), 3)
        except Exception as e:  # noqa: BLE001
            logger.debug("FRED DGS2/T5YIE 失败: %s", e)
            out["y2"] = None

        # 信号判断
        if out["spread_bps"] is not None:
            sp = out["spread_bps"]
            if sp < 0:
                out["signal"] = "🔴 倒挂 (衰退预警)"
            elif sp < 25:
                out["signal"] = "🟠 接近倒挂 (风险)"
            elif sp < 75:
                out["signal"] = "🟡 正常 (中性)"
            else:
                out["signal"] = "🟢 走陡 (复苏/降息预期)"
    except Exception as e:  # noqa: BLE001
        logger.warning("2Y Scorecard 失败: %s", e)
        out["error"] = str(e)
    return out


# ---------------------------------------------------------------------------
# 11. U.S. National Debt（联邦政府总债务）
# ---------------------------------------------------------------------------
# FRED: GFDEBTN (Federal Debt: Total Public Debt, 百万美元)
# 1.6 万亿美元 → 实际值约 36 万亿 (2025-2026)

def fetch_us_debt() -> Dict[str, Any]:
    """从 FRED 拉联邦总债务 (GFDEBTN)。"""
    out: Dict[str, Any] = {"value_trillion": None, "yoy_chg_pct": None, "asof": ""}
    try:
        from fredapi import Fred
        fred_key = _get_secret("FRED_API")
        if not fred_key:
            return {**out, "error": "未配置 FRED_API"}
        fred = Fred(api_key=fred_key)
        # GFDEBTN 单位是百万美元
        s = fred.get_series("GFDEBTN", observation_start=(datetime.now() - timedelta(days=400)))
        if s is None or len(s) < 2:
            return {**out, "error": "FRED GFDEBTN 数据为空"}
        s = s.dropna()
        out["value_trillion"] = round(float(s.iloc[-1]) / 1_000_000, 3)  # 转成万亿
        out["asof"] = str(s.index[-1].date())
        # 同比
        if len(s) >= 252:
            last_year = s.iloc[-252] if len(s) >= 252 else s.iloc[0]
            yoy = (float(s.iloc[-1]) / float(last_year) - 1.0) * 100
            out["yoy_chg_pct"] = round(yoy, 2)
    except Exception as e:  # noqa: BLE001
        logger.warning("US Debt 拉取失败: %s", e)
        out["error"] = str(e)
    return out


# ---------------------------------------------------------------------------
# 12. FINRA Retail Margin Debt（融资余额）
# ---------------------------------------------------------------------------
# 主源：FRED MDEBT (Margin Debt, All Customers, FINRA)。
# ⚠️ 重要：FRED 已于 2024 年前后停用 MDEBT 序列（访问 series/MDEBT 返回 404），
#   因此主源不可用时必须降级到 FINRA 官网 margin-statistics 页面 HTML 表格直抓
#   （官网无公开 API feed，页面表格每月第三个星期更新，与 FRED 同源、官方直出）。
#   两源交叉校验，取最新。

_FINRA_MARGIN_URL = "https://www.finra.org/investors/learn-to-invest/advanced-investing/margin-statistics"


def fetch_finra_margin_web() -> Dict[str, Any]:
    """
    直接抓取 FINRA 官网 margin-statistics 页面 HTML 表格。
    返回 {ok, rows:[("Jun-26", 1502072.0), ...], value_billion, month_label, asof}，
    失败返回 {"error": ...}。rows 中数值单位为 $ millions，按时间倒序（最新在前）。
    """
    out: Dict[str, Any] = {"ok": False, "rows": [], "value_billion": None, "month_label": "", "asof": "", "error": None}
    try:
        import pandas as pd  # noqa: F811

        tables = pd.read_html(_FINRA_MARGIN_URL, flavor="bs4")
        for tbl in tables:
            cols = [str(c) for c in tbl.columns]
            lower = [c.lower() for c in cols]
            # 定位 Debit Balances 列（排除 Free Credit Balances 干扰列）
            debit_idx = None
            for i, c in enumerate(lower):
                if "debit" in c and "credit" not in c:
                    debit_idx = i
                    break
            if debit_idx is None:
                continue
            flat = tbl.dropna(subset=[tbl.columns[0]]).copy()
            if flat.empty:
                continue
            rows = []
            for _, row in flat.iterrows():
                m = str(row.iloc[0]).strip()
                try:
                    v = float(str(row.iloc[debit_idx]).replace(",", "").replace("$", "").strip())
                except (TypeError, ValueError):
                    continue
                rows.append((m, v))
            if rows:
                out["rows"] = rows
                out["ok"] = True
                out["value_billion"] = round(rows[0][1] / 1000.0, 2)  # $ millions → $ billions
                out["month_label"] = rows[0][0]
                out["asof"] = rows[0][0]
                return out
        out["error"] = "未找到匹配的 margin 表格"
    except Exception as e:  # noqa: BLE001
        out["error"] = f"FINRA 页面抓取失败: {e}"
        logger.warning("FINRA margin 页面抓取失败: %s", e)
    return out


def fetch_margin_debt() -> Dict[str, Any]:
    """
    FINRA 融资余额（散户保证金债务）。
    主源：FRED MDEBT（注意：FRED 已停用 MDEBT 序列，调用会抛 404）。
    降级：MDEBT 不可用时直接抓取 FINRA 官网 margin-statistics 表格（与 FRED 同源、官方直出），
         并从表格近 13 个月数据计算环比/同比。
    返回 {value_billion, mom_chg_pct, yoy_chg_pct, signal, asof, source}。
    """
    out: Dict[str, Any] = {
        "value_billion": None, "mom_chg_pct": None, "yoy_chg_pct": None,
        "signal": "—", "asof": "", "source": "FRED MDEBT",
    }
    try:
        fred_key = _get_secret("FRED_API")
        fred_series = None
        if fred_key:
            try:
                from fredapi import Fred
                fred = Fred(api_key=fred_key)
                s = fred.get_series("MDEBT", observation_start=(datetime.now() - timedelta(days=400)))
                if s is not None and len(s.dropna()) >= 2:
                    fred_series = s.dropna()
            except Exception as e:  # noqa: BLE001
                logger.warning("FRED MDEBT 不可用(或已停更): %s", e)

        if fred_series is not None:
            out["value_billion"] = round(float(fred_series.iloc[-1]) / 1000.0, 2)  # 百万 → 十亿
            out["asof"] = str(fred_series.index[-1].date())
            out["source"] = "FRED MDEBT"
            if len(fred_series) >= 2:
                mom = (float(fred_series.iloc[-1]) / float(fred_series.iloc[-2]) - 1.0) * 100
                out["mom_chg_pct"] = round(mom, 2)
            if len(fred_series) >= 12:
                yoy = (float(fred_series.iloc[-1]) / float(fred_series.iloc[-12]) - 1.0) * 100
                out["yoy_chg_pct"] = round(yoy, 2)
            # 交叉校验：FINRA 官网（官方页面往往比 FRED 早几天）
            try:
                web = fetch_finra_margin_web()
                if web.get("ok") and web.get("value_billion") is not None:
                    out["finra_web_value"] = web["value_billion"]
                    out["finra_web_month"] = web["asof"]
            except Exception:  # noqa: BLE001
                pass
        else:
            # 降级：FINRA 官网直抓（MDEBT 停用后的唯一官方源）
            web = fetch_finra_margin_web()
            if web.get("ok") and web.get("value_billion") is not None:
                out["value_billion"] = web["value_billion"]
                out["asof"] = web["asof"]
                out["source"] = "FINRA官网直抓"
                rows = web.get("rows") or []
                vals = [r[1] for r in rows]  # $ millions，倒序（最新在前）
                if len(vals) >= 2:
                    out["mom_chg_pct"] = round((vals[0] / vals[1] - 1.0) * 100, 2)
                if len(vals) >= 13:
                    out["yoy_chg_pct"] = round((vals[0] / vals[12] - 1.0) * 100, 2)
            else:
                return {**out, "error": web.get("error") or "FINRA 官网抓取失败"}

        # 信号：融资余额快速上升 = 散户加杠杆（牛市后期）；快速下降 = 强平/恐慌
        yoy = out.get("yoy_chg_pct")
        if yoy is not None:
            if yoy > 30:
                out["signal"] = "🔴 融资激增 (杠杆高)"
            elif yoy > 10:
                out["signal"] = "🟠 融资扩张"
            elif yoy < -15:
                out["signal"] = "🟢 融资去杠杆"
            else:
                out["signal"] = "🟡 平稳"
    except Exception as e:  # noqa: BLE001
        logger.warning("Margin Debt 拉取失败: %s", e)
        out["error"] = str(e)
    return out


# ---------------------------------------------------------------------------
# 13. Chicago Fed NFCI Leverage Subindex
# ---------------------------------------------------------------------------
# 芝加哥联储国家金融状况指数 - 杠杆子指数
# 数据源：FRED: NFCILEVERAGE
# < 0 = 金融状况宽松（杠杆可获得）；> 0 = 紧缩

def fetch_nfci_leverage() -> Dict[str, Any]:
    """Chicago Fed NFCI Leverage Subindex。"""
    out: Dict[str, Any] = {"value": None, "prev": None, "chg": None, "signal": "—", "asof": ""}
    try:
        from fredapi import Fred
        fred_key = _get_secret("FRED_API")
        if not fred_key:
            return {**out, "error": "未配置 FRED_API"}
        fred = Fred(api_key=fred_key)
        s = fred.get_series("NFCILEVERAGE", observation_start=(datetime.now() - timedelta(days=400)))
        if s is None or len(s) < 2:
            return {**out, "error": "FRED NFCILEVERAGE 数据为空"}
        s = s.dropna()
        out["value"] = round(float(s.iloc[-1]), 3)
        out["prev"] = round(float(s.iloc[-2]), 3)
        out["chg"] = round(out["value"] - out["prev"], 3)
        out["asof"] = str(s.index[-1].date())
        v = out["value"]
        if v > 0.5:
            out["signal"] = "🔴 杠杆紧缩"
        elif v > 0:
            out["signal"] = "🟠 轻度紧缩"
        elif v > -0.5:
            out["signal"] = "🟡 宽松"
        else:
            out["signal"] = "🟢 极度宽松"
    except Exception as e:  # noqa: BLE001
        logger.warning("NFCI Leverage 拉取失败: %s", e)
        out["error"] = str(e)
    return out


def fetch_fred_linkage_series(days: int = 400) -> Dict[str, Any]:
    """
    FRED 三条序列联动（美债规模 GFDEBTN / 融资余额 MDEBT / NFCI 杠杆子指数）。
    返回按公共日期对齐的归一化对比：debt_pct / margin_pct（相对各自起点变化 %），nfci（原值）。
    """
    out: Dict[str, Any] = {"ok": False, "dates": [], "debt_pct": [], "margin_pct": [], "nfci": [], "error": ""}
    try:
        from fredapi import Fred
        fred_key = _get_secret("FRED_API")
        if not fred_key:
            out["error"] = "未配置 FRED_API"
            return out
        fred = Fred(api_key=fred_key)
        start = datetime.now() - timedelta(days=days)
        s_debt = fred.get_series("GFDEBTN", observation_start=start).dropna()
        s_margin = fred.get_series("MDEBT", observation_start=start).dropna()
        s_nfci = fred.get_series("NFCILEVERAGE", observation_start=start).dropna()
        idx = s_debt.index.intersection(s_margin.index).intersection(s_nfci.index)
        if len(idx) < 2:
            out["error"] = "FRED 公共日期不足（三序列对齐后 < 2 个点）"
            return out
        debt = s_debt[idx]
        margin = s_margin[idx]
        nfci = s_nfci[idx]
        debt0, margin0 = float(debt.iloc[0]), float(margin.iloc[0])
        out["dates"] = [d.strftime("%Y-%m-%d") for d in idx]
        out["debt_pct"] = [(float(x) / debt0 - 1.0) * 100 for x in debt] if debt0 else []
        out["margin_pct"] = [(float(x) / margin0 - 1.0) * 100 for x in margin] if margin0 else []
        out["nfci"] = [float(x) for x in nfci]
        out["ok"] = True
    except Exception as e:  # noqa: BLE001
        logger.warning("FRED 联动序列拉取失败: %s", e)
        out["error"] = str(e)
    return out


# ---------------------------------------------------------------------------
# 14. Vol / OI PCR（每只个股的期权 PCR）
# ---------------------------------------------------------------------------
# 用 yfinance 的 option_chain() 抓最近一期到期的 calls/puts DataFrame。
# 汇总所有 strikes 的 volume / openInterest 后计算 put/call 比。

def fetch_options_pcr(symbol: str, max_exp_days: int = 60) -> Dict[str, Any]:
    """
    返回该 symbol 的 Vol PCR / OI PCR / 最近期权到期日 / 隐含波动率均值。
    """
    out: Dict[str, Any] = {
        "symbol": symbol,
        "expiry": None,
        "call_volume": 0, "put_volume": 0,
        "call_oi": 0, "put_oi": 0,
        "vol_pcr": None, "oi_pcr": None,
        "iv_call": None, "iv_put": None,
        "error": None,
    }
    try:
        # 数据源能力边界：yfinance 不覆盖港股/A股个股期权链（HKEX/沪深个股期权），
        # 提前短路并返回结构化错误码，避免每次白跑网络请求且让前端能区分"源不支持"。
        if symbol.endswith((".HK", ".SS", ".SZ")):
            out["error"] = "market_not_supported"
            return out
        t = yf.Ticker(symbol)
        expirations = t.options or []
        if not expirations:
            out["error"] = "no_expiry"
            return out
        # 选最近一期（且在 60 天内）
        today = datetime.now().date()
        chosen = None
        for exp in expirations[:6]:
            try:
                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                if 0 < (exp_date - today).days <= max_exp_days:
                    chosen = exp
                    break
            except Exception:  # noqa: BLE001
                continue
        if chosen is None:
            chosen = expirations[0]
        chain = t.option_chain(chosen)
        calls = chain.calls
        puts = chain.puts
        if calls is None or puts is None or calls.empty or puts.empty:
            out["error"] = "empty_chain"
            return out
        out["expiry"] = chosen
        out["call_volume"] = int(calls["volume"].fillna(0).sum())
        out["put_volume"] = int(puts["volume"].fillna(0).sum())
        out["call_oi"] = int(calls["openInterest"].fillna(0).sum())
        out["put_oi"] = int(puts["openInterest"].fillna(0).sum())
        if out["call_volume"] > 0:
            out["vol_pcr"] = round(out["put_volume"] / out["call_volume"], 3)
        if out["call_oi"] > 0:
            out["oi_pcr"] = round(out["put_oi"] / out["call_oi"], 3)
        if "impliedVolatility" in calls.columns:
            out["iv_call"] = round(float(calls["impliedVolatility"].mean()), 3)
        if "impliedVolatility" in puts.columns:
            out["iv_put"] = round(float(puts["impliedVolatility"].mean()), 3)
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)
        logger.warning("%s PCR 抓取失败: %s", symbol, e)
    return out


def fetch_all_pcr(symbols: List[str], out_path: Optional[Path] = None) -> Dict[str, Any]:
    """并发拉多只股票的 PCR。"""
    import concurrent.futures
    results: Dict[str, Any] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(fetch_options_pcr, sym): sym for sym in symbols}
        for f in concurrent.futures.as_completed(futs):
            r = f.result()
            results[r["symbol"]] = r
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


# ---------------------------------------------------------------------------
# 15. 多源新闻聚合（免费 + 付费 fallback）
# ---------------------------------------------------------------------------
# SerpApi 仍是主力，但提供 5 个免费 fallback，让用户能"零成本"启动。

def _safe_get_json(url: str, params: Dict[str, Any] = None, headers: Dict[str, str] = None, timeout: int = 10) -> Optional[Dict]:
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:  # noqa: BLE001
        logger.debug("HTTP %s 失败: %s", url, e)
        return None


def fetch_yahoo_rss(query: str = "", ticker: str = "") -> List[Dict[str, Any]]:
    """
    Yahoo Finance RSS（完全免费，无 key）。
    例如：
      - https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US
      - https://news.yahoo.com/rss/search?p=Fed+CPI
    """
    out: List[Dict[str, Any]] = []
    try:
        if ticker:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        else:
            from urllib.parse import quote_plus
            url = f"https://news.yahoo.com/rss/search?p={quote_plus(query)}"
        # 用 feedparser 解析（轻量，streamlit 友好）
        try:
            import feedparser
            feed = feedparser.parse(url)
            for e in feed.entries[:8]:
                out.append({
                    "title": e.get("title", ""),
                    "link": e.get("link", ""),
                    "source": "Yahoo Finance",
                    "date": e.get("published", ""),
                    "snippet": e.get("summary", "")[:200],
                })
        except ImportError:
            # 没有 feedparser 就用 requests + xml 解析
            r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            import xml.etree.ElementTree as ET
            root = ET.fromstring(r.text)
            for item in root.iter("item")[:8]:
                out.append({
                    "title": item.findtext("title", ""),
                    "link": item.findtext("link", ""),
                    "source": "Yahoo Finance",
                    "date": item.findtext("pubDate", ""),
                    "snippet": item.findtext("description", "")[:200],
                })
    except Exception as e:  # noqa: BLE001
        logger.debug("Yahoo RSS 失败: %s", e)
    return out


def fetch_finnhub_news(symbol: str, api_key: str = "", days_back: int = 7) -> List[Dict[str, Any]]:
    """
    Finnhub 免费 API（60 calls/min，有 key 时推荐）。
    文档: https://finnhub.io/docs/api/company-news
    """
    if not api_key:
        return []
    key = _get_secret("FINNHUB_API") or api_key
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        url = f"https://finnhub.io/api/v1/company-news"
        params = {"symbol": symbol, "from": start, "to": today, "token": key}
        data = _safe_get_json(url, params=params)
        if not data:
            return []
        return [
            {
                "title": n.get("headline", ""),
                "link": n.get("url", ""),
                "source": n.get("source", "Finnhub"),
                "date": datetime.fromtimestamp(n.get("datetime", 0)).strftime("%Y-%m-%d %H:%M"),
                "snippet": n.get("summary", "")[:200],
            }
            for n in data[:8]
        ]
    except Exception as e:  # noqa: BLE001
        logger.warning("Finnhub %s 失败: %s", symbol, e)
        return []


def fetch_newsapi(query: str, api_key: str = "", days_back: int = 3, page_size: int = 8) -> List[Dict[str, Any]]:
    """
    NewsAPI.org (newsapi.org) - 免费 100 次/天。
    文档: https://newsapi.org/docs/endpoints/everything
    """
    if not api_key:
        return []
    key = _get_secret("NEWSAPI_KEY") or api_key
    try:
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        params = {
            "q": query,
            "from": from_date,
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "language": "en",
            "apiKey": key,
        }
        data = _safe_get_json("https://newsapi.org/v2/everything", params=params)
        if not data or data.get("status") != "ok":
            return []
        return [
            {
                "title": n.get("title", ""),
                "link": n.get("url", ""),
                "source": n.get("source", {}).get("name", "NewsAPI"),
                "date": n.get("publishedAt", ""),
                "snippet": n.get("description", "")[:200],
            }
            for n in data.get("articles", [])
        ]
    except Exception as e:  # noqa: BLE001
        logger.warning("NewsAPI '%s' 失败: %s", query, e)
        return []


def fetch_stocktwits(symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Stocktwits API（完全免费，无 key，但需要 User-Agent 头）。
    适合抓散户情绪。
    """
    out: List[Dict[str, Any]] = []
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        data = _safe_get_json(url, headers={"User-Agent": "Mozilla/5.0"})
        if not data:
            return out
        for m in (data.get("messages") or [])[:limit]:
            out.append({
                "title": (m.get("body", "") or "")[:100],
                "link": f"https://stocktwits.com/symbol/{symbol}",
                "source": f"Stocktwits · @{m.get('user', {}).get('username', '?')}",
                "date": m.get("created_at", ""),
                "snippet": (m.get("body", "") or "")[:200],
            })
    except Exception as e:  # noqa: BLE001
        logger.debug("Stocktwits %s 失败: %s", symbol, e)
    return out


def fetch_eastmoney_stock_news(symbol: str) -> List[Dict[str, Any]]:
    """
    A 股个股新闻（东方财富个股快讯，完全免费，无 key，按个股精确过滤）。
    接口：np-listapi.eastmoney.com/comm/wap/getListInfo，用 mTypeAndCode 指定个股：
      - 沪市 → 1.XXXXXX  - 深市 → 0.XXXXXX
    返回标准新闻条目：{title, link, source, date, snippet}（link 为可点击全文 URL）。
    说明：原 ann（公告）接口已不可用（恒返回空），本函数改用可正常按个股过滤的快讯接口。
    """
    out: List[Dict[str, Any]] = []
    if not symbol.endswith((".SS", ".SZ")):
        return out  # 仅 A 股
    code = symbol.replace(".SS", "").replace(".SZ", "")
    mkt = "1" if symbol.endswith(".SS") else "0"
    try:
        url = "https://np-listapi.eastmoney.com/comm/wap/getListInfo"
        params = {"client": "wap", "type": 1, "mTypeAndCode": f"{mkt}.{code}", "pageSize": 10, "pageIndex": 1}
        data = _safe_get_json(url, params=params)
        items = []
        if isinstance(data, dict):
            items = (data.get("data") or {}).get("list", []) or []
        for it in items[:10]:
            title = it.get("Art_Title") or it.get("title") or ""
            if not title:
                continue
            link = it.get("Art_Url") or it.get("Art_OriginUrl") or it.get("url") or ""
            out.append({
                "title": title,
                "link": link,
                "source": it.get("Art_MediaName") or it.get("source") or "东方财富网",
                "date": it.get("Art_ShowTime") or it.get("showTime") or it.get("date") or "",
                "snippet": it.get("Art_Summary") or it.get("summary") or "",
            })
    except Exception as e:  # noqa: BLE001
        logger.debug("EastMoney 个股新闻 %s 失败: %s", symbol, e)
    return out


def fetch_cninfo_market_announcements(top_n: int = 10) -> List[Dict[str, Any]]:
    """
    巨潮资讯网（交易所官方）全市场最新公告 —— A 股最权威的公告源。

    重要说明：cninfo 公开接口（hisAnnouncement/query）对「按个股过滤」不稳定——
    传入 stockCode/orgId 仍返回全市场最新公告（已实测：totalAnnouncement≈47 万，
    返回条目为其他股票，且 orgId 查询接口目前 404）。因此本函数直接以「全市场公告
    快讯」形态使用，这正是该接口最可靠、最有价值的输出，与东方财富「个股快讯」互补：
       · 个股深度页 → 用 fetch_eastmoney_stock_news（按股精准）
       · 大盘/新闻中心 → 用本函数（官方全市场公告，权威且稳定）
    """
    out: List[Dict[str, Any]] = []
    try:
        url = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
        payload = {"pageNum": 1, "pageSize": top_n, "tabName": "latest"}
        data = _safe_post_json(url, payload)
        if not data:
            return out
        for a in (data.get("announcements") or [])[:top_n]:
            title = a.get("announcementTitle") or ""
            if not title:
                continue
            sec_name = a.get("secName") or ""
            aid = a.get("announcementId") or ""
            out.append({
                "title": (f"{sec_name}：" if sec_name else "") + title,
                "link": f"https://www.cninfo.com.cn/new/disclosure/detail?announcementId={aid}" if aid else "",
                "source": "巨潮资讯网(官方)",
                "date": _cninfo_ts(a.get("announcementTime")),
                "snippet": "",
            })
    except Exception as e:  # noqa: BLE001
        logger.debug("cninfo 全市场公告失败: %s", e)
    return out


def _cninfo_ts(ms) -> str:
    """巨潮公告时间戳（毫秒）转 'YYYY-MM-DD'。"""
    try:
        return datetime.fromtimestamp(int(ms) / 1000).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return ""


def _safe_post_json(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None, timeout: int = 10) -> Optional[Dict]:
    """POST 并解析 JSON（与 _safe_get_json 对称）。"""
    try:
        h = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/", "Content-Type": "application/x-www-form-urlencoded"}
        if headers:
            h.update(headers)
        resp = requests.post(url, data=payload, headers=h, timeout=timeout)
        return resp.json()
    except Exception as e:  # noqa: BLE001
        logger.debug("POST %s 失败: %s", url, e)
        return None


def fetch_eastmoney_global_news(top_n: int = 15) -> List[Dict[str, Any]]:
    """
    东方财富全球财经新闻（完全免费）。
    https://np-listapi.eastmoney.com/comm/wap/getListInfo?cb=&client=wap&type=1&mTypeAndCode=&pageSize=20&pageIndex=1&callback=&_=
    """
    out: List[Dict[str, Any]] = []
    try:
        url = "https://np-listapi.eastmoney.com/comm/wap/getListInfo"
        params = {"client": "wap", "type": 1, "mTypeAndCode": "", "pageSize": top_n, "pageIndex": 1}
        data = _safe_get_json(url, params=params)
        if not data:
            return out
        # 新版东财 API 结构可能变化；做最宽松的解析
        items = []
        if isinstance(data.get("data"), dict):
            items = data["data"].get("list", []) or []
        elif isinstance(data.get("data"), list):
            items = data["data"]
        for item in items[:top_n]:
            out.append({
                "title": item.get("Art_Title") or item.get("title", ""),
                "link": item.get("Art_Url") or item.get("url", ""),
                "source": "东方财富网 · 全球财经",
                "date": item.get("Art_Time") or item.get("showTime", ""),
                "snippet": item.get("Art_Abstract", "") or item.get("digest", ""),
            })
    except Exception as e:  # noqa: BLE001
        logger.debug("EastMoney global 失败: %s", e)
    return out


# ---------------------------------------------------------------------------
# 6.x 雪球 / 同花顺 / 无 API 的 RSS 源（美股 / 港股 / A 股）
# ---------------------------------------------------------------------------

def fetch_rss_feed(url: str, source_name: str, top_n: int = 10, timeout: int = 12) -> List[Dict[str, Any]]:
    """通用 RSS 抓取（用 feedparser）。完全免费、无 key。"""
    out: List[Dict[str, Any]] = []
    try:
        import feedparser
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        feed = feedparser.parse(resp.content)
        for e in feed.entries[:top_n]:
            out.append({
                "title": (e.get("title") or "").strip(),
                "link": e.get("link", ""),
                "source": source_name,
                "date": e.get("published", e.get("updated", "")),
                "snippet": (e.get("summary", "") or e.get("description", ""))[:200],
            })
    except Exception as e:  # noqa: BLE001
        logger.debug("RSS[%s] 失败: %s", source_name, e)
    return out


def fetch_10jqka_news(top_n: int = 12) -> List[Dict[str, Any]]:
    """
    同花顺快讯（完全免费、无 key、无 cookie）。
    API: http://kuaixun.10jqka.com.cn/api/kuaixun/1
    """
    out: List[Dict[str, Any]] = []
    try:
        data = _safe_get_json("http://kuaixun.10jqka.com.cn/api/kuaixun/1", headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        items = (data or {}).get("data", []) or (data or [])
        if isinstance(items, dict):
            items = items.get("list", [])
        for it in items[:top_n]:
            if not isinstance(it, dict):
                continue
            title = (it.get("title") or (it.get("content") or "")[:50] or "").strip()
            out.append({
                "title": title,
                "link": it.get("url") or it.get("detailurl") or "http://kuaixun.10jqka.com.cn/",
                "source": "同花顺快讯",
                "date": str(it.get("time", it.get("date", ""))),
                "snippet": (it.get("content") or it.get("digest") or "")[:200],
            })
    except Exception as e:  # noqa: BLE001
        logger.debug("同花顺快讯失败: %s", e)
    return out


def fetch_xueqiu_news(top_n: int = 12) -> List[Dict[str, Any]]:
    """
    雪球热帖（完全免费，但雪球接口通常需要 cookie，成功率不保证；失败自动跳过）。
    抓 https://xueqiu.com/statuses/hot/list.json
    """
    out: List[Dict[str, Any]] = []
    try:
        import re
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Referer": "https://xueqiu.com/"})
        try:
            s.get("https://xueqiu.com/", timeout=8)
        except Exception:  # noqa: BLE001
            pass
        r = s.get("https://xueqiu.com/statuses/hot/list.json?page=1&pre_page=20", timeout=12)
        r.raise_for_status()
        data = r.json()
        items = (data or {}).get("list", []) or []
        for it in items[:top_n]:
            if not isinstance(it, dict):
                continue
            txt = re.sub(r"<[^>]+>", "", (it.get("text") or "")).strip()
            if not txt:
                continue
            uid = (it.get("user") or {}).get("id", "")
            out.append({
                "title": txt[:80],
                "link": f"https://xueqiu.com/{uid}/{it.get('id', '')}",
                "source": f"雪球 · {(it.get('user') or {}).get('screen_name', '')}",
                "date": str(it.get("created_at", "")),
                "snippet": txt[:200],
            })
    except Exception as e:  # noqa: BLE001
        logger.debug("雪球失败(可能需cookie): %s", e)
    return out


# ---------------------------------------------------------------------------
# 新闻启发式解读（确定性、无需联网/密钥）
# 说明：基于标题关键词做多空情绪判断，非语义级深度分析，也不构成投顾建议。
# ---------------------------------------------------------------------------
def _parse_news_date(s: str) -> Optional[datetime]:
    """尽力解析新闻日期，支持 RFC822(pubDate)、ISO、常见格式、相对时间。

    返回 None 表示无法解析（调用方会将其归入「日期未知」）。
    """
    if not s:
        return None
    s = str(s).strip()
    if not s:
        return None
    # 相对时间（如 "3小时前" / "2 days ago"）→ 近似为 now 之前
    m = re.search(r"(\d+)\s*(分钟|小时|天|min|hour|day|小时前|分钟前)", s, re.IGNORECASE)
    if m and ("前" in s or "ago" in s.lower()):
        num = int(m.group(1))
        unit = m.group(2).lower()
        if "分钟" in unit or unit == "min":
            return datetime.now() - timedelta(minutes=num)
        if "小时" in unit or unit == "hour":
            return datetime.now() - timedelta(hours=num)
        return datetime.now() - timedelta(days=num)
    # RFC822（Yahoo RSS: "Wed, 29 Jul 2026 12:00:00 GMT"）
    try:
        from email.utils import parsedate_to_datetime  # 局部导入，避免污染顶层

        dt = parsedate_to_datetime(s)
        if dt is not None:
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            return dt
    except Exception:  # noqa: BLE001
        pass
    # 常见固定格式
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
    ):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    # ISO 8601（兜底）
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:  # noqa: BLE001
        pass
    return None


# 看多 / 看空 关键词（中英文，覆盖标题常见表述）
_BULL_KW = [
    "涨", "升", "利好", "超预期", "上调", "买入", "增持", "增长", "突破", "盈利",
    "创新高", "新高", "合作", "中标", "获批", "复苏", "反弹", "回暖", "强劲",
    "beat", "upgrade", "buy", "rally", "surge", "record", "partnership",
    "approval", "strong", "growth", "profit", "outperform", "看多", "扩容", "中标",
]
_BEAR_KW = [
    "跌", "降", "利空", "不及预期", "下调", "减持", "卖出", "亏损", "裁员", "调查",
    "诉讼", "风险", "监管", "破位", "暴雷", "下滑", "承压", "警告", "召回", "被查",
    "miss", "downgrade", "sell", "lawsuit", "probe", "fine", "layoff", "loss",
    "decline", "weak", "warning", "risk", "recall", "underperform", "看空", "违约", "亏损",
]


def interpret_news(
    sym: str, news_list: List[Dict[str, Any]], within_days: int = 2
) -> Dict[str, Any]:
    """对个股近 N 日新闻做**启发式**解读（基于标题关键词，确定性、无需联网/密钥）。

    注意：这是基于标题关键词的机器解读，非投顾建议，也不是深度语义分析。
    返回结构：
        {
          "total": int, "near_total": int, "has_near": bool,
          "tone": "看多" | "看空" | "中性" | "信息不足",
          "score": float,            # 多空净分 [-1,1]
          "positives": List[str],    # 偏多标题（Top3）
          "negatives": List[str],    # 偏空标题（Top3）
          "summary": str,            # 分析师口吻的一句话总结
        }
    """
    result: Dict[str, Any] = {
        "total": 0, "near_total": 0, "has_near": False,
        "tone": "信息不足", "score": 0.0,
        "positives": [], "negatives": [], "summary": "",
    }
    if not news_list:
        result["summary"] = "近 2 日未抓取到有效新闻，无法生成解读。"
        return result

    now = datetime.now()
    near_items: List[Dict[str, Any]] = []
    for n in news_list:
        d = _parse_news_date(n.get("date", ""))
        if d is not None and 0 <= (now - d).days <= within_days:
            near_items.append(n)
    result["total"] = len(news_list)
    result["near_total"] = len(near_items)
    result["has_near"] = len(near_items) > 0

    # 解读样本：优先近 N 日；若不足 2 条则放宽到全部抓取项（仍标注窗口情况）
    sample = near_items if len(near_items) >= 2 else news_list

    b_scores: List[int] = []
    pos: List[str] = []
    neg: List[str] = []
    for n in sample:
        title = (n.get("title") or "")
        if not title:
            continue
        low = title.lower()
        b = sum(1 for k in _BULL_KW if k.lower() in low)
        r = sum(1 for k in _BEAR_KW if k.lower() in low)
        b_scores.append(b - r)
        if b > r:
            pos.append(title)
        elif r > b:
            neg.append(title)

    total_hits = len(b_scores)
    if total_hits == 0:
        result["tone"] = "中性"
        result["summary"] = (
            f"近 2 日共 {result['near_total']} 条窗口内新闻（合计抓取 {result['total']} 条），"
            "标题未呈现明显多空倾向，市场对该标的短期消息面偏中性，建议结合量价与技术位综合判断。"
        )
        return result

    net = sum(b_scores)
    score = net / total_hits
    result["score"] = round(score, 2)
    result["tone"] = "看多" if score > 0.15 else ("看空" if score < -0.15 else "中性")
    result["positives"] = pos[:3]
    result["negatives"] = neg[:3]

    # 焦点句（pos/neg 都可能为空：标题存在但无多空关键词命中时二者皆空，不能假设必有 neg）
    if pos and neg:
        focus = f"多空交织：偏多线索聚焦「{pos[0][:18]}…」，偏空线索聚焦「{neg[0][:18]}…」。"
    elif pos:
        focus = f"消息面整体偏暖，正面催化集中于「{pos[0][:20]}…」。"
    elif neg:
        focus = f"消息面承压，负面风险集中于「{neg[0][:20]}…」。"
    else:
        focus = "近 2 日暂无重大利空消息，消息面中性偏平静，未见明确方向性催化。"

    window_txt = (
        f"近 {within_days} 日窗口内抓取 {result['near_total']} 条"
        if result["has_near"]
        else f"近 {within_days} 日窗口内无明确日期新闻，已对全部 {result['total']} 条抓取项做解读"
    )
    tone_map = {"看多": "偏多", "看空": "偏空", "中性": "中性"}
    result["summary"] = (
        f"{window_txt}；标题情绪 {len(pos)} 条偏多 / {len(neg)} 条偏空，"
        f"综合解读：**消息面{tone_map[result['tone']]}**。"
        f"{focus}"
    )
    return result


# ---------------------------------------------------------------------------
# 实时报价（HK / A股）：东方财富 → 腾讯 gtimg → 新浪 多源冗余
# 用途：覆盖「收盘价 / 涨跌幅」等头条数字，避免依赖可能失真/过期的日线历史
# （akshare qfq 对港股杠杆ETF 等会产生错乱，已实测 7709.HK / 00981.HK 严重偏差）。
# 技术指标（RSI/MA/ATR）仍用日线历史计算。
# ---------------------------------------------------------------------------
def _futu_secret(name: str, default: str = "") -> str:
    """读取 Futu 配置：优先 os.environ，其次 Streamlit Secrets。"""
    v = os.environ.get(name, "")
    if v:
        return v
    try:
        import streamlit as st  # 懒导入，避免无 streamlit 环境报错

        if hasattr(st, "secrets"):
            v = st.secrets.get(name, "")
            if v:
                return str(v)
    except Exception:  # noqa: BLE001
        pass
    return default


def fetch_realtime_via_futu(symbol: str) -> Dict[str, Any]:
    """通过本地/可达的 FutuOpenD 网关获取实时报价（最高优先级）。

    开启条件：USE_FUTU=true 且 OpenD 在运行。
    配置项（os.environ 或 Streamlit Secrets 二选一即可）：
      USE_FUTU        true / false
      FUTU_OPEND_HOST 默认 127.0.0.1
      FUTU_OPEND_PORT 默认 11111
    仅支持 .HK / .SS / .SZ；其余返回 ok=False 由上层免费源兜底。
    返回全字段快照（last/prev_close/open/high/low/volume/amount/turnover/
    amplitude/pe/pb），可同时服务于「收盘价校正」与「实时快照」两条链路。
    任何失败（未安装 / 网关不可达 / 空数据）均返回 ok=False，绝不抛异常。
    """
    out: Dict[str, Any] = {"ok": False, "symbol": symbol, "source": "富途OpenD"}
    if _futu_secret("USE_FUTU", "false").lower() != "true":
        out["error"] = "disabled"
        return out
    is_hk = symbol.endswith(".HK")
    is_cn = symbol.endswith((".SS", ".SZ"))
    if not (is_hk or is_cn):
        out["error"] = "market_not_supported"
        return out
    try:
        from futu import OpenQuoteContext, RET_ERROR
    except Exception:  # noqa: BLE001
        out["error"] = "futu_api_not_installed"
        return out
    # 富途内部代码格式：HK.00700 / SH.600519 / SZ.000001
    if is_hk:
        code = "HK." + symbol.replace(".HK", "").zfill(5)
    elif symbol.endswith(".SS"):
        code = "SH." + symbol.replace(".SS", "")
    else:
        code = "SZ." + symbol.replace(".SZ", "")
    host = _futu_secret("FUTU_OPEND_HOST", "127.0.0.1")
    try:
        port = int(_futu_secret("FUTU_OPEND_PORT", "11111") or "11111")
    except ValueError:
        port = 11111
    ctx = None
    try:
        ctx = OpenQuoteContext(host=host, port=port)
        ret, data = ctx.get_stock_quote([code])
        if ret == RET_ERROR or data is None or data.empty:
            out["error"] = "quote_error"
            return out
        row = data.iloc[0]
        last = _rt_f(row.get("last_price"))
        prev = _rt_f(row.get("prev_close_price"))
        if last is None or prev is None or last <= 0 or prev <= 0:
            out["error"] = "empty_quote"
            return out
        pct = (last - prev) / prev * 100.0
        out.update(
            ok=True,
            name=str(row.get("stock_name", "") or STOCK_NAMES.get(symbol, "")),
            last=round(last, 4),
            prev_close=round(prev, 4),
            pct=round(pct, 2),
            open=_rt_f(row.get("open_price")),
            high=_rt_f(row.get("high_price")),
            low=_rt_f(row.get("low_price")),
            volume=_rt_f(row.get("volume")),
            amount=_rt_f(row.get("turnover")),
            turnover=_rt_f(row.get("turnover_rate")),
            amplitude=_rt_f(row.get("amplitude")),
            pe=_rt_f(row.get("pe_ratio")),
            pb=_rt_f(row.get("pb_ratio")),
            source="富途OpenD",
        )
    except Exception as e:  # noqa: BLE001
        out["error"] = f"opend_unreachable: {e}"
    finally:
        try:
            if ctx is not None:
                ctx.close()
        except Exception:  # noqa: BLE001
            pass
    return out


def fetch_realtime_via_ths(symbol: str) -> Dict[str, Any]:
    """同花顺 Financial-API 实时行情（A股最高优先级之一，需 HITHINK_FINANCE_API_KEY）。

    仅支持 A股(.SS/.SZ)；港股/美股返回 ok=False 由上层兜底。
    复用 utils_ths.fetch_ths_quote 单标的调用，返回与 fetch_realtime_snapshot
    同构的全字段字典。无 key / 空数据 / 异常均返回 ok=False，绝不抛异常。
    """
    out: Dict[str, Any] = {"ok": False, "symbol": symbol, "source": "同花顺"}
    if not symbol.endswith((".SS", ".SZ")):
        out["error"] = "market_not_supported"
        return out
    try:
        import utils_ths as THS
        # 双路读取 key：os.environ 优先，其次 Streamlit Cloud Secrets（st.secrets）；
        # 保证云端在 Settings→Secrets 配了 HITHINK_FINANCE_API_KEY 后即生效。
        _key = _get_secret("HITHINK_FINANCE_API_KEY")
        if not _key and not THS.ths_available:
            out["error"] = "no_key"
            return out
        res = THS.fetch_ths_quote([symbol], api_key=_key)
        d = res.get(symbol) if isinstance(res, dict) else None
        if not d or d.get("price") is None:
            out["error"] = "empty"
            return out
        last = _rt_f(d.get("price"))
        prev = _rt_f(d.get("prev_close"))
        if last is None or prev is None or last <= 0:
            out["error"] = "empty"
            return out
        pct = _rt_f(d.get("chg_pct"))
        if pct is None and prev:
            pct = (last - prev) / prev * 100.0
        out.update(
            ok=True,
            name=STOCK_NAMES.get(symbol) or symbol,
            last=round(last, 4),
            prev_close=round(prev, 4),
            pct=round(pct, 2) if pct is not None else None,
            open=_rt_f(d.get("open")),
            high=_rt_f(d.get("high")),
            low=_rt_f(d.get("low")),
            volume=_rt_f(d.get("volume")),
            amount=_rt_f(d.get("turnover")),
            chg=_rt_f(d.get("chg")),
            source="同花顺",
        )
    except Exception as e:  # noqa: BLE001
        out["error"] = f"ths_error: {e}"
    return out


def fetch_realtime_quote(symbol: str) -> Dict[str, Any]:
    """返回 {ok, symbol, name, last, prev_close, pct, source}。

    仅处理港股(.HK)与A股(.SS/.SZ)；美股沿用 Yahoo（CI 上可靠），返回 ok=False。
    优先级：富途OpenD → 东方财富 → 腾讯 → 新浪；任一源成功即返回。
    最终 ok=False 表示所有源都无法取得可信实时价。
    """
    out: Dict[str, Any] = {"ok": False, "symbol": symbol}
    _nm = STOCK_NAMES.get(symbol)  # 中文名映射优先，防止新浪英文名/腾讯简称污染 name 字段
    is_hk = symbol.endswith(".HK")
    is_cn = symbol.endswith((".SS", ".SZ"))
    if not (is_hk or is_cn):
        return out

    # 0) 富途 OpenD（若已开启 USE_FUTU 且网关可达）：优先级最高
    try:
        futu_q = fetch_realtime_via_futu(symbol)
        if futu_q.get("ok"):
            return futu_q
    except Exception:  # noqa: BLE001
        pass

    # 0.5) 同花顺 Financial-API（A股；需 HITHINK_FINANCE_API_KEY）：次高优先级
    if is_cn:
        try:
            ths_q = fetch_realtime_via_ths(symbol)
            if ths_q.get("ok"):
                return ths_q
        except Exception:  # noqa: BLE001
            pass

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://finance.sina.com.cn",
    }

    def _pct(last: float, prev: float) -> Optional[float]:
        try:
            if last is None or prev is None or last <= 0 or prev <= 0:
                return None
            p = (last - prev) / prev * 100.0
            if abs(p) > 300:  # 过滤明显错乱
                return None
            return p
        except Exception:  # noqa: BLE001
            return None

    if is_hk:
        code = symbol.replace(".HK", "").zfill(5)
        em_secid = f"116.{code}"
        gt = f"hk{code}"
        div = 1000.0
    else:
        code = symbol.replace(".SS", "").replace(".SZ", "")
        mkt = "1" if symbol.endswith(".SS") else "0"
        em_secid = f"{mkt}.{code}"
        gt = ("sh" if symbol.endswith(".SS") else "sz") + code
        div = 100.0

    # 1) 东方财富 push2（实测 HK 实时准确）
    try:
        url = (
            f"https://push2.eastmoney.com/api/qt/stock/get?secid={em_secid}"
            f"&fields=f43,f44,f45,f46,f47,f57,f58,f60,f169,f170"
        )
        d = requests.get(url, headers=headers, timeout=6).json().get("data") or {}
        f43, f60 = d.get("f43"), d.get("f60")
        if f43 and f60:
            p = _pct(float(f43), float(f60))
            if p is not None:
                out.update(
                    ok=True, name=_nm or d.get("f58", ""), last=round(float(f43) / div, 4),
                    prev_close=round(float(f60) / div, 4), pct=round(p, 2), source="东方财富",
                )
                return out
    except Exception:  # noqa: BLE001
        pass

    # 2) 腾讯 gtimg（HK 与 A股：idx3=最新 idx4=昨收）
    try:
        r = requests.get(f"https://qt.gtimg.cn/q={gt}", headers=headers, timeout=6)
        r.encoding = "gbk"
        payload = r.text.split('="', 1)[1].rstrip('";\n ')
        parts = payload.split("~")
        last, prev = float(parts[3]), float(parts[4])
        p = _pct(last, prev)
        if p is not None:
            out.update(ok=True, name=_nm or (parts[1] if len(parts) > 1 else ""), last=last,
                       prev_close=prev, pct=round(p, 2), source="腾讯")
            return out
    except Exception:  # noqa: BLE001
        pass

    # 3) 新浪
    try:
        r = requests.get(f"https://hq.sinajs.cn/list={gt}", headers=headers, timeout=6)
        r.encoding = "gbk"
        payload = r.text.split('="', 1)[1].rstrip('";\n ')
        parts = payload.split(",")
        if is_hk:
            prev, last = float(parts[3]), float(parts[6])
        else:
            prev, last = float(parts[2]), float(parts[3])
        p = _pct(last, prev)
        if p is not None:
            out.update(ok=True, name=_nm or (parts[1] if len(parts) > 1 else ""), last=last,
                       prev_close=prev, pct=round(p, 2), source="新浪")
            return out
    except Exception:  # noqa: BLE001
        pass

    return out


# ---------------------------------------------------------------------------
# 实时行情：全字段快照 + 当日分时，多源降级、逐只容错
# 数据源优先级（行情）：
#   A股：富途OpenD → 同花顺 Financial-API → 腾讯 gtimg → 东财 push2 → 新浪
#   港股：富途OpenD → 东财 push2 → 腾讯 gtimg → 新浪
#   美股：腾讯 us → yfinance 兜底
# 任何源成功即返回；失败自动降级到下一源（服务端无本地依赖时 富途/同花顺
# 自动跳过，由免费源兜底，保证公网部署稳定可用）。
# 分时：东财 trends2 → 腾讯分钟线 → 新浪 5 分钟 K（A股）
# ---------------------------------------------------------------------------
_RT_UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://gu.qq.com/",
}


def _rt_f(v):
    """安全转 float（容忍千分位/空串/None）。"""
    try:
        x = float(str(v).replace(",", "").strip())
        return x
    except (TypeError, ValueError):
        return None


def _rt_gt(symbol: str) -> str:
    """腾讯 gtimg 代码：sh600519 / sz000001 / hk00700 / usAAPL。"""
    if symbol.endswith(".HK"):
        return "hk" + symbol.replace(".HK", "").zfill(5)
    if symbol.endswith(".SS"):
        return "sh" + symbol.replace(".SS", "")
    if symbol.endswith(".SZ"):
        return "sz" + symbol.replace(".SZ", "")
    return "us" + symbol.split(".")[0].upper()


def _rt_sina_gt(symbol: str) -> str:
    """新浪代码：sh600519 / sz000001 / hk00700。"""
    if symbol.endswith(".HK"):
        return "hk" + symbol.replace(".HK", "").zfill(5)
    if symbol.endswith(".SS"):
        return "sh" + symbol.replace(".SS", "")
    return "sz" + symbol.replace(".SZ", "")


def fetch_realtime_snapshot(symbol: str) -> Dict[str, Any]:
    """
    全字段实时行情快照（监控页用）：
      {ok, symbol, name, last, prev_close, open, high, low, volume, amount,
       pct, chg, turnover, amplitude, pe, source, ts}
    多源降级（任一成功即返回）：
      1) 腾讯 gtimg —— A/港/美 字段结构统一（PE 只有腾讯可靠，东财 f169 实测返回负值不可用）
      2) 东财 push2 —— A股(×100)/港股(×1000)
      3) 新浪 hq —— A股/港股
      4) yfinance —— 美股兜底
    任何失败返回 ok=False，绝不抛异常。
    """
    out: Dict[str, Any] = {"ok": False, "symbol": symbol}
    _nm = STOCK_NAMES.get(symbol)  # 中文名映射优先

    # 0) 富途 OpenD（若已开启 USE_FUTU 且网关可达）：最高优先级（全字段）
    try:
        futu_q = fetch_realtime_via_futu(symbol)
        if futu_q.get("ok"):
            return futu_q
    except Exception:  # noqa: BLE001
        pass

    # 0.5) 同花顺 Financial-API（A股；需 HITHINK_FINANCE_API_KEY）：次高优先级
    if symbol.endswith((".SS", ".SZ")):
        try:
            ths_q = fetch_realtime_via_ths(symbol)
            if ths_q.get("ok"):
                return ths_q
        except Exception:  # noqa: BLE001
            pass

    # 1) 腾讯（主源，免费源中字段最全、含 PE）
    try:
        r = requests.get(f"https://qt.gtimg.cn/q={_rt_gt(symbol)}", headers=_RT_UA, timeout=6)
        r.encoding = "gbk"
        parts = r.text.split('="', 1)[1].rstrip('";\n ').split("~")
        last, prev = _rt_f(parts[3]), _rt_f(parts[4])
        if last and prev and last > 0:
            amount = _rt_f(parts[37])
            if amount is not None and symbol.endswith((".SS", ".SZ")):
                amount = amount * 1e4  # 腾讯 A股成交额单位是万元
            pct = _rt_f(parts[32])
            if pct is None:
                pct = (last - prev) / prev * 100.0
            out.update(
                ok=True, name=_nm or (parts[1] if len(parts) > 1 else ""),
                last=round(last, 4), prev_close=round(prev, 4),
                open=_rt_f(parts[5]), high=_rt_f(parts[33]), low=_rt_f(parts[34]),
                volume=_rt_f(parts[36]), amount=amount,
                pct=round(pct, 2), chg=_rt_f(parts[31]),
                turnover=_rt_f(parts[38]), amplitude=_rt_f(parts[43]),
                pe=_rt_f(parts[39]), source="腾讯", ts=parts[30] if len(parts) > 30 else "",
            )
            return out
    except Exception:  # noqa: BLE001
        pass

    # 2) 东财（A股/港股）
    if symbol.endswith((".SS", ".SZ", ".HK")):
        try:
            code = symbol.split(".")[0]
            if symbol.endswith(".HK"):
                secid = "116." + code.zfill(5)
            else:
                secid = ("1." if symbol.endswith(".SS") else "0.") + code
            div = 1000.0 if symbol.endswith(".HK") else 100.0
            d = requests.get(
                f"https://push2.eastmoney.com/api/qt/stock/get?secid={secid}"
                f"&fields=f43,f44,f45,f46,f47,f48,f57,f58,f60,f86,f168,f170,f171",
                headers=_RT_UA, timeout=6,
            ).json().get("data") or {}
            f43, f60 = d.get("f43"), d.get("f60")
            if f43 and f60:
                last, prev = float(f43) / div, float(f60) / div
                pct = (float(d.get("f170") or 0) / 100.0) if d.get("f170") is not None else ((last - prev) / prev * 100.0)
                out.update(
                    ok=True, name=_nm or d.get("f58", ""),
                    last=round(last, 4), prev_close=round(prev, 4),
                    open=round(float(d.get("f46") or f43) / div, 4),
                    high=round(float(d.get("f44") or f43) / div, 4),
                    low=round(float(d.get("f45") or f43) / div, 4),
                    volume=float(d.get("f47") or 0), amount=float(d.get("f48") or 0),
                    pct=round(pct, 2), chg=round(last - prev, 3),
                    turnover=round(float(d.get("f168") or 0) / 100.0, 2),
                    amplitude=round(float(d.get("f171") or 0) / 100.0, 2),
                    pe=None, source="东方财富", ts=d.get("f86"),
                )
                return out
        except Exception:  # noqa: BLE001
            pass

    # 3) 新浪（A股/港股）
    if symbol.endswith((".SS", ".SZ", ".HK")):
        try:
            r = requests.get(f"https://hq.sinajs.cn/list={_rt_sina_gt(symbol)}", headers=_RT_UA, timeout=6)
            r.encoding = "gbk"
            parts = r.text.split('="', 1)[1].rstrip('";\n ').split(",")
            if len(parts) >= 10:
                if symbol.endswith(".HK"):
                    last, prev = _rt_f(parts[6]), _rt_f(parts[3])
                    pct = _rt_f(parts[8])
                    chg = _rt_f(parts[7])
                    volume, amount = _rt_f(parts[12]), _rt_f(parts[11])
                    name = _nm or (parts[1] if len(parts) > 1 else "")
                    ts = (parts[17] if len(parts) > 17 else "") + " " + (parts[18] if len(parts) > 18 else "")
                else:
                    last, prev = _rt_f(parts[3]), _rt_f(parts[2])
                    pct = (last - prev) / prev * 100.0 if prev else None
                    chg = (last - prev) if last is not None and prev is not None else None
                    volume, amount = _rt_f(parts[8]), _rt_f(parts[9])
                    name = _nm or parts[0]
                    ts = (parts[30] if len(parts) > 30 else "") + " " + (parts[31] if len(parts) > 31 else "")
                if last and prev and last > 0:
                    out.update(
                        ok=True, name=name, last=round(last, 4), prev_close=round(prev, 4),
                        open=_rt_f(parts[1]), high=_rt_f(parts[4]), low=_rt_f(parts[5]),
                        volume=volume, amount=amount,
                        pct=round(pct, 2) if pct is not None else None,
                        chg=round(chg, 3) if chg is not None else None,
                        turnover=None, amplitude=None, pe=None, source="新浪", ts=ts,
                    )
                    return out
        except Exception:  # noqa: BLE001
            pass

    # 4) 美股 yfinance 兜底
    if not symbol.endswith((".HK", ".SS", ".SZ")):
        try:
            t = yf.Ticker(_yf_sym(symbol))
            info = t.info or {}
            last = _rt_f(info.get("regularMarketPrice") or info.get("currentPrice"))
            prev = _rt_f(info.get("regularMarketPreviousClose") or info.get("previousClose"))
            if last and prev:
                pct = (last - prev) / prev * 100.0
                h = t.history(period="2d")
                high = low = None
                if h is not None and not h.empty:
                    high = round(float(h["High"].iloc[-1]), 4)
                    low = round(float(h["Low"].iloc[-1]), 4)
                out.update(
                    ok=True, name=_nm or info.get("shortName") or info.get("longName") or symbol,
                    last=round(last, 4), prev_close=round(prev, 4),
                    open=round(_rt_f(info.get("regularMarketOpen")), 4) if _rt_f(info.get("regularMarketOpen")) else None,
                    high=high, low=low,
                    volume=info.get("volume"), amount=None,
                    pct=round(pct, 2), chg=round(last - prev, 3),
                    turnover=None, amplitude=None,
                    pe=_rt_f(info.get("trailingPE")), source="yfinance", ts="",
                )
                return out
        except Exception:  # noqa: BLE001
            pass

    return out


def fetch_realtime_snapshots(symbols, max_workers: int = 6) -> Dict[str, Dict[str, Any]]:
    """批量实时快照：并行拉取，每只独立容错（失败返回 ok=False，不影响其他）。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    syms = list(dict.fromkeys(symbols))
    results: Dict[str, Dict[str, Any]] = {}
    if not syms:
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_realtime_snapshot, s): s for s in syms}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                results[s] = fut.result()
            except Exception:  # noqa: BLE001
                results[s] = {"ok": False, "symbol": s}
    return results


def fetch_intraday_trend(symbol: str, days: int = 1) -> pd.DataFrame:
    """
    当日分时（分钟级）。返回 DataFrame[time, price, avg, volume, amount]。
    数据源：东财 trends2（A股/港股）→ 腾讯分钟线（A股/港股）→ 新浪 5 分钟 K（A股）。
    美股无免费分时 → 返回空 DataFrame。失败返回空 DataFrame，绝不抛异常。
    """
    # 1) 东财 trends2（A股/港股，字段：时间,开,收,高,低,成交量,成交额,均价）
    if symbol.endswith((".SS", ".SZ", ".HK")):
        try:
            code = symbol.split(".")[0]
            if symbol.endswith(".HK"):
                secid = "116." + code.zfill(5)
            else:
                secid = ("1." if symbol.endswith(".SS") else "0.") + code
            url = (
                f"https://push2his.eastmoney.com/api/qt/stock/trends2/get?secid={secid}"
                f"&fields1=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13"
                f"&fields2=f51,f52,f53,f54,f55,f56,f57,f58&ndays=1&iscr=0"
            )
            d = requests.get(url, headers=_RT_UA, timeout=8).json().get("data") or {}
            trends = d.get("trends") or []
            if trends:
                rows = []
                for line in trends:
                    p = line.split(",")
                    if len(p) >= 8:
                        rows.append({
                            "time": pd.to_datetime(p[0]),
                            "price": float(p[2]),
                            "avg": float(p[7]),
                            "volume": float(p[5]),
                            "amount": float(p[6]),
                        })
                if rows:
                    return pd.DataFrame(rows)
        except Exception:  # noqa: BLE001
            pass

    # 2) 腾讯分钟线（A股/港股）
    if symbol.endswith((".SS", ".SZ", ".HK")):
        try:
            gt = _rt_gt(symbol)
            j = requests.get(
                f"https://web.ifzq.gtimg.cn/appstock/app/minute/query?code={gt}",
                headers=_RT_UA, timeout=8,
            ).json()
            node = (j.get("data") or {}).get(gt) or {}
            rows_raw = (node.get("data") or {}).get("data") or []
            if rows_raw:
                today = datetime.now().strftime("%Y-%m-%d")
                rows = []
                for row in rows_raw:
                    if len(row) >= 2:
                        t, price = row[0], _rt_f(row[1])
                        if price is None:
                            continue
                        hh, mm = (t[:2], t[2:4]) if len(t) >= 4 else (t[:2], "00")
                        rows.append({
                            "time": pd.to_datetime(f"{today} {hh}:{mm}"),
                            "price": price,
                            "avg": None,
                            "volume": _rt_f(row[2]) or 0,
                            "amount": 0,
                        })
                if rows:
                    return pd.DataFrame(rows)
        except Exception:  # noqa: BLE001
            pass

    # 3) 新浪 5 分钟 K（A股）
    if symbol.endswith((".SS", ".SZ")):
        try:
            gt = ("sh" if symbol.endswith(".SS") else "sz") + symbol.split(".")[0]
            j = requests.get(
                f"https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData"
                f"?symbol={gt}&scale=5&ma=no&datalen=96",
                headers=_RT_UA, timeout=8,
            ).json()
            if isinstance(j, list) and j:
                rows = [{
                    "time": pd.to_datetime(x.get("day")),
                    "price": float(x["close"]),
                    "avg": None,
                    "volume": float(x.get("volume") or 0),
                    "amount": 0,
                } for x in j if x.get("close")]
                if rows:
                    return pd.DataFrame(rows)
        except Exception:  # noqa: BLE001
            pass

    return pd.DataFrame()


def fetch_intraday_trends(symbols, max_workers: int = 6) -> Dict[str, pd.DataFrame]:
    """批量分时：并行拉取，每只独立容错（失败返回空 DataFrame）。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    syms = list(dict.fromkeys(symbols))
    results: Dict[str, pd.DataFrame] = {}
    if not syms:
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_intraday_trend, s): s for s in syms}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                results[s] = fut.result()
            except Exception:  # noqa: BLE001
                results[s] = pd.DataFrame()
    return results


# ---------------------------------------------------------------------------
# 扩展数据维度：盘口 / 成交明细 / 公告 / 龙虎榜
# 设计：
#   · 盘口与成交明细以 富途 OpenD 为权威源（本地运行 OpenD 时最准、含 L2 逐笔）；
#     公网无 OpenD 时降级到 东财（字段映射以公开文档为准，并做合理性校验，
#     校验不通过则 ok=False，绝不展示可疑数据）。
#   · 公告以巨潮 cninfo 为权威源（官方）；龙虎榜以 东财 datacenter 为源。
#   · 所有函数单只独立容错，失败返回 ok=False / 空列表，绝不抛异常。
# ---------------------------------------------------------------------------

def _em_secid(symbol: str) -> Optional[str]:
    """yfinance 代码 → 东财 secid（"1.600519" / "0.000001" / "116.00700"）。"""
    if symbol.endswith(".HK"):
        return "116." + symbol.replace(".HK", "").zfill(5)
    if symbol.endswith(".SS"):
        return "1." + symbol.replace(".SS", "")
    if symbol.endswith(".SZ"):
        return "0." + symbol.replace(".SZ", "")
    return None


def fetch_order_book(symbol: str) -> Dict[str, Any]:
    """5 档盘口（买卖各五档）。

    优先级：富途 OpenD(get_order_book) → 东财 push2(ff)。
    返回 {ok, symbol, bids:[(price, vol), ...5], asks:[(price, vol), ...5],
          source, ts}；校验失败返回 ok=False。
    """
    out: Dict[str, Any] = {"ok": False, "symbol": symbol, "bids": [], "asks": [], "source": ""}

    # 0) 富途 OpenD（权威，含 L2）
    if _futu_secret("USE_FUTU", "false").lower() == "true" and symbol.endswith((".HK", ".SS", ".SZ")):
        try:
            from futu import OpenQuoteContext, RET_ERROR
            if symbol.endswith(".HK"):
                code = "HK." + symbol.replace(".HK", "").zfill(5)
            elif symbol.endswith(".SS"):
                code = "SH." + symbol.replace(".SS", "")
            else:
                code = "SZ." + symbol.replace(".SZ", "")
            host = _futu_secret("FUTU_OPEND_HOST", "127.0.0.1")
            try:
                port = int(_futu_secret("FUTU_OPEND_PORT", "11111") or "11111")
            except ValueError:
                port = 11111
            ctx = None
            try:
                ctx = OpenQuoteContext(host=host, port=port)
                ret, data = ctx.get_order_book(code)
                if ret == RET_ERROR or data is None or data.empty:
                    pass
                else:
                    row = data.iloc[0]
                    bids = _futu_lvls(row.get("bid_price"), row.get("bid_volume"))
                    asks = _futu_lvls(row.get("ask_price"), row.get("ask_volume"))
                    if bids and asks and _validate_book(bids, asks):
                        out.update(ok=True, bids=bids, asks=asks, source="富途OpenD",
                                   ts=str(row.get("ts", "")))
                        return out
            finally:
                if ctx is not None:
                    try:
                        ctx.close()
                    except Exception:  # noqa: BLE001
                        pass
        except Exception as e:  # noqa: BLE001
            logger.debug("富途盘口失败 %s: %s", symbol, e)

    # 1) 东财 push2 盘口（云降级，需校验）
    secid = _em_secid(symbol)
    if secid:
        try:
            url = (
                f"https://push2.eastmoney.com/api/qt/stock/ff?secid={secid}"
                f"&fields=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,"
                f"f61,f62,f63,f64,f65,f66,f67,f68,f69,f70&fltt=2&invt=2"
            )
            d = requests.get(url, headers=_RT_UA, timeout=8).json().get("data") or {}
            # 东财 ff：f51-f55=买1~5价, f56-f60=买1~5量, f61-f65=卖1~5价, f66-f70=卖1~5量
            bids = [( _rt_f(d.get(f"f5{i}")), _rt_f(d.get(f"f6{i}")) ) for i in range(1, 6)]
            asks = [( _rt_f(d.get(f"f7{i}")), _rt_f(d.get(f"f8{i}")) ) for i in range(1, 6)]
            bids = [(p, v) for p, v in bids if p and v]
            asks = [(p, v) for p, v in asks if p and v]
            if len(bids) == 5 and len(asks) == 5 and _validate_book(bids, asks):
                out.update(ok=True, bids=bids, asks=asks, source="东方财富", ts=d.get("f31", ""))
                return out
        except Exception as e:  # noqa: BLE001
            logger.debug("东财盘口失败 %s: %s", symbol, e)
    return out


def _futu_lvls(prices, vols) -> List[Tuple[float, float]]:
    """富途盘口价格/量（可能是 list 或标量）→ [(price, vol), ...]。"""
    out = []
    try:
        pl = prices if isinstance(prices, (list, tuple)) else [prices]
        vl = vols if isinstance(vols, (list, tuple)) else [vols]
        for p, v in zip(pl, vl):
            pf, vf = _rt_f(p), _rt_f(v)
            if pf and vf:
                out.append((pf, vf))
    except Exception:  # noqa: BLE001
        pass
    return out


def _validate_book(bids: List, asks: List) -> bool:
    """盘口合理性校验：买价递减、卖价递增、买1<卖1、量>0、价格为正。"""
    try:
        if not bids or not asks:
            return False
        bp = [p for p, _ in bids]
        ap = [p for p, _ in asks]
        if any(p <= 0 for p in bp + ap):
            return False
        if any(v <= 0 for _, v in bids + asks):
            return False
        if bp != sorted(bp, reverse=True):
            return False
        if ap != sorted(ap):
            return False
        if bp[0] >= ap[0]:
            return False
        return True
    except Exception:  # noqa: BLE001
        return False


def fetch_tick_detail(symbol: str, count: int = 20) -> Dict[str, Any]:
    """当日成交明细（逐笔）。

    优先级：富途 OpenD(get_rt_ticker) → 东财逐笔(push2 stock/query)。
    返回 {ok, symbol, ticks:[{time, price, volume, direction}], source}；
    direction ∈ BUY/SELL/NEUTRAL。校验失败返回 ok=False。
    """
    out: Dict[str, Any] = {"ok": False, "symbol": symbol, "ticks": [], "source": ""}

    # 0) 富途 OpenD（权威逐笔）
    if _futu_secret("USE_FUTU", "false").lower() == "true" and symbol.endswith((".HK", ".SS", ".SZ")):
        try:
            from futu import OpenQuoteContext, RET_ERROR
            if symbol.endswith(".HK"):
                code = "HK." + symbol.replace(".HK", "").zfill(5)
            elif symbol.endswith(".SS"):
                code = "SH." + symbol.replace(".SS", "")
            else:
                code = "SZ." + symbol.replace(".SZ", "")
            host = _futu_secret("FUTU_OPEND_HOST", "127.0.0.1")
            try:
                port = int(_futu_secret("FUTU_OPEND_PORT", "11111") or "11111")
            except ValueError:
                port = 11111
            ctx = None
            try:
                ctx = OpenQuoteContext(host=host, port=port)
                ret, data = ctx.get_rt_ticker(code, num=count)
                if ret != RET_ERROR and data is not None and not data.empty:
                    ticks = []
                    for _, r in data.iterrows():
                        p = _rt_f(r.get("price"))
                        v = _rt_f(r.get("volume"))
                        if p is None or v is None:
                            continue
                        ticks.append({
                            "time": str(r.get("time", "")),
                            "price": round(p, 4),
                            "volume": v,
                            "direction": str(r.get("ticker_direction", "NEUTRAL") or "NEUTRAL"),
                        })
                    if ticks:
                        out.update(ok=True, ticks=ticks, source="富途OpenD")
                        return out
            finally:
                if ctx is not None:
                    try:
                        ctx.close()
                    except Exception:  # noqa: BLE001
                        pass
        except Exception as e:  # noqa: BLE001
            logger.debug("富途逐笔失败 %s: %s", symbol, e)

    # 1) 东财逐笔（云降级，需校验）
    secid = _em_secid(symbol)
    if secid:
        try:
            url = (
                f"https://push2.eastmoney.com/api/qt/stock/query?secid={secid}"
                f"&fields=f51,f52,f53,f54,f55&fltt=2&invt=2&pn=1&pz={count}"
            )
            j = requests.get(url, headers=_RT_UA, timeout=8).json()
            data = (j.get("data") or {}).get("details") or []
            if isinstance(data, list) and data:
                ticks = []
                for item in data:
                    parts = item.split(",") if isinstance(item, str) else []
                    if len(parts) < 5:
                        continue
                    t = parts[0]
                    p = _rt_f(parts[1])
                    v = _rt_f(parts[2])
                    dirc = parts[4] if len(parts) > 4 else ""
                    if p is None or v is None:
                        continue
                    ticks.append({
                        "time": t,
                        "price": round(p, 4),
                        "volume": v,
                        "direction": "BUY" if dirc in ("1", "B", "买") else ("SELL" if dirc in ("2", "S", "卖") else "NEUTRAL"),
                    })
                if ticks:
                    out.update(ok=True, ticks=ticks, source="东方财富")
                    return out
        except Exception as e:  # noqa: BLE001
            logger.debug("东财逐笔失败 %s: %s", symbol, e)
    return out


def fetch_cninfo_stock_announcements(symbol: str, top_n: int = 8) -> List[Dict[str, Any]]:
    """巨潮资讯网（官方）个股公告。

    symbol: yfinance 代码（"600519.SS"）。优先按股过滤；若返回为空（cninfo
    按股过滤不稳定）则降级返回全市场最新公告（保证有内容）。
    任何失败返回空列表。
    """
    out: List[Dict[str, Any]] = []
    code = symbol.split(".")[0] if symbol else ""
    if not code:
        return out
    try:
        # 先尝试按股精准过滤
        url = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
        payload = {"pageNum": 1, "pageSize": top_n, "tabName": "latest", "stockCode": code}
        data = _safe_post_json(url, payload)
        anns = (data or {}).get("announcements") or [] if isinstance(data, dict) else []
        for a in anns[:top_n]:
            title = a.get("announcementTitle") or ""
            if not title:
                continue
            sec_name = a.get("secName") or ""
            aid = a.get("announcementId") or ""
            out.append({
                "title": (f"{sec_name}：" if sec_name else "") + title,
                "link": f"https://www.cninfo.com.cn/new/disclosure/detail?announcementId={aid}" if aid else "",
                "source": "巨潮资讯网(官方)",
                "date": _cninfo_ts(a.get("announcementTime")),
                "snippet": "",
            })
        if out:
            return out
    except Exception as e:  # noqa: BLE001
        logger.debug("cninfo 个股公告失败 %s: %s", symbol, e)
    # 降级：全市场公告（标出与该股无关的噪声）
    return fetch_cninfo_market_announcements(top_n=top_n)


def fetch_dragon_tiger(top_n: int = 20, trade_date: str = "") -> List[Dict[str, Any]]:
    """龙虎榜（东方财富 datacenter）。

    返回 [{code, name, close, pct, amount, net_buy, explain, trade_date}, ...]。
    不指定 trade_date 时取最近一个交易日。任何失败返回空列表。
    """
    out: List[Dict[str, Any]] = []
    try:
        if not trade_date:
            trade_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        fdate = trade_date.replace("-", "")
        url = (
            "https://datacenter.eastmoney.com/securities/api/data/v1/get"
            f"?reportName=RPT_DAILYBILLBOARD_DETAILS"
            "&columns=SECURITY_CODE,SECURITY_NAME_ABBR,TRADE_DATE,CLOSE_PRICE,"
            "CHANGE_RATE,AMOUNT,NET_BUY_AMOUNT,EXPLAIN"
            f"&filter=(TRADE_DATE='{trade_date}')"
            f"&pageSize={top_n}&sortColumns=AMOUNT&sortTypes=-1&source=WEB&client=WEB"
        )
        j = requests.get(url, headers=_RT_UA, timeout=10).json()
        rows = ((j.get("data") or {}).get("data") or []) if isinstance(j, dict) else []
        for r in rows:
            if not isinstance(r, dict):
                continue
            out.append({
                "code": r.get("SECURITY_CODE", ""),
                "name": r.get("SECURITY_NAME_ABBR", ""),
                "close": _rt_f(r.get("CLOSE_PRICE")),
                "pct": _rt_f(r.get("CHANGE_RATE")),
                "amount": _rt_f(r.get("AMOUNT")),
                "net_buy": _rt_f(r.get("NET_BUY_AMOUNT")),
                "explain": r.get("EXPLAIN", "") or "",
                "trade_date": r.get("TRADE_DATE") or trade_date,
            })
    except Exception as e:  # noqa: BLE001
        logger.debug("龙虎榜失败: %s", e)
    return out


def validate_dedup_realtime(snapshots: Dict[str, Dict[str, Any]], tol_pct: float = 1.5) -> Dict[str, Any]:
    """多源实时价校验与去重。

    输入 {symbol: {ok, last, source, ...}}（来自 fetch_realtime_snapshots 等多源合并）。
    逻辑：
      · 汇集每个 symbol 所有 ok 源的 last；以中位数（或众数）为共识价。
      · 任一源与共识价偏差 > tol_pct → 标记为 outlier（疑似异常源，不参与共识）。
      · 返回 {consensus:{symbol: last}, outliers:{symbol:[(source,last),...]}}。
    用于「结论区」实时看板过滤脏数据，保证多源一致可靠。
    """
    consensus: Dict[str, Any] = {}
    outliers: Dict[str, Any] = {}
    try:
        import statistics
        for sym, snap in snapshots.items():
            if not isinstance(snap, dict) or not snap.get("ok"):
                continue
            last = snap.get("last")
            if last is None or not isinstance(last, (int, float)):
                continue
            # 多源合并：snap 可能携带 sources 列表
            srcs = snap.get("sources") or [{"source": snap.get("source", "?"), "last": last}]
            prices = [s["last"] for s in srcs if isinstance(s, dict) and isinstance(s.get("last"), (int, float))]
            if not prices:
                prices = [last]
            med = statistics.median(prices)
            out_list = []
            used = []
            for s in srcs:
                pl = s.get("last") if isinstance(s, dict) else None
                if pl is None:
                    continue
                dev = abs(pl - med) / med * 100.0 if med else 0.0
                if dev > tol_pct:
                    out_list.append((s.get("source", "?"), pl))
                else:
                    used.append(pl)
            if used:
                consensus[sym] = round(statistics.median(used), 4)
            if out_list:
                outliers[sym] = out_list
    except Exception as e:  # noqa: BLE001
        logger.debug("实时校验去重失败: %s", e)
    return {"consensus": consensus, "outliers": outliers}



# ---------------------------------------------------------------------------
# 新闻相关性过滤：只保留会影响股价的 宏观 / 政策 / 市场 类新闻
# ---------------------------------------------------------------------------
_FINANCE_KW_EN = [
    "stock", "market", "stocks", "equity", "equities", "fed", "federal reserve",
    "rate", "rates", "interest", "inflation", "cpi", "ppi", "gdp", "earnings",
    "revenue", "profit", "guidance", "dividend", "buyback", "ipo", "merger",
    "acquisition", "economy", "economic", "recession", "yield", "treasury",
    "bond", "bonds", "dollar", "fx", "forex", "crypto", "bitcoin", "oil",
    "crude", "gold", "commodity", "tariff", "trade", "sec", "fomc", "powell",
    "ecb", "jobs", "payroll", "unemployment", "nasdaq", "s&p", "dow",
    "hang seng", "nikkei", "policy", "stimulus", "central bank", "chip",
    "semiconductor", "ai", "tech", "bank", "banking", "credit", "default",
    "debt", "leverage", "volatility", "vix", "china", "earnings",
]
_FINANCE_KW_ZH = [
    "股", "市", "涨", "跌", "央", "美联储", "利率", "通胀", "经济", "衰退", "财报",
    "盈利", "营收", "分红", "回购", "上市", "并购", "关税", "贸易", "政策", "刺激",
    "货币", "债券", "收益率", "美元", "人民币", "黄金", "原油", "商品", "半导体",
    "芯片", "人工智能", "科技", "银行", "信贷", "违约", "债务", "杠杆", "波动",
    "就业", "非农", "港股", "A股", "美股", "大盘", "指数", "创业板", "北向", "资金",
    "主力", "国务院", "央行", "证监会", "期货", "外汇",
]
_OFFTOPIC_KW = [
    "murder", "killed", "shooting", "celebrity", "actor", "actress", "singer",
    "movie", "film", "album", "sport", "football", "soccer", "nba", "nfl",
    "olympic", "world cup", "weather", "storm", "hurricane", "earthquake",
    "tornado", "festival", "wedding", "royal", "prince", "princess", "scandal",
    "arrested", "kidnap", "terror", "tsunami", "wildfire", "recipe", "travel",
]


# 英文关键词用「词边界」正则，避免 nfl 命中 signals、ai 命中 said 等子串碰撞
_FIN_RE_EN = [re.compile(r"\b" + re.escape(k) + r"\b", re.I) for k in _FINANCE_KW_EN]
_OFF_RE_EN = [re.compile(r"\b" + re.escape(k) + r"\b", re.I) for k in _OFFTOPIC_KW]


def _is_finance_relevant(item: Dict[str, Any]) -> bool:
    """判断一条新闻是否与股价/宏观/政策相关。强噪声（犯罪/娱乐/体育/天气）直接剔除。"""
    text = " ".join(
        str(item.get(k, "")) for k in ("title", "snippet", "summary", "description")
    ).lower()
    if not text.strip():
        return False
    # 强噪声：整词匹配才剔除（避免 signals→nfl 这类子串误杀）
    if any(p.search(text) for p in _OFF_RE_EN):
        return False
    # 含金融关键词（英文整词 / 中文子串）则保留
    if any(p.search(text) for p in _FIN_RE_EN) or any(k in text for k in _FINANCE_KW_ZH):
        return True
    # 无金融词也无噪声词：来自金融 RSS 源，模糊保留（避免过度过滤）
    return True


# 美股 / 港股 无 API 的 RSS 源（推荐，完全免费）
US_HK_RSS_SOURCES: List[Tuple[str, str]] = [
    ("CNBC", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
    ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    ("Seeking Alpha", "https://seekingalpha.com/market_currents.xml"),
    ("Investing.com", "https://www.investing.com/rss/news.rss"),
    ("AAStocks 港股", "https://www.aastocks.com/rscs/rss/eng/breakingnews.xml"),
    ("HKET 香港经济日报", "https://www.hket.com/rss"),
]


# ----- 多源聚合 -----
def fetch_all_news_multi_source(
    symbols: List[str],
    serpapi_key: str = "",
    finnhub_key: str = "",
    newsapi_key: str = "",
    out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    多源新闻聚合，按优先级：
      1. SerpApi（如果配置）
      2. Finnhub + NewsAPI（如果配置）
      3. Yahoo RSS（始终可用，免费）
      4. Stocktwits（始终可用，免费）
      5. EastMoney（始终可用，免费）
      6. 同花顺/雪球 快讯（A 股/中文宏观，无 key）
      7. 巨潮资讯 全市场公告（A 股官方公告，无 key，大盘页）
    A 股个股新闻：东方财富「个股快讯」按股精确过滤（已验证可用）；巨潮按股过滤接口不稳定，
    故巨潮仅作为「全市场公告」在大盘/新闻中心提供。
    每个 source 失败不影响其他。
    """
    payload: Dict[str, Any] = {
        "macro": [], "policy": [], "stocks": {},
        "sources_used": [], "errors": [],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    has_any = False

    # 1) SerpApi 宏观 + 政策
    if serpapi_key:
        try:
            payload["macro"] = fetch_macro_news(serpapi_key, top_n=12)
            payload["policy"] = fetch_policy_news(serpapi_key, top_n=8)
            payload["sources_used"].append("SerpApi")
            has_any = True
        except Exception as e:  # noqa: BLE001
            payload["errors"].append(f"SerpApi: {e}")

    # 1.5) 政策新闻免费兜底（Google News RSS，无需 key）—— 政策板块不能为空
    try:
        free_policy = fetch_policy_news_free(top_n=8)
        if free_policy:
            payload["policy"] = _dedup_news(payload.get("policy", []) + free_policy)[:10]
            payload["sources_used"].append("Google News RSS(政策)")
            has_any = True
    except Exception as e:  # noqa: BLE001
        payload["errors"].append(f"政策RSS: {e}")

    # 2) Finnhub / NewsAPI 宏观 (作为补充)
    if newsapi_key:
        try:
            for q in ["Federal Reserve", "inflation CPI", "stock market"]:
                items = fetch_newsapi(q, newsapi_key, days_back=2, page_size=3)
                payload["macro"].extend(items)
            payload["sources_used"].append("NewsAPI")
            has_any = True
        except Exception as e:  # noqa: BLE001
            payload["errors"].append(f"NewsAPI: {e}")

    # 3) Yahoo RSS - 始终免费 (抓 macro 兜底)
    try:
        for q in ["Federal Reserve", "stock market", "CPI inflation", "earnings"]:
            items = fetch_yahoo_rss(query=q)
            payload["macro"].extend(items)
        payload["macro"] = _dedup_news(payload["macro"])[:15]
        payload["sources_used"].append("Yahoo RSS")
        has_any = True
    except Exception as e:  # noqa: BLE001
        payload["errors"].append(f"Yahoo RSS: {e}")

    # 4) EastMoney 全球新闻（中文宏观）
    try:
        em_news = fetch_eastmoney_global_news(top_n=12)
        if em_news:
            payload["macro"].extend(em_news)
            payload["sources_used"].append("东方财富网")
            has_any = True
    except Exception as e:  # noqa: BLE001
            payload["errors"].append(f"东方财富: {e}")

    # 4.5) 无 API 的 RSS 源（美股/港股宏观 + 市场新闻，完全免费）
    try:
        for src_name, src_url in US_HK_RSS_SOURCES:
            items = fetch_rss_feed(src_url, src_name, top_n=4)
            payload["macro"].extend(items)
        payload["sources_used"].append("RSS(CNBC/MarketWatch/AAStocks/HKET...)")
        has_any = True
    except Exception as e:  # noqa: BLE001
        payload["errors"].append(f"RSS: {e}")

    # 4.6) 同花顺快讯 + 雪球热帖（A股/中文宏观，无 key）
    try:
        tjqk = fetch_10jqka_news(top_n=10)
        if tjqk:
            payload["macro"].extend(tjqk)
            payload["sources_used"].append("同花顺快讯")
            has_any = True
    except Exception as e:  # noqa: BLE001
        payload["errors"].append(f"同花顺: {e}")
    try:
        xq = fetch_xueqiu_news(top_n=10)
        if xq:
            payload["macro"].extend(xq)
            payload["sources_used"].append("雪球")
            has_any = True
    except Exception as e:  # noqa: BLE001
        payload["errors"].append(f"雪球: {e}")

    # 4.7) 巨潮资讯全市场公告（A 股官方公告，无 key，稳定可用）
    try:
        cn = fetch_cninfo_market_announcements(top_n=8)
        if cn:
            payload["macro"].extend(cn)
            payload["sources_used"].append("巨潮资讯网(官方公告)")
            has_any = True
    except Exception as e:  # noqa: BLE001
        payload["errors"].append(f"巨潮: {e}")

    # 5) 个股新闻
    # A 股个股补充：同花顺/雪球全局快讯（循环外预取一次，避免重复请求）
    _a_share_extra: List[Dict[str, Any]] = []
    try:
        _a_share_extra.extend(fetch_10jqka_news(top_n=5))
    except Exception:  # noqa: BLE001
        pass
    try:
        _a_share_extra.extend(fetch_xueqiu_news(top_n=5))
    except Exception:  # noqa: BLE001
        pass
    for sym in symbols:
        is_hk = sym.endswith(".HK")
        is_cn = sym.endswith((".SS", ".SZ"))
        per_sym: List[Dict[str, Any]] = []
        # SerpApi
        if serpapi_key:
            try:
                per_sym.extend(fetch_stock_news(sym, serpapi_key, is_hk=is_hk, top_n=5))
            except Exception as e:  # noqa: BLE001
                payload["errors"].append(f"SerpApi {sym}: {e}")
        # Finnhub
        if finnhub_key and not is_hk and not is_cn:
            try:
                per_sym.extend(fetch_finnhub_news(sym, finnhub_key))
            except Exception as e:  # noqa: BLE001
                payload["errors"].append(f"Finnhub {sym}: {e}")
        # Yahoo RSS
        try:
            per_sym.extend(fetch_yahoo_rss(ticker=sym))
        except Exception as e:  # noqa: BLE001
            payload["errors"].append(f"Yahoo {sym}: {e}")
        # Stocktwits
        if not is_hk and not is_cn:
            try:
                per_sym.extend(fetch_stocktwits(sym))
            except Exception as e:  # noqa: BLE001
                pass
        # 东方财富（A 股个股精准快讯，已验证可用）
        if is_cn:
            try:
                per_sym.extend(fetch_eastmoney_stock_news(sym))
            except Exception as e:  # noqa: BLE001
                payload["errors"].append(f"EastMoney {sym}: {e}")
            if _a_share_extra:
                per_sym.extend(_a_share_extra)
        payload["stocks"][sym] = _dedup_news(per_sym)[:8]
        if payload["stocks"][sym]:
            has_any = True

    # 过滤与「股价 / 宏观 / 政策」无关的噪声（犯罪/娱乐/体育/天气等），保证新闻中心相关性
    payload["macro"] = [n for n in payload["macro"] if _is_finance_relevant(n)]
    payload["policy"] = [n for n in payload.get("policy", []) if _is_finance_relevant(n)]
    payload["macro"] = _dedup_news(payload["macro"])[:30]
    payload["sources_used"] = list(set(payload["sources_used"]))
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# 16. 下周走势预测（综合政策/消息/基本面/技术面）
# ---------------------------------------------------------------------------

def build_prediction_prompt(
    symbol: str,
    technical: Dict[str, Any],
    news: List[Dict[str, Any]],
    policy_events: List[Dict[str, Any]],
    fundamentals: Dict[str, Any],
    options_data: Dict[str, Any],
) -> str:
    """
    构造给 DeepSeek 的下周走势预测 prompt。
    综合四维：政策面 + 消息面 + 基本面 + 技术面。
    """
    news_text = "\n".join([f"- {n.get('title','')} ({n.get('source','')})" for n in news[:6]]) or "暂无新闻"
    policy_text = "\n".join([f"- {e.get('date','')} {e.get('event','')} ({e.get('importance','')})" for e in policy_events[:5]]) or "暂无近期重大事件"
    fund = fundamentals or {}
    tech = technical or {}
    opt = options_data or {}
    return f"""你是华尔街资深卖方分析师，专注于 1-2 周短期走势判断。
请基于以下四维信息，给出 {symbol} 下周 (5 个交易日) 的走势预测。

【一、技术面】
- 收盘价: ${tech.get('close', 'N/A')}
- 涨跌幅: {tech.get('change_pct', 'N/A')}%
- RSI(14): {tech.get('rsi', 'N/A')} (>70 超买 / <30 超卖)
- MACD: {tech.get('macd', 'N/A')} (信号线 {tech.get('macd_signal', 'N/A')})
- MA20 / MA50: ${tech.get('ma20', 'N/A')} / ${tech.get('ma50', 'N/A')}
- ATR (波动幅度): {tech.get('atr', 'N/A')}

【二、消息面（最近新闻）】
{news_text}

【三、政策面（未来 1-2 周关键事件）】
{policy_text}

【四、基本面】
- PE: {fund.get('pe_ratio', 'N/A')}
- 近期财报: {fund.get('last_earnings', 'N/A')}
- 行业: {fund.get('sector', 'N/A')}

【五、期权市场】
- Vol PCR: {opt.get('vol_pcr', 'N/A')} (>1 看空 / <1 看多)
- OI PCR: {opt.get('oi_pcr', 'N/A')}
- 隐含波动率 (Call/Put): {opt.get('iv_call', 'N/A')} / {opt.get('iv_put', 'N/A')}

请按以下结构输出（300-500 字，专业克制）：

=== 1. 综合判断 ===
一句话定位下周走势（看多 / 中性偏多 / 中性 / 中性偏空 / 看空）

=== 2. 关键驱动 ===
3-5 个支撑你判断的核心因素，按重要性排序

=== 3. 关键价位 ===
- 上方阻力位
- 下方支撑位
- 预计波动区间

=== 4. 风险因素 ===
2-3 个可能颠覆判断的变量

=== 5. 操作建议 ===
不推荐具体股票点位，给出方向性建议（加仓 / 减仓 / 观望 / 对冲）
"""


def predict_next_week(
    api_key: str = "",
    symbol: str = "",
    technical: Optional[Dict[str, Any]] = None,
    news: Optional[List[Dict[str, Any]]] = None,
    policy_events: Optional[List[Dict[str, Any]]] = None,
    fundamentals: Optional[Dict[str, Any]] = None,
    options_data: Optional[Dict[str, Any]] = None,
    *,
    prefer: str = "deepseek",
) -> Optional[str]:
    """调用 LLM 生成下周走势预测（DeepSeek / OpenRouter）。"""
    if technical is None or fundamentals is None or options_data is None:
        return None
    prompt = build_prediction_prompt(
        symbol,
        technical,
        news or [],
        policy_events or [],
        fundamentals,
        options_data,
    )
    return _call_llm(
        messages=[
            {"role": "system", "content": "你是专业卖方策略师，输出必须是简体中文，分析风格克制专业。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.55,
        max_tokens=1500,
        prefer=prefer,
    )


# ---------------------------------------------------------------------------
# 17. 额外的 4 卡指标聚合（一次拉取，缓存 1 小时）
# ---------------------------------------------------------------------------

def fetch_extra_indicators() -> Dict[str, Any]:
    """一次拉取 2Y/US Debt/Margin Debt/NFCI Leverage 四张卡。"""
    return {
        "2y_scorecard": fetch_2y_scorecard(),
        "us_debt": fetch_us_debt(),
        "margin_debt": fetch_margin_debt(),
        "nfci_leverage": fetch_nfci_leverage(),
        "asof": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# ---------------------------------------------------------------------------
# 18. akshare A股数据采集（中国市场全景 / 指数 / 涨跌家数 / 北向资金 / 个股）
# ---------------------------------------------------------------------------
# 接入 github.com/akfamily/akshare — A股最全的开源数据接口
# 软依赖：未安装时所有 akshare 函数自动 short-circuit 返回空

_AKSHARE_AVAILABLE = False
try:
    import akshare as ak  # type: ignore

    _AKSHARE_AVAILABLE = True
except ImportError:
    ak = None  # type: ignore


def akshare_available() -> bool:
    """检查 akshare 是否已安装（用于 app 端提示用户安装）。"""
    return _AKSHARE_AVAILABLE


def _fetch_a_share_indices_tencent() -> List[Dict[str, Any]]:
    """腾讯 gtimg A股指数行情（非东方财富源，东方财富整体限流时作降级）。

    接口: https://qt.gtimg.cn/q=sh000001,sz399001,... 返回 v_sh000001="1~上证指数~000001~现价~昨收~今开~...~涨跌%~..."
    字段以 ~ 分隔: [1]=名称 [3]=最新价 [32]=涨跌幅%
    """
    codes = [
        ("sh000001", "上证指数"), ("sz399001", "深证成指"), ("sz399006", "创业板指"),
        ("sh000300", "沪深300"), ("sh000905", "中证500"), ("sh000688", "科创50"),
    ]
    try:
        q = ",".join(c for c, _ in codes)
        url = f"https://qt.gtimg.cn/q={q}"
        txt = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"},
            timeout=8,
        ).text
        rows: List[Dict[str, Any]] = []
        for line in txt.replace("\n", ";").split(";"):
            if "v_" not in line or "=" not in line:
                continue
            payload = line.split("=", 1)[1].strip().strip('"')
            f = payload.split("~")
            if len(f) < 33:
                continue
            try:
                last = float(f[3])
                pct = float(f[32])
            except (ValueError, IndexError):
                continue
            rows.append({"名称": f[1], "最新价": round(last, 2), "涨跌幅": round(pct, 2)})
        return rows
    except Exception as e:  # noqa: BLE001
        logger.debug("腾讯 A股指数失败: %s", e)
        return []


def fetch_a_share_overview() -> Dict[str, Any]:
    """
    A股市场全景：
      - 上证综指 / 深证成指 / 创业板 / 沪深 300 / 中证 500 / 科创 50 实时行情
      - 涨/平/跌家数
      - 北向资金净流入
    优先级（多源降级）：东方财富 push2 → 腾讯 gtimg → akshare（东方财富整体限流时腾讯可补）。
    """
    out: Dict[str, Any] = {"indices": [], "advance": None, "decline": None, "north_flow": None, "asof": ""}
    # 1) 东财单只指数（首选）
    try:
        index_map = [
            ("1.000001", "上证指数"), ("0.399001", "深证成指"), ("0.399006", "创业板指"),
            ("1.000300", "沪深300"), ("1.000905", "中证500"), ("1.000688", "科创50"),
        ]
        rows = []
        for secid, name in index_map:
            url = f"https://push2.eastmoney.com/api/qt/stock/get?secid={secid}&fields=f43,f58,f60,f170"
            d = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6).json().get("data") or {}
            last, prev = d.get("f43"), d.get("f60")
            if last and prev:
                rows.append({"名称": name, "最新价": round(float(last) / 100.0, 2),
                             "涨跌幅": round((float(last) - float(prev)) / float(prev) * 100, 2)})
        if rows:
            out["indices"] = rows[:8]
    except Exception as e:  # noqa: BLE001
        logger.debug("东财 A股指数失败: %s", e)
    # 1.5) 腾讯 gtimg（非东财源，东方财富整体限流/断连时降级）
    if not out["indices"]:
        out["indices"] = _fetch_a_share_indices_tencent()
    # 2) akshare 降级（仅当东财+腾讯 都无结果时）
    if not out["indices"]:
        if not _AKSHARE_AVAILABLE:
            out["error"] = "akshare 未安装（pip install akshare）"
            return out
        try:
            df = ak.stock_zh_index_spot_em(symbol="沪深重要指数")
            if df is not None and not df.empty:
                keep = ["最新价", "涨跌幅", "涨跌额", "代码", "名称"]
                df = df[[c for c in keep if c in df.columns]]
                out["indices"] = df.head(8).to_dict(orient="records")
        except Exception as e:  # noqa: BLE001
            logger.warning("akshare 指数失败: %s", e)
            out["error"] = f"东财/腾讯/akshare 全部失败: {e}"
    # 3) 涨跌家数（仅 akshare 有，尽力而为）
    if _AKSHARE_AVAILABLE:
        try:
            ad = ak.stock_market_activity_legu()
            if ad is not None and not ad.empty:
                cols = ad.columns.tolist()
                for kw in ["上涨", "下降", "平盘", "涨停", "跌停"]:
                    for c in cols:
                        if kw in c:
                            out[kw] = int(ad[c].iloc[0])
                out["advance"] = out.get("上涨")
                out["decline"] = out.get("下降")
        except Exception as e:  # noqa: BLE001
            logger.debug("akshare 涨跌家数失败: %s", e)
        try:
            nb = ak.stock_hsgt_north_net_flow_in_em(symbol="北向")
            if nb is not None and not nb.empty:
                out["north_flow"] = float(nb.iloc[-1, 1]) if nb.shape[1] >= 2 else None
        except Exception as e:  # noqa: BLE001
            logger.debug("akshare 北向资金失败: %s", e)
    out["asof"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    return out


def fetch_a_share_heatmap_data(max_n: int = 30) -> List[Dict[str, Any]]:
    """
    A股热力图：实时获取沪深300 + 中证500 成分股的涨跌幅（市值前 max_n）。
    用于 Dashboard 全市场热力图。
    """
    if not _AKSHARE_AVAILABLE:
        return []
    try:
        # 沪深300 实时行情
        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            return []
        # 必含列：代码 / 名称 / 最新价 / 涨跌幅 / 流通市值
        keep = ["代码", "名称", "最新价", "涨跌幅", "流通市值"]
        for c in keep:
            if c not in df.columns:
                return []
        df = df[keep].copy()
        df["流通市值"] = pd.to_numeric(df["流通市值"], errors="coerce")
        df = df.dropna(subset=["流通市值"])
        df = df.nlargest(max_n, "流通市值")
        return df.to_dict(orient="records")
    except Exception as e:  # noqa: BLE001
        logger.warning("akshare A股热力图失败: %s", e)
        return []


def fetch_a_share_kline(symbol: str, days: int = 120) -> pd.DataFrame:
    """
    A股个股 K线。
    symbol 格式：'600519' / '000001' / '300750'（不带后缀）
    优先：东方财富 push2 单只 K线（不受 akshare 限流影响）
    降级：akshare stock_zh_a_hist（后复权）
    """
    # 1) 东方财富单只 K线（首选）
    try:
        mkt = "1" if symbol.startswith(("6", "9", "5")) else "0"
        url = (f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={mkt}.{symbol}"
               f"&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57"
               f"&klt=101&fqt=1&end=20500101&lmt={days}")
        d = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8).json().get("data") or {}
        klines = d.get("klines") or []
        if len(klines) >= 30:
            rows = []
            for kl in klines:
                # 格式: 日期,开,收,高,低,成交量,成交额
                parts = kl.split(",")
                if len(parts) < 6:
                    continue
                rows.append({
                    "Date": pd.to_datetime(parts[0]),
                    "Open": float(parts[1]), "Close": float(parts[2]),
                    "High": float(parts[3]), "Low": float(parts[4]),
                    "Volume": float(parts[5]),
                })
            df = pd.DataFrame(rows).set_index("Date").tail(days)
            return df if not df.empty else pd.DataFrame()
    except Exception as e:  # noqa: BLE001
        logger.debug("东财 A股 %s K线失败: %s", symbol, e)
    # 2) akshare 后复权（降级）
    if not _AKSHARE_AVAILABLE:
        return pd.DataFrame()
    try:
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days + 30)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="hfq")
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            "日期": "Date", "开盘": "Open", "收盘": "Close",
            "最高": "High", "最低": "Low", "成交量": "Volume",
        })
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        return df.tail(days)
    except Exception as e:  # noqa: BLE001
        logger.warning("akshare A股 K线 %s 失败: %s", symbol, e)
        return pd.DataFrame()


def fetch_a_share_quote(symbol: str) -> Dict[str, Any]:
    """
    A股个股实时行情（最新价 / 涨跌幅 / 换手率 / 市盈率）。
    优先：东方财富 push2 单只接口（快、稳、不受 akshare 全市场快照限流影响）
    降级：akshare stock_zh_a_spot_em 全市场快照
    symbol 格式：'600519' / '000001' / '300750'（不带后缀）
    """
    # 1) 东方财富单只接口（首选）
    try:
        mkt = "1" if symbol.startswith(("6", "9", "5")) else "0"
        url = (f"https://push2.eastmoney.com/api/qt/stock/get?secid={mkt}.{symbol}"
               f"&fields=f43,f44,f45,f46,f47,f57,f58,f60,f168,f169,f170")
        d = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6).json().get("data") or {}
        if d.get("f43") and d.get("f60"):
            last = float(d["f43"]) / 100.0
            prev = float(d["f60"]) / 100.0
            chg = (last - prev) / prev * 100.0 if prev else 0.0
            out = {
                "代码": str(d.get("f57", symbol)),
                "名称": d.get("f58", ""),
                "最新价": round(last, 3),
                "涨跌幅": round(chg, 2),
                "涨跌额": round(last - prev, 3),
                "成交量": int(float(d.get("f47") or 0)),
                "换手率": round(float(d.get("f168") or 0) / 100.0, 2),
            }
            try:
                pe = float(d.get("f169") or 0) / 100.0
                out["市盈率-动态"] = round(pe, 2) if pe > 0 else None
            except (TypeError, ValueError):
                out["市盈率-动态"] = None
            return out
    except Exception as e:  # noqa: BLE001
        logger.debug("东财 A股 %s 实时行情失败: %s", symbol, e)
    # 2) akshare 全市场快照（降级）
    if not _AKSHARE_AVAILABLE:
        return {}
    try:
        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            return {}
        row = df[df["代码"] == symbol]
        if row.empty:
            return {}
        keep = ["代码", "名称", "最新价", "涨跌幅", "涨跌额", "成交量", "换手率", "市盈率-动态"]
        out = {}
        for c in keep:
            if c in row.columns:
                out[c] = row[c].iloc[0]
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning("akshare A股 %s 实时行情失败: %s", symbol, e)
        return {}


def fetch_a_share_top_movers(top_n: int = 10) -> Dict[str, Any]:
    """A股涨幅榜 / 跌幅榜 / 涨停板 / 跌停板。"""
    if not _AKSHARE_AVAILABLE:
        return {}
    out: Dict[str, Any] = {}
    try:
        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            return out
        keep = ["代码", "名称", "最新价", "涨跌幅", "换手率"]
        for c in keep:
            if c not in df.columns:
                return out
        df = df[keep].copy()
        df["涨跌幅"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
        df = df.dropna()
        out["top_gainers"] = df.nlargest(top_n, "涨跌幅").to_dict(orient="records")
        out["top_losers"] = df.nsmallest(top_n, "涨跌幅").to_dict(orient="records")
        # 涨停: 涨跌幅 >= 9.5%（科创 20% 单独处理）
        out["limit_up"] = df[df["涨跌幅"] >= 9.5].head(top_n).to_dict(orient="records")
        out["limit_down"] = df[df["涨跌幅"] <= -9.5].head(top_n).to_dict(orient="records")
    except Exception as e:  # noqa: BLE001
        logger.warning("akshare A股涨跌榜失败: %s", e)
    return out



# ---------------------------------------------------------------------------
# v2.4 新增：组合策略多空占优、微表情、宏观风险雷达、ZhuLinsen YAML 拼接到 Morning Brief
# ---------------------------------------------------------------------------

# ---- 微表情 ----

EMOJI_SENTIMENT = {  # 情绪分数 (0-100，越高越贪婪)
    "extreme_greed": "🤑",
    "greed": "😀",
    "neutral": "😐",
    "fear": "😟",
    "extreme_fear": "😱",
}


def emoji_for_sentiment(score):
    """情绪分数 (0-100) → 微表情。"""
    try:
        s = float(score)
    except Exception:
        return EMOJI_SENTIMENT["neutral"]
    if s >= 80:
        return EMOJI_SENTIMENT["extreme_greed"]
    if s >= 60:
        return EMOJI_SENTIMENT["greed"]
    if s >= 40:
        return EMOJI_SENTIMENT["neutral"]
    if s >= 20:
        return EMOJI_SENTIMENT["fear"]
    return EMOJI_SENTIMENT["extreme_fear"]


def emoji_for_panic(vix_value):
    """恐慌指数 VIX/VXN → 微表情（值越大越恐慌）。"""
    try:
        v = float(vix_value)
    except Exception:
        return EMOJI_SENTIMENT["neutral"]
    if v >= 30:
        return EMOJI_SENTIMENT["extreme_fear"]
    if v >= 22:
        return EMOJI_SENTIMENT["fear"]
    if v >= 14:
        return EMOJI_SENTIMENT["neutral"]
    return EMOJI_SENTIMENT["greed"]


def emoji_for_market_regime(chg_pct):
    """一段时间涨跌幅 (%) → 🐂/➡️/🐻。"""
    try:
        c = float(chg_pct)
    except Exception:
        return "➡️"
    if c >= 5.0:
        return "🐂"
    if c <= -5.0:
        return "🐻"
    return "➡️"


def emoji_for_dominance(label):
    return {"多头占优": "📈", "空头占优": "📉", "空仓": "💤", "震荡": "⚖️"}.get(label, "❓")


# ---- 组合策略多空占优 ----

PORTFOLIO_DOMINANCE_THRESH = 0.3  # 平均涨跌绝对值小于此阈值视为震荡/空仓


def compute_portfolio_dominance(stocks_df):
    """
    根据自选股今日涨跌幅判断组合多空占优状态。

    返回:
      - long_count / short_count / flat_count / total
      - long_pct / short_pct / flat_pct
      - avg_chg (所有持股今日平均涨跌 %)
      - positive_ratio / negative_ratio
      - etf_count (代码以 ETF/3306/510/511/159 等常见 ETF 前缀计数)
      - dominance_label: "多头占优" / "空头占优" / "空仓" / "震荡"
      - emoji: 📈 / 📉 / 💤 / ⚖️
    """
    out = {
        "long_count": 0,
        "short_count": 0,
        "flat_count": 0,
        "total": 0,
        "long_pct": 0.0,
        "short_pct": 0.0,
        "flat_pct": 0.0,
        "avg_chg": 0.0,
        "positive_ratio": 0.0,
        "negative_ratio": 0.0,
        "etf_count": 0,
        "dominance_label": "空仓",
        "emoji": "💤",
        "error": None,
    }
    if stocks_df is None or stocks_df.empty:
        out["error"] = "stocks_df 为空"
        return out
    if "涨跌幅" not in stocks_df.columns:
        out["error"] = "缺少 '涨跌幅' 列"
        return out
    try:
        chg = pd.to_numeric(stocks_df["涨跌幅"], errors="coerce").fillna(0.0)
        total = len(chg)
        if total == 0:
            out["error"] = "无有效持仓"
            return out
        long_count = int((chg > 0.0).sum())
        short_count = int((chg < 0.0).sum())
        flat_count = int((chg == 0.0).sum())
        avg_chg = float(chg.mean())

        # ETF 计数（启发：常见 ETF 代码前缀）
        etf_prefixes = ("3306", "510", "511", "512", "513", "515", "159", "561", "588")
        etf_count = 0
        if "symbol" in stocks_df.columns:
            for s in stocks_df["symbol"].astype(str).tolist():
                base = s.split(".")[0]
                if any(base.startswith(pfx) for pfx in etf_prefixes) or "ETF" in s.upper():
                    etf_count += 1

        positive_ratio = long_count / total
        negative_ratio = short_count / total
        if avg_chg > PORTFOLIO_DOMINANCE_THRESH and positive_ratio >= 0.5:
            label = "多头占优"
        elif avg_chg < -PORTFOLIO_DOMINANCE_THRESH and negative_ratio >= 0.5:
            label = "空头占优"
        elif total > 0 and positive_ratio == 0 and negative_ratio == 0:
            label = "空仓"
        else:
            label = "震荡"

        out.update({
            "long_count": long_count,
            "short_count": short_count,
            "flat_count": flat_count,
            "total": total,
            "long_pct": round(positive_ratio * 100, 1),
            "short_pct": round(negative_ratio * 100, 1),
            "flat_pct": round(flat_count / total * 100, 1),
            "avg_chg": round(avg_chg, 2),
            "positive_ratio": round(positive_ratio, 3),
            "negative_ratio": round(negative_ratio, 3),
            "etf_count": etf_count,
            "dominance_label": label,
            "emoji": emoji_for_dominance(label),
        })
        return out
    except Exception as e:
        out["error"] = str(e)
        return out


# ---- 单只指数实时 quote ----

def fetch_index_quote(symbol):
    """单只指数实时 quote: {symbol, last, prev_close, chg_pct}。失败返回 {error}。"""
    try:
        t = yf.Ticker(symbol)
        info = t.fast_info
        last = float(info.last_price) if info.last_price else 0.0
        prev = float(info.previous_close) if info.previous_close else 0.0
        chg = (last - prev) / prev * 100.0 if prev else 0.0
        return {
            "symbol": symbol,
            "last": last,
            "prev_close": prev,
            "chg_pct": chg,
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


# ---- 宏观风险雷达 ----

def _safe_yf_close(symbol, period="5d"):
    """快速取某标的最新收盘价，失败返回 None。"""
    try:
        t = yf.Ticker(symbol)
        info = t.fast_info
        v = info.last_price or info.previous_close
        return float(v) if v else None
    except Exception:
        return None


def compute_macro_risk_radar():
    """
    宏观风险雷达：6 组信号 (regime / rates / risk / ratios / cross_asset / a_share)。
    每组: {metrics: {key: value}, signal: "green"|"yellow"|"red", note: str, emoji}
    overall: 综合灯号 + emoji。
    """
    out = {
        "regime": {},
        "rates": {},
        "risk": {},
        "ratios": {},
        "cross_asset": {},
        "a_share": {},
        "overall": {"signal": "yellow", "note": "数据不足", "emoji": "😐"},
    }
    scores = []

    # --- Risk: VIX / VXN / DXY ---
    vix = _safe_yf_close("^VIX") or 0.0
    vxn = _safe_yf_close("^VXN") or 0.0
    dxy = _safe_yf_close("DX-Y.NYB") or _safe_yf_close("^DXY") or 0.0
    if vix >= 30 or vxn >= 35:
        risk_signal, risk_note, risk_score = "red", "VIX/VXN 突破警戒，市场恐慌", -2
    elif vix >= 22 or vxn >= 25:
        risk_signal, risk_note, risk_score = "yellow", "波动率偏高，情绪偏谨慎", -1
    else:
        risk_signal, risk_note, risk_score = "green", "波动率正常，风险偏好稳定", 1
    out["risk"] = {
        "metrics": {"VIX": round(vix, 2), "VXN": round(vxn, 2), "DXY": round(dxy, 2)},
        "signal": risk_signal,
        "note": risk_note,
        "emoji": emoji_for_panic(max(vix, vxn)),
    }
    scores.append(risk_score)

    # --- Rates: 10Y / 2Y_proxy / spread ---
    y10 = _safe_yf_close("^TNX") or 0.0
    y2 = _safe_yf_close("^IRX") or _safe_yf_close("^FVX") or 0.0  # 13W T-Bill 兜底
    spread = (y10 - y2) if (y10 and y2) else 0.0
    if spread < 0:
        rates_signal, rates_note, rates_score = "red", "2s10s 倒挂，衰退预警", -2
    elif spread < 50:
        rates_signal, rates_note, rates_score = "yellow", "利差扁平，周期顶部信号", -1
    else:
        rates_signal, rates_note, rates_score = "green", "利差健康，曲线正常", 1
    out["rates"] = {
        "metrics": {"10Y": round(y10, 2), "2Y_proxy": round(y2, 2), "spread_bps": round(spread, 0)},
        "signal": rates_signal,
        "note": rates_note,
        "emoji": "🐻" if rates_signal == "red" else ("➡️" if rates_signal == "yellow" else "🐂"),
    }
    scores.append(rates_score)

    # --- Regime (用 rates + risk 粗略合成) ---
    regime_score = (rates_score + risk_score) / 2
    if regime_score <= -1.5:
        regime_signal, regime_note = "red", "衰退/紧缩信号共振"
        regime_label = "衰退"
    elif regime_score >= 1.0:
        regime_signal, regime_note = "green", "扩张期，利率+波动双低"
        regime_label = "扩张"
    else:
        regime_signal, regime_note = "yellow", "周期过渡期"
        regime_label = "滞胀/过渡"
    out["regime"] = {
        "metrics": {"label": regime_label, "score": round(regime_score, 2)},
        "signal": regime_signal,
        "note": regime_note,
        "emoji": "🐂" if regime_label == "扩张" else ("🐻" if regime_label == "衰退" else "➡️"),
    }
    scores.append(regime_score)

    # --- Ratios: 黄金/白银、铜/金 ---
    gold = _safe_yf_close("GC=F") or _safe_yf_close("^XAU") or 0.0
    silver = _safe_yf_close("SI=F") or 0.0
    copper = _safe_yf_close("HG=F") or 0.0
    au_ag = (gold / silver) if silver else 0.0
    cu_au = (copper / gold) if gold else 0.0
    out["ratios"] = {
        "metrics": {
            "Gold": round(gold, 1),
            "Silver": round(silver, 2),
            "Copper": round(copper, 2),
            "Au/Ag": round(au_ag, 1),
            "Cu/Au": round(cu_au, 4),
        },
        "signal": "yellow",
        "note": "Au/Ag " + str(round(au_ag, 1)) + " · Cu/Au " + str(round(cu_au, 4)),
        "emoji": "➡️",
    }
    scores.append(0)

    # --- Cross Asset: BTC / Oil ---
    btc = _safe_yf_close("BTC-USD") or 0.0
    oil = _safe_yf_close("CL=F") or 0.0
    out["cross_asset"] = {
        "metrics": {"BTC": round(btc, 0), "Oil_WTI": round(oil, 2)},
        "signal": "yellow",
        "note": "BTC/原油 = 风险偏好代理",
        "emoji": "➡️",
    }
    scores.append(0)

    # --- A-Share: akshare 拉北向 ---
    a_signal, a_note, a_emoji, a_metrics = "yellow", "akshare 未启用或拉取失败", "➡️", {}
    if _AKSHARE_AVAILABLE:
        try:
            overview = fetch_a_share_overview()
            north = overview.get("north_flow")
            if north is not None:
                a_metrics["北向_亿"] = round(float(north), 1)
                a_signal = "green" if north > 0 else "red"
                a_note = "北向净流入" if north > 0 else "北向净流出"
                a_emoji = "📈" if north > 0 else "📉"
        except Exception:
            pass
    out["a_share"] = {
        "metrics": a_metrics,
        "signal": a_signal,
        "note": a_note,
        "emoji": a_emoji,
    }
    scores.append(0)

    # --- Overall ---
    avg = sum(scores) / max(1, len(scores))
    if avg >= 0.5:
        overall_signal, overall_note, overall_emoji = "green", "宏观环境偏宽松/扩张", "🟢"
    elif avg <= -0.5:
        overall_signal, overall_note, overall_emoji = "red", "宏观环境偏紧缩/衰退", "🔴"
    else:
        overall_signal, overall_note, overall_emoji = "yellow", "宏观信号混合，谨慎观望", "🟡"
    out["overall"] = {"signal": overall_signal, "note": overall_note, "emoji": overall_emoji, "score": round(avg, 2)}
    return out


def macro_risk_narrative(radar: Dict[str, Any]) -> str:
    """
    结合宏观风险雷达，生成一段中文文字总结，说明各维度数据反映的问题与趋势。
    纯基于雷达已有的 metrics / signal / note，不引入外部假设。
    """
    if not radar:
        return "宏观数据暂不可用，无法生成总结。"
    parts = []
    overall = radar.get("overall", {})
    sig = overall.get("signal", "yellow")
    label = {"green": "偏宽松/扩张", "yellow": "信号混合、谨慎观望", "red": "偏紧缩/衰退"}.get(sig, "—")
    parts.append(
        f"【总体】当前宏观综合灯号为「{ {'green':'🟢 正常','yellow':'🟡 关注','red':'🔴 警戒'}.get(sig,'—') }」——{overall.get('note','')}（{label}）。"
    )

    # 逐维度解读
    def _dim(key, name, good, bad):
        g = radar.get(key, {})
        if not g:
            return
        s = g.get("signal", "yellow")
        note = g.get("note", "")
        m = g.get("metrics", {}) or {}
        mtxt = "，".join(f"{k} {v}" for k, v in m.items() if v not in (None, "")) if m else ""
        verdict = good if s == "green" else (bad if s == "red" else "需留意边际变化")
        line = f"【{name}】{note}（{verdict}）"
        if mtxt:
            line += f" 当前读数：{mtxt}。"
        parts.append(line)

    _dim("regime", "周期", "周期处于扩张期，股债风险偏好健康", "周期出现衰退/紧缩共振，应压低仓位与久期")
    _dim("risk", "恐慌/波动", "波动率处于低位，风险偏好稳定，可适度积极", "VIX/VXN 突破警戒，市场进入恐慌定价，需防守")
    _dim("rates", "利率曲线", "利率曲线正常、利差健康，估值环境友好", "2s10s 倒挂或利差扁平，衰退/周期顶部信号显现")
    _dim("ratios", "金银/铜金比", "比值处于常态区间", "金银比/铜金比异常，反映避险或工业需求背离")
    _dim("cross_asset", "跨资产", "BTC/原油风险偏好代理处于常态", "跨资产联动发出风险偏好转折信号")
    _dim("a_share", "A股", "A股资金面平稳/北向净流入，结构偏多", "A股资金面承压/北向净流出，需降低风险暴露")

    # 勾稽建议
    if sig == "red":
        parts.append("【勾稽结论】宏观底色偏紧，个股估值与风险偏好承压；应优先控制总仓位，等待雷达转绿再放大暴露。")
    elif sig == "green":
        parts.append("【勾稽结论】宏观环境友好，可顺势放大权益暴露，但需结合个股维科夫结构与 R 倍数纪律分批介入。")
    else:
        parts.append("【勾稽结论】宏观信号分化，建议中性仓位、结构择优；以个股自身吸筹结构与止盈纪律对冲宏观不确定性。")
    return "\n".join(parts)


# ---- ZhuLinsen YAML 加载 + 拼接到 Morning Brief ----

def ensure_zhu_linsen_repo(repo_dir="vendor/daily_stock_analysis"):
    """
    若本地无 ZhuLinsen/daily_stock_analysis 仓库，自动从 GitHub 下载 zip 解压。
    返回最终 strategies/ 目录的父目录路径。
    """
    import urllib.request
    import zipfile
    import shutil

    p = Path(repo_dir)
    strategies = p / "strategies"
    if strategies.exists() and any(strategies.glob("*.yaml")):
        return str(p)
    p.mkdir(parents=True, exist_ok=True)
    zip_path = p / "src.zip"
    url = "https://github.com/ZhuLinsen/daily_stock_analysis/archive/refs/heads/main.zip"
    try:
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(p)
        extracted = p / "daily_stock_analysis-main"
        if extracted.exists() and (extracted / "strategies").exists():
            target = p / "strategies"
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(extracted / "strategies"), str(target))
        return str(p)
    except Exception as e:
        logger.warning("下载 ZhuLinsen 仓库失败: %s", e)
        return str(p)


def load_zhu_linsen_strategies(
    repo_dir="vendor/daily_stock_analysis",
    only_categories=None,
    max_total_chars=3000,
):
    """
    读取 ZhuLinsen 仓库 strategies/*.yaml，提取 instructions 字段。
    - only_categories: 过滤分类 (trend / pattern / reversal / framework / ...)
    - max_total_chars: 所有 instructions 累计字符上限，避免 prompt 爆炸
    返回 list[dict]: {name, display_name, category, instructions, file}
    """
    try:
        import yaml as _yaml  # PyYAML
    except ImportError:
        logger.warning("PyYAML 未安装，跳过 ZhuLinsen YAML 加载")
        return []
    base = ensure_zhu_linsen_repo(repo_dir)
    strat_dir = Path(base) / "strategies"
    if not strat_dir.exists():
        return []
    out = []
    for yf_path in sorted(strat_dir.glob("*.yaml")):
        try:
            with open(yf_path, encoding="utf-8") as f:
                d = _yaml.safe_load(f) or {}
            if not isinstance(d, dict):
                continue
            cat = d.get("category", "framework")
            if only_categories and cat not in only_categories:
                continue
            out.append({
                "name": d.get("name", yf_path.stem),
                "display_name": d.get("display_name", yf_path.stem),
                "category": cat,
                "instructions": (d.get("instructions") or "").strip(),
                "file": str(yf_path.relative_to(base)),
            })
        except Exception as e:
            logger.warning("加载 %s 失败: %s", yf_path, e)
    truncated, total = [], 0
    for s in out:
        n = len(s["instructions"])
        if total + n > max_total_chars:
            remain = max(0, max_total_chars - total)
            if remain > 0:
                s = dict(s)
                s["instructions"] = s["instructions"][:remain] + "\n…(截断)"
                truncated.append(s)
            break
        total += n
        truncated.append(s)
    return truncated


def splice_strategies_into_prompt(base_prompt, strategies, header="已挂载的策略框架"):
    """把 strategies 列表的 instructions 拼到 prompt 末尾。"""
    if not strategies:
        return base_prompt
    block = "\n\n【" + header + "（盘前/盘中请参照下列框架评估）】\n"
    for s in strategies:
        title = s.get("display_name") or s.get("name") or "?"
        cat = s.get("category", "")
        inst = (s.get("instructions") or "").strip()
        if not inst:
            continue
        block += "\n--- " + str(title) + " (" + str(cat) + ") ---\n" + inst + "\n"
    return base_prompt + block


def build_morning_brief_prompt(context, *,
                                include_zhu_linsen=True,
                                only_categories=None,
                                max_chars=3000):
    """
    构造最终发给 LLM 的 Morning Brief 用户 prompt。
    = MORNING_BRIEF_PROMPT.format(**context) + ZhuLinsen YAML instructions
    include_zhu_linsen=False 时只输出基础 prompt。
    """
    base = MORNING_BRIEF_PROMPT.format(**context)
    if include_zhu_linsen:
        strategies = load_zhu_linsen_strategies(
            only_categories=only_categories,
            max_total_chars=max_chars,
        )
        base = splice_strategies_into_prompt(base, strategies)
    return base


# ---- 跨资产：VXN vs VIX 历史对比 ----

def fetch_panic_history(period="10y"):
    """拉 VIX / VXN 历史 close，用于跨资产对比。"""
    out = {}
    for name, sym in (("VIX", "^VIX"), ("VXN", "^VXN")):
        try:
            df = yf.download(sym, period=period, progress=False, auto_adjust=True)
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close = df["Close"].dropna()
            if not close.empty:
                close.name = name
                out[name] = close
        except Exception as e:
            logger.warning("拉取 %s 历史失败: %s", sym, e)
    return out


def fetch_macro_history(period="10y", tickers=None):
    """通用宏观历史拉取：tickers = {name: yf_symbol}。"""
    if tickers is None:
        tickers = {"10Y美债收益率": "^TNX"}
    out = {}
    for name, sym in tickers.items():
        try:
            df = yf.download(sym, period=period, progress=False, auto_adjust=True)
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close = df["Close"].dropna()
            if not close.empty:
                close.name = name
                out[name] = close
        except Exception as e:
            logger.warning("拉取 %s 历史失败: %s", sym, e)
    return out


# ---------------------------------------------------------------------------
# v2.5 新增：统一公司名 + 智能荐股引擎（技术面/估值/动量/杠杆止损）
# ---------------------------------------------------------------------------

STOCK_NAMES = {
    "MU": "美光", "AAOI": "应用光电", "GOOGL": "谷歌", "MSFT": "微软", "AMZN": "亚马逊",
    "MRVL": "迈威尔", "LITE": "Lumentum", "SNDK": "闪迪", "NVDA": "英伟达", "ORCL": "甲骨文",
    "SPCX": "SpaceX", "SKHY": "Sky Harbour Group", "TSLA": "特斯拉",
    "PLTR": "Palantir", "02500.HK": "曦智科技",
    "0700.HK": "腾讯控股", "0883.HK": "中国海洋石油", "3750.HK": "宁德时代",
    "07709.HK": "南方两倍做多海力士", "7709.HK": "南方两倍做多海力士", "00981.HK": "中芯国际",
    "688809.SS": "强一股份", "300408.SZ": "三环集团", "300679.SZ": "电连技术",
    "000426.SZ": "兴业银锡", "002624.SZ": "完美世界", "601872.SS": "招商轮船",
    "601975.SS": "招商南油", "002258.SZ": "利尔化学", "001331.SZ": "胜通能源",
    "600150.SS": "中国船舶",
    "00293.HK": "国泰航空", "03690.HK": "美团-W", "01138.HK": "中远海能", "03968.HK": "招商银行",
    "EUV": "Corgi Lithography", "RKLB": "Rocket Lab", "GEV": "GE Vernova", "FUTU": "富途",
    "UNH": "联合健康", "NVO": "诺和诺德", "NFLX": "Netflix", "JNJ": "强生", "INTU": "Intuit",
}
# yfinance 港股代码归一化：Yahoo 将 HKEX 5 位代码去掉 1 个前导零转成 4 位
# （00981.HK→0981.HK、00293.HK→0293.HK、03690.HK→3690.HK、07709.HK→7709.HK…）；
# 已是 4 位的代码（0700.HK / 0883.HK / 3750.HK）保持原样。
YF_NORMALIZE = {
    "03690.HK": "3690.HK", "07709.HK": "7709.HK", "00981.HK": "0981.HK",
    "00293.HK": "0293.HK", "01138.HK": "1138.HK", "03968.HK": "3968.HK",
}


def get_stock_name(symbol: str) -> str:
    """优先返回中文名映射；否则尝试 yfinance shortName；兜底返回原代码。"""
    if symbol in STOCK_NAMES:
        return STOCK_NAMES[symbol]
    try:
        t = yf.Ticker(symbol)
        info = t.info or {}
        return info.get("shortName") or info.get("longName") or symbol
    except Exception:
        return symbol


def _yf_sym(symbol: str) -> str:
    """yfinance 代码归一化：显式映射优先，通用规则兜底（5 位港股去 1 个前导零）。"""
    if symbol in YF_NORMALIZE:
        return YF_NORMALIZE[symbol]
    m = re.fullmatch(r"0(\d{4})\.HK", symbol)
    if m:
        return m.group(1) + ".HK"
    return symbol


def fetch_all_metrics(symbols) -> List[Dict[str, Any]]:
    """批量抓每股指标：last/chgPct/pe/ma20/ma60/atrPct/ret20。失败项给中性占位。"""
    out: List[Dict[str, Any]] = []
    syms = list(symbols)
    # 1) 批量历史（一次请求）算 MA / ATR / 动量
    hist_map: Dict[str, Any] = {}
    try:
        data = yf.download(" ".join(_yf_sym(s) for s in syms), period="3mo",
                            interval="1d", group_by="ticker", auto_adjust=True,
                            progress=False, threads=False)
        for s in syms:
            try:
                # 批量下载的列名是归一化后的代码（07709.HK→7709.HK），必须用 _yf_sym 回查
                _k = _yf_sym(s)
                sub = data[_k] if _k in data.columns.get_level_values(0) else data.xs(_k, level=0, axis=1)
                closes = sub["Close"].dropna().astype(float).tolist()
                hist_map[s] = closes
            except Exception:
                hist_map[s] = []
    except Exception:
        hist_map = {s: [] for s in syms}
    # 2) PE + Sector（best-effort，逐只 .info，一次调用同时取两个字段）
    pe_map: Dict[str, Optional[float]] = {}
    sector_map: Dict[str, str] = {}
    for s in syms:
        pe_map[s] = None
        sector_map[s] = ""
        try:
            info = yf.Ticker(_yf_sym(s)).info or {}
            pe = info.get("trailingPE")
            pe_map[s] = float(pe) if isinstance(pe, (int, float)) else None
            sector_map[s] = str(info.get("sector") or info.get("industry") or "")
        except Exception:
            pass
    # 3) 组装
    for s in syms:
        closes = hist_map.get(s, [])
        last = None; chgPct = 0.0; ma20 = ma60 = atrPct = ret20 = None
        if len(closes) >= 2:
            last = closes[-1]
            prev = closes[-2]
            chgPct = (last - prev) / prev * 100.0 if prev else 0.0
            ma20 = float(pd.Series(closes[-20:]).mean()) if len(closes) >= 2 else None
            ma60 = float(pd.Series(closes[-60:]).mean()) if len(closes) >= 60 else None
            rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
            atrPct = float(pd.Series(rets[-20:]).abs().mean() * 100) if len(rets) >= 2 else None
            ret20 = (last / ma20 - 1) * 100 if ma20 else None
        out.append({
            "symbol": s, "name": get_stock_name(s), "last": last, "chgPct": round(chgPct, 2),
            "pe": pe_map.get(s), "sector": sector_map.get(s, ""),
            "ma20": ma20, "ma60": ma60,
            "atrPct": round(atrPct, 2) if atrPct is not None else None,
            "ret20": round(ret20, 2) if ret20 is not None else None,
        })
    # 4) 港股/A股 实时价校正：yfinance 收盘可能滞后或批量下载失配，用多源实时报价覆盖
    for m in out:
        s = m["symbol"]
        if s.endswith((".HK", ".SS", ".SZ")):
            try:
                _q = fetch_realtime_quote(s)
                if _q.get("ok") and _q.get("last"):
                    m["last"] = float(_q["last"])
                    m["chgPct"] = round(float(_q.get("pct") or m.get("chgPct") or 0.0), 2)
            except Exception:  # noqa: BLE001
                pass
    return out


def recommend_stocks(metrics) -> Dict[str, Any]:
    """
    统一荐股评分（v3.0 升级）：
      - 相对强度 RS（个股 20 日收益 vs 板块/大盘均值）维度
      - PE 行业归一化（按 self.pe_group 分组内分位打分）
      - 建议买入价位 = MA20 与 POC 区间（保守/积极两档）
    返回 intraday_t / midterm_hold / buy / details / strategy_meta。
    """
    # 行业 PE 分组（用于归一化；未分组标的用全样本中位数作基准）
    PE_GROUPS: Dict[str, Tuple[float, float]] = {
        # 行业: (合理PE下限, 合理PE上限)
        "半导体": (15, 35), "科技": (18, 40), "消费": (15, 30),
        "能源": (8, 18), "金融": (6, 15), "医药": (15, 35),
        "航运": (5, 12), "军工": (20, 45), "有色": (10, 25),
        "工业": (12, 25), "互联网": (20, 45),
    }

    rows = []
    # 全样本中位数 PE 作为归一化基准
    pes = [e.get("pe") for e in metrics if isinstance(e.get("pe"), (int, float)) and e["pe"] > 0]
    med_pe = float(pd.Series(pes).median()) if pes else 20.0

    for e in metrics:
        last = e.get("last")
        chg = e.get("chgPct") or 0
        pe = e.get("pe")
        ma20 = e.get("ma20"); ma60 = e.get("ma60"); atr = e.get("atrPct"); ret20 = e.get("ret20")
        sector = e.get("sector", "")
        tech = 50.0
        if ma20 is not None:
            tech = 50 + (ret20 or 0) * 1.5 + (10 if (last and last > ma20) else 0) \
                   + (10 if (ma60 is not None and ma20 > ma60) else 0) - (8 if (last and last < ma20) else 0)
            tech = min(100.0, max(0.0, tech))
        # PE 行业归一化：按行业合理区间线性映射，未分组用中位数基准
        val = 50.0
        if pe is not None:
            lo, hi = PE_GROUPS.get(sector, (med_pe * 0.5, med_pe * 1.8))
            if pe <= 0:
                val = 45.0
            elif pe >= hi:
                val = 30.0
            elif pe <= lo:
                val = 88.0
            else:
                val = 88.0 - (pe - lo) / max(hi - lo, 0.01) * 55.0
            val = min(100.0, max(0.0, val))
        mom = min(100.0, max(0.0, 50 + chg * 3))
        # 相对强度 RS：个股 ret20 相对全样本中位数的超额（无 ret20 则中性）
        ret20s = [x.get("ret20") for x in metrics if isinstance(x.get("ret20"), (int, float))]
        med_ret20 = float(pd.Series(ret20s).median()) if ret20s else 0.0
        rs_val = 50.0
        if isinstance(ret20, (int, float)):
            rs_val = min(100.0, max(0.0, 50 + (ret20 - med_ret20) * 3))
        comp = min(100.0, max(0.0, 0.35 * tech + 0.20 * val + 0.20 * mom + 0.25 * rs_val))
        bias = "看多" if comp >= 60 else ("看空" if comp <= 40 else "震荡")
        stop = None
        buy_zone = None
        if last is not None and atr is not None:
            sp = max(atr * 2.2, 7) / 100.0
            stop = round(last * (1 - sp), 2)
            # 建议买入价位：MA20 附近（保守 = MA20，积极 = 当前价回调 1/2 风险距离）
            if ma20 is not None:
                buy_zone = {
                    "conservative": round(min(last, ma20), 2),
                    "aggressive": round(last - (last - stop) * 0.5, 2),
                    "stop": stop,
                }
        rows.append({**e, "tech": round(tech), "val": round(val), "mom": round(mom),
                     "rs": round(rs_val), "comp": round(comp), "bias": bias,
                     "stop": stop, "buy_zone": buy_zone})
    intraday_t = sorted([r for r in rows if r.get("atrPct") and r["atrPct"] >= 2.5],
                        key=lambda r: -r["atrPct"])[:8]
    midterm = sorted([r for r in rows if r["comp"] >= 55 or r["bias"] == "看多"],
                     key=lambda r: -r["comp"])[:8]
    buy = sorted([r for r in rows if r["comp"] >= 60], key=lambda r: -r["comp"])
    return {
        "details": rows, "intraday_t": intraday_t, "midterm_hold": midterm, "buy": buy,
        "strategy_meta": {"threshold": 60, "med_pe": round(med_pe, 1), "med_ret20": round(med_ret20, 2),
                          "weights": {"tech": 0.35, "val": 0.20, "mom": 0.20, "rs": 0.25}},
    }


# ---------------------------------------------------------------------------
# 17. 明日观察位（规则引擎 + LLM 研判）
# ---------------------------------------------------------------------------

def _tw_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI(period)（Wilder 平滑），用于明日观察位风险维度。"""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-12)
    return 100 - 100 / (1 + rs)


def _tw_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR(period)（真实波幅均值）。"""
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _tw_macd_hist(close: pd.Series) -> pd.Series:
    """MACD 柱状值（默认 12/26/9）。"""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    return (dif - dea) * 2


def _tw_ma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n, min_periods=1).mean()


def _tw_evidence(text: str) -> str:
    return text.strip()


def compute_tomorrow_watch(
    symbol: str,
    df: Optional[pd.DataFrame] = None,
    capital_flow: Optional[Dict[str, Any]] = None,
    *,
    support_lookback: int = 20,
    breakout_vol_mult: float = 1.5,
    oversold_rsi: float = 30.0,
    overbought_rsi: float = 70.0,
    risk_vol_pct: float = 5.0,
) -> Dict[str, Any]:
    """
    「明日观察位」规则引擎：基于近 N 日 K 线 + 资金流（可选）生成 4 维研判。

    输入：
      - symbol:  yfinance 代码（"600519.SS" / "AAPL" / "0700.HK"）
      - df:      DataFrame(columns=[date,open,high,low,close,volume])，缺省时自动从
                 fetch_history_fixed() 取最近 60 个交易日
      - capital_flow: dict（东财主力资金），可选；keys 推荐 main_net_inflow, main_5d_sum, main_20d_sum（单位：元）

    输出 dict 结构：
      {
        "ok": True/False,                     # 数据是否足够
        "reason": "..." (失败原因),
        "support":   {state, status, value, evidence, score},
        "breakout":  {state, status, value, evidence, score},
        "capital":   {state, status, value, evidence, score},
        "risk":      {state, status, value, evidence, score},
        "overall":   {action, status, score, summary},  # 整体结论
        "raw":       {current_price, ma5/20, rsi, atr, vol_ratio, main_net_inflow, ...},
      }

    状态语义：
      - state  (中文): 守/观察/突破/放量/流入/中性/警示 等
      - status (色码): "ok" | "caution" | "danger" | "neutral"
      - score        : 0-100 信心分（用于综合排序/UI 强度条）
    """
    if df is None:
        # Fallback 1: yfinance 直接拉（最稳定，含 OHLCV）
        df = _safe_yf_ohlcv(symbol, days=120)
        # Fallback 2: 尝试 utils.fetch_history_fixed（返回 Series），仅收价 → 用日线 close + 内部 NaN
        if (df is None or len(df) < 20):
            try:
                hist = fetch_history_fixed(period="6mo")  # {ticker: Series(close)}
                ser = hist.get(symbol)
                if ser is not None and len(ser) >= 20:
                    s = ser.tail(60).reset_index()
                    s.columns = ["date", "close"]
                    df = s.assign(open=s["close"], high=s["close"], low=s["close"], volume=0)
            except Exception:  # noqa: BLE001
                pass
        if df is None or len(df) < 20:
            return {"ok": False, "reason": f"无 {symbol} 历史数据（yfinance/兜底源均不可用）", "overall": {
                "action": "观察", "status": "neutral", "score": 50, "summary": "数据不足，建议先观察。"
            }}

    if df is None or len(df) < 20 or "close" not in df.columns:
        return {"ok": False, "reason": "K 线行数不足（需 ≥20）", "overall": {
            "action": "观察", "status": "neutral", "score": 50, "summary": "样本不足，建议先观察。"
        }}

    # 实时价校正（港股/A股）：yfinance 最后一根 K 线可能停在昨收，
    # 用东财→腾讯→新浪多源实时价覆盖最后一根，避免「当前价=昨收」误导。
    _rt_prev: Optional[float] = None
    if symbol.endswith((".HK", ".SS", ".SZ")):
        try:
            _rq = fetch_realtime_quote(symbol)
            if _rq.get("ok") and _rq.get("last"):
                _df = df.copy()
                _df.loc[_df.index[-1], "close"] = float(_rq["last"])
                if "high" in _df.columns:
                    _df.loc[_df.index[-1], "high"] = max(float(_rq["last"]), float(_df["high"].iloc[-1]))
                if "low" in _df.columns:
                    _df.loc[_df.index[-1], "low"] = min(float(_rq["last"]), float(_df["low"].iloc[-1]))
                df = _df
                _rt_prev = _rq.get("prev_close")
        except Exception:  # noqa: BLE001
            pass

    df = df.tail(max(support_lookback + 10, 60)).reset_index(drop=True)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]
    cur = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) >= 2 else cur
    chg_pct = (cur / prev - 1.0) * 100 if prev > 0 else 0.0
    # 实时源提供了官方昨收 → 用它重算当日涨跌幅（yfinance bar 差值可能失真）
    if _rt_prev and _rt_prev > 0:
        chg_pct = (cur / float(_rt_prev) - 1.0) * 100

    ma5 = float(_tw_ma(close, 5).iloc[-1])
    ma20 = float(_tw_ma(close, 20).iloc[-1])
    rsi = float(_tw_rsi(close, 14).iloc[-1])
    atr = float(_tw_atr(df, 14).iloc[-1])
    atr_pct = (atr / cur * 100) if cur > 0 else 0.0
    macd_hist = float(_tw_macd_hist(close).iloc[-1])

    vol_today = float(vol.iloc[-1])
    vol_avg5 = float(vol.tail(5).mean())
    vol_avg20 = float(vol.tail(20).mean())
    vol_ratio5 = vol_today / vol_avg5 if vol_avg5 > 0 else 1.0
    vol_ratio20 = vol_today / vol_avg20 if vol_avg20 > 0 else 1.0

    # --- 1. 支撑观察 ---
    support_window = low.tail(support_lookback)
    support_level = float(support_window.min())
    dist_to_support = (cur - support_level) / support_level * 100 if support_level > 0 else 0
    if dist_to_support <= 1.5 and chg_pct <= 0:
        sup_state, sup_status, sup_score = "已破支撑", "danger", 25
        sup_ev = _tw_evidence(f"近 {support_lookback} 日低点 {support_level:.2f}，现价 {cur:.2f} 距支撑仅 {dist_to_support:.1f}%，且当日收跌 — 支撑已破。")
    elif dist_to_support <= 5:
        sup_state, sup_status, sup_score = "贴近支撑", "caution", 50
        sup_ev = _tw_evidence(f"近 {support_lookback} 日低点 {support_level:.2f}，现价 {cur:.2f} 距支撑 {dist_to_support:.1f}%，明日重点观察能否守稳。")
    elif chg_pct <= -2 and cur < ma5 < ma20:
        sup_state, sup_status, sup_score = "均线压制", "caution", 45
        sup_ev = _tw_evidence(f"现价 {cur:.2f} 跌破 MA5 ({ma5:.2f}) 且 MA5 低于 MA20 ({ma20:.2f})，短线结构转弱。")
    else:
        sup_state, sup_status, sup_score = "支撑稳固", "ok", 80
        sup_ev = _tw_evidence(f"近 {support_lookback} 日低点 {support_level:.2f}，现价 {cur:.2f} 高出 {dist_to_support:.1f}%，支撑稳固。")

    # --- 2. 放量信号（突破 vs 下跌）---
    high_vol = vol_ratio5 >= breakout_vol_mult
    big_up = chg_pct >= 1.5
    big_down = chg_pct <= -1.5
    if high_vol and big_up:
        br_state, br_status, br_score = "放量突破", "ok", 85
        br_ev = _tw_evidence(f"量比 {vol_ratio5:.2f}x（5 日均量）且当日 +{chg_pct:.1f}%，放量突破信号明确。")
    elif high_vol and big_down:
        br_state, br_status, br_score = "放量下跌", "danger", 20
        br_ev = _tw_evidence(f"量比 {vol_ratio5:.2f}x 且当日 {chg_pct:.1f}%，放量下跌 — 主力出货概率大。")
    elif high_vol:
        br_state, br_status, br_score = "放量震荡", "caution", 50
        br_ev = _tw_evidence(f"量比 {vol_ratio5:.2f}x 但当日仅 {chg_pct:+.1f}%，多空分歧、方向待定。")
    elif big_up:
        br_state, br_status, br_score = "缩量上行", "caution", 55
        br_ev = _tw_evidence(f"当日 +{chg_pct:.1f}% 但量比仅 {vol_ratio5:.2f}x，缩量上行需补量确认。")
    elif big_down:
        br_state, br_status, br_score = "缩量下跌", "ok", 60
        br_ev = _tw_evidence(f"当日 {chg_pct:.1f}% 但量比 {vol_ratio5:.2f}x，缩量下跌 — 抛压有限。")
    else:
        br_state, br_status, br_score = "缩量整理", "neutral", 50
        br_ev = _tw_evidence(f"量比 {vol_ratio5:.2f}x，当日 {chg_pct:+.1f}%，进入缩量整理阶段。")

    # --- 3. 主力资金 ---
    if capital_flow:
        main_today = float(capital_flow.get("main_net_inflow") or 0)
        main_5d = float(capital_flow.get("main_5d_sum") or 0)
        main_20d = float(capital_flow.get("main_20d_sum") or 0)
        # 阈值：>0 流入；<0 流出；累计 5 日 < 0 且当日 < 0 → 警示
        if main_today > 0 and main_5d > 0:
            cf_state, cf_status, cf_score = "持续流入", "ok", 80
            cf_ev = _tw_evidence(f"主力今日净流入 {main_today/1e4:.0f} 万，5 日累计 {main_5d/1e4:.0f} 万。")
        elif main_today > 0 and main_5d < 0:
            cf_state, cf_status, cf_score = "短期回流", "caution", 55
            cf_ev = _tw_evidence(f"主力今日净流入 {main_today/1e4:.0f} 万，但 5 日累计仍为 {main_5d/1e4:.0f} 万。")
        elif main_today < 0 and main_5d < 0:
            cf_state, cf_status, cf_score = "持续流出", "danger", 25
            cf_ev = _tw_evidence(f"主力今日净流出 {-main_today/1e4:.0f} 万，5 日累计 {-main_5d/1e4:.0f} 万 — 主力离场信号。")
        else:
            cf_state, cf_status, cf_score = "资金中性", "neutral", 50
            cf_ev = _tw_evidence(f"主力今日 {main_today/1e4:+.0f} 万，5 日累计 {main_5d/1e4:+.0f} 万，整体中性。")
        cf_value = {"today_wan": round(main_today/1e4, 1), "5d_wan": round(main_5d/1e4, 1),
                    "20d_wan": round(main_20d/1e4, 1)}
    else:
        cf_state, cf_status, cf_score = "资金未知", "neutral", 50
        cf_ev = _tw_evidence("暂无主力资金数据（未配置东财资金流接口），明日重点用其他三维度交叉验证。")
        cf_value = None

    # --- 4. 风险预警（综合 RSI / MACD / 波动率）---
    risk_notes: List[str] = []
    risk_danger = 0
    if rsi >= overbought_rsi:
        risk_danger += 1
        risk_notes.append(f"RSI {rsi:.0f} ≥ {overbought_rsi:.0f}，超买")
    elif rsi <= oversold_rsi:
        risk_notes.append(f"RSI {rsi:.0f} ≤ {oversold_rsi:.0f}，超卖（机会型）")
    if atr_pct >= risk_vol_pct:
        risk_danger += 1
        risk_notes.append(f"ATR/现价 {atr_pct:.1f}% ≥ {risk_vol_pct:.0f}%，高波动")
    if macd_hist < 0 and chg_pct < 0:
        risk_danger += 1
        risk_notes.append("MACD 柱状为负且当日收跌，动能向下")
    if cur > ma20 * 1.08:
        risk_danger += 1
        risk_notes.append(f"现价较 MA20 偏离 +{(cur/ma20-1)*100:.1f}%，过热")
    if risk_danger >= 2:
        rk_state, rk_status, rk_score = "风险偏高", "danger", 25
    elif risk_danger == 1:
        rk_state, rk_status, rk_score = "风险中性", "caution", 55
    else:
        rk_state, rk_status, rk_score = "风险可控", "ok", 80
    rk_ev = _tw_evidence("；".join(risk_notes) if risk_notes else "RSI / MACD / 波动率均在合理区间。")

    # --- 整体结论（按风险优先规则）---
    # 规则：风险预警 danger → 减仓；否则资金流 danger → 减仓；
    # 否则支撑 danger 或放量下跌 → 减仓/观察；否则全 ok → 持有；其他观察
    if rk_status == "danger":
        action, ov_status, ov_score = "减仓", "danger", 25
        ov_summary = f"风险预警维度评为「{rk_state}」({rk_ev}) — 优先降低风险暴露，谨防次日继续放量下跌。"
    elif cf_status == "danger":
        action, ov_status, ov_score = "减仓", "danger", 30
        ov_summary = f"主力资金持续流出（{cf_ev}），建议降低仓位以对冲主力离场风险。"
    elif br_status == "danger" or sup_status == "danger":
        action, ov_status, ov_score = "减仓/观望", "danger", 30
        if br_status == "danger":
            ov_summary = f"放量下跌信号 — {br_ev} 建议先减仓观察。"
        else:
            ov_summary = f"支撑位已破 — {sup_ev} 建议减仓等待企稳。"
    elif sup_status == "caution" or br_status == "caution" or cf_status == "caution":
        action, ov_status, ov_score = "观察", "caution", 50
        cautions = []
        if sup_status == "caution": cautions.append(f"支撑 {sup_state}")
        if br_status == "caution": cautions.append(f"量价 {br_state}")
        if cf_status == "caution": cautions.append(f"资金 {cf_state}")
        ov_summary = f"次要点位「{'; '.join(cautions)}」，建议明日盘中确认分时强度再决定加减仓。"
    else:
        action, ov_status, ov_score = "持有", "ok", 80
        ov_summary = f"四维度均为正向（{sup_state} / {br_state} / {cf_state} / {rk_state}），可继续持有，关注 MA5 ({ma5:.2f}) 守稳。"

    return {
        "ok": True,
        "symbol": symbol,
        "support":  {"state": sup_state, "status": sup_status, "score": sup_score, "value": round(support_level, 4),
                    "evidence": sup_ev, "distance_pct": round(dist_to_support, 2)},
        "breakout": {"state": br_state, "status": br_status, "score": br_score,
                    "value": round(vol_ratio5, 2), "evidence": br_ev,
                    "vol_ratio5": round(vol_ratio5, 2), "vol_ratio20": round(vol_ratio20, 2)},
        "capital":  {"state": cf_state, "status": cf_status, "score": cf_score, "value": cf_value,
                    "evidence": cf_ev, "has_data": bool(capital_flow)},
        "risk":     {"state": rk_state, "status": rk_status, "score": rk_score, "value": round(atr_pct, 2),
                    "evidence": rk_ev, "rsi": round(rsi, 1), "atr_pct": round(atr_pct, 2),
                    "macd_hist": round(macd_hist, 4)},
        "overall":  {"action": action, "status": ov_status, "score": ov_score, "summary": ov_summary},
        "raw": {
            "current_price": round(cur, 4),
            "prev_close": round(prev, 4),
            "chg_pct": round(chg_pct, 2),
            "ma5": round(ma5, 4),
            "ma20": round(ma20, 4),
            "rsi": round(rsi, 1),
            "atr": round(atr, 4),
            "atr_pct": round(atr_pct, 2),
            "macd_hist": round(macd_hist, 4),
            "vol_today": int(vol_today),
            "vol_avg5": int(vol_avg5),
            "vol_avg20": int(vol_avg20),
            "vol_ratio5": round(vol_ratio5, 2),
            "vol_ratio20": round(vol_ratio20, 2),
            "support_level": round(support_level, 4),
        },
    }


def _safe_yf_ohlcv(symbol: str, days: int = 120) -> Optional[pd.DataFrame]:
    """
    轻量级 yfinance OHLCV 拉取（仅在 compute_tomorrow_watch 内部 fallback 使用），
    不引入新依赖（yfinance 已在 utils.py 中使用）。失败返回 None。
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(_yf_sym(symbol))
        df = ticker.history(period=f"{max(60, days)}d", auto_adjust=False)
        if df is None or df.empty:
            return None
        df = df.reset_index()
        # 统一列名
        rename = {"Date": "date", "Datetime": "date", "index": "date",
                  "Open": "open", "High": "high", "Low": "low", "Close": "close",
                  "Volume": "volume"}
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        keep = [c for c in ("date", "open", "high", "low", "close", "volume") if c in df.columns]
        return df[keep].copy() if keep else None
    except Exception:  # noqa: BLE001
        return None


def narrate_tomorrow_watch(
    watch: Dict[str, Any],
    *,
    prefer: str = "deepseek",
    max_tokens: int = 350,
) -> Optional[str]:
    """
    LLM 自然语言研判：基于 compute_tomorrow_watch 结构化输出生成 150-250 字研判。
    无 LLM Key / 失败时返回 None —— 调用方应回退到 watch["overall"]["summary"]。
    """
    if not watch or not watch.get("ok"):
        return None
    raw = watch.get("raw", {})
    parts = [
        f"标的：{watch.get('symbol', '')}",
        f"现价：{raw.get('current_price')}（日 {raw.get('chg_pct', 0):+.2f}%）",
        f"MA5 / MA20：{raw.get('ma5')} / {raw.get('ma20')}",
        f"量比（5d / 20d）：{raw.get('vol_ratio5')} / {raw.get('vol_ratio20')}",
        f"RSI(14)：{raw.get('rsi')} | ATR%：{raw.get('atr_pct')}% | MACD 柱：{raw.get('macd_hist')}",
        f"近期支撑：{raw.get('support_level')}",
        "",
        "四维度结论：",
        f"· 支撑观察：{watch['support']['state']}（{watch['support']['evidence']}）",
        f"· 放量信号：{watch['breakout']['state']}（{watch['breakout']['evidence']}）",
        f"· 主力资金：{watch['capital']['state']}（{watch['capital']['evidence']}）",
        f"· 风险预警：{watch['risk']['state']}（{watch['risk']['evidence']}）",
        "",
        f"整体规则结论：{watch['overall']['action']} — {watch['overall']['summary']}",
    ]
    user = "\n".join(parts)
    system = (
        "你是专业 A 股 / 港股 / 美股 量化分析师。"
        "基于给出的结构化数据，输出一段 150-250 字的次日观察研判中文文本。"
        "要求：(1) 明确给出操作倾向（持有/观察/减仓）和触发条件；"
        "(2) 重点关注「放量下跌时优先提示降低风险暴露」；"
        "(3) 不要重复结构化字段原文；"
        "(4) 末尾加「非投资建议」四字。"
    )
    try:
        return _call_llm(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.55,
            max_tokens=max_tokens,
            prefer=prefer,
        )
    except Exception:  # noqa: BLE001
        return None


def fetch_capital_flow_eastmoney(symbol: str) -> Optional[Dict[str, Any]]:
    """
    东方财富个股主力资金流（best-effort，无 Key）。返回 dict 或 None。
    失败原因：接口反爬、限流、字段变动等；调用方应容错为 None 并降级。
    """
    if not symbol or not symbol.endswith((".SS", ".SZ")):
        return None
    try:
        mkt = "1" if symbol.endswith(".SS") else "0"
        code = symbol.replace(".SS", "").replace(".SZ", "")
        secid = f"{mkt}.{code}"
        url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
        params = {"secid": secid, "fields1": "f1,f2,f3,f4", "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
                  "klt": 101, "lmt": 30, "fqt": 0, "secid": secid}
        data = _safe_get_json(url, params=params, timeout=8)
        if not data:
            return None
        klines = (data.get("data") or {}).get("klines") or []
        if not klines:
            return None
        # 字段：日期,主力净流入(元),小单,中单,大单,特大单,主力净流入占比
        # 实际不同接口顺序略有不同，下面做最宽松解析
        rows: List[List[str]] = []
        for line in klines:
            parts = line.split(",")
            if len(parts) >= 3:
                rows.append(parts)
        if not rows:
            return None
        # 取最近一日 + 累计
        last = rows[-1]
        main_today = 0.0
        try:
            # 尝试解析主力净流入：通常在索引 1
            main_today = float(last[1]) if len(last) >= 2 else 0.0
        except (ValueError, IndexError):
            main_today = 0.0
        main_5d = 0.0
        main_20d = 0.0
        for i, r in enumerate(rows):
            try:
                v = float(r[1]) if len(r) >= 2 else 0.0
            except (ValueError, IndexError):
                v = 0.0
            if i >= len(rows) - 5:
                main_5d += v
            if i >= len(rows) - 20:
                main_20d += v
        return {
            "main_net_inflow": main_today,
            "main_5d_sum": main_5d,
            "main_20d_sum": main_20d,
            "source": "东方财富",
            "rows_n": len(rows),
        }
    except Exception:  # noqa: BLE001
        return None
