"""
utils.py — 投资分析工作台 (Investment Copilot) 工具模块

提供以下能力：
  1. SerpApi 多源新闻抓取（宏观 / 政策 / 个股），自动去重
  2. 自建 Fear & Greed 指数（5 因子模型：VIX、市场宽度、动量、安全资产、垃圾债利差）
  3. FedWatch 降息概率（用 SOFR/联邦基金期货反推下次会议概率）
  4. Economic Calendar（财报 / FOMC / CPI / PPI / NFP 抓取 + 静态兜底）
  5. 美股 / 港股热力图数据源（标普 500 + 纳指 100 + 恒生指数 + 国企指数）
  6. Morning Brief / Evening Recap 提示词模板

设计原则：
  - 所有外部调用都做超时 + 异常兜底，单点失败不能让 Streamlit 崩溃
  - 所有数据可被 Streamlit @st.cache_data 装饰（提供 hash 函数）
  - 不依赖付费 API（除 SERPAPI_KEY），所有 fallback 优先用 yfinance 免费数据
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
FOMC_2026_DATES = [
    "2026-01-28", "2026-01-29",
    "2026-03-17", "2026-03-18",
    "2026-04-28", "2026-04-29",
    "2026-06-16", "2026-06-17",
    "2026-07-28", "2026-07-29",
    "2026-09-15", "2026-09-16",
    "2026-10-27", "2026-10-28",
    "2026-12-15", "2026-12-16",
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

    current_ffr = 5.33  # 兜底；理想情况用 FRED DFF
    next_meeting = FOMC_2026_DATES[0] if FOMC_2026_DATES else "TBD"
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
STATIC_CALENDAR = [
    {"date": "2026-08-01", "time": "20:30", "event": "美国 7 月非农就业 (NFP)", "importance": "🔴 高", "type": "宏观"},
    {"date": "2026-08-05", "time": "未定", "event": "苹果 (AAPL) 财报", "importance": "🟠 中", "type": "财报"},
    {"date": "2026-08-12", "time": "20:30", "event": "美国 7 月 CPI 同比", "importance": "🔴 高", "type": "宏观"},
    {"date": "2026-08-14", "time": "20:30", "event": "美国 7 月 PPI 同比", "importance": "🟡 中", "type": "宏观"},
    {"date": "2026-08-15", "time": "02:00", "event": "FOMC 会议纪要公布", "importance": "🟠 中", "type": "宏观"},
    {"date": "2026-08-20", "time": "20:30", "event": "美联储主席 Powell 讲话 (Jackson Hole)", "importance": "🔴 高", "type": "宏观"},
    {"date": "2026-08-26", "time": "20:30", "event": "美国 7 月 PCE 物价指数", "importance": "🔴 高", "type": "宏观"},
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
                        "symbol": sym.replace(".HK", ""),
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
                        "symbol": sym.replace(".HK", ""),
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


def render_morning_brief(
    api_key: str,
    context: Dict[str, Any],
) -> Optional[str]:
    """调用 DeepSeek 生成 Morning Brief。"""
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        prompt = MORNING_BRIEF_PROMPT.format(**context)
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是专业卖方策略师，输出必须是简体中文，分析风格克制专业。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.65,
            max_tokens=1600,
        )
        return resp.choices[0].message.content
    except Exception as e:  # noqa: BLE001
        logger.error("Morning Brief 生成失败: %s", e)
        return None


def render_evening_recap(
    api_key: str,
    context: Dict[str, Any],
) -> Optional[str]:
    """调用 DeepSeek 生成 Evening Recap。"""
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        prompt = EVENING_RECAP_PROMPT.format(**context)
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是专业卖方策略师，输出必须是简体中文，分析风格克制专业。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.65,
            max_tokens=2000,
        )
        return resp.choices[0].message.content
    except Exception as e:  # noqa: BLE001
        logger.error("Evening Recap 生成失败: %s", e)
        return None


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
      - 2Y 当前收益率 (^IRX 不可用，2Y 用 ^FVX 错的；正确 ticker 是 ^FVX 5Y)
        → 2Y 没有完美 yfinance ticker，用 DGS2 from FRED
      - 10Y 收益率
      - 2s10s 利差
      - 利差走势（5 日变化）
    """
    out: Dict[str, Any] = {"y2": None, "y10": None, "spread_bps": None, "spread_5d_chg": None, "signal": "—", "asof": ""}
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
            fred_key = os.environ.get("FRED_API", "")
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
        except Exception as e:  # noqa: BLE001
            logger.debug("FRED DGS2 失败: %s", e)
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
        fred_key = os.environ.get("FRED_API", "")
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
# FRED: MDEBT (Margin Debt, All Customers, FINRA)
# 单位百万美元

def fetch_margin_debt() -> Dict[str, Any]:
    """FINRA 融资余额 = FRED MDEBT。"""
    out: Dict[str, Any] = {"value_billion": None, "mom_chg_pct": None, "yoy_chg_pct": None, "signal": "—", "asof": ""}
    try:
        from fredapi import Fred
        fred_key = os.environ.get("FRED_API", "")
        if not fred_key:
            return {**out, "error": "未配置 FRED_API"}
        fred = Fred(api_key=fred_key)
        s = fred.get_series("MDEBT", observation_start=(datetime.now() - timedelta(days=400)))
        if s is None or len(s) < 2:
            return {**out, "error": "FRED MDEBT 数据为空"}
        s = s.dropna()
        out["value_billion"] = round(float(s.iloc[-1]) / 1000.0, 2)  # 百万 → 十亿
        out["asof"] = str(s.index[-1].date())
        # 环比（一般 1-2 个月频率）
        if len(s) >= 2:
            mom = (float(s.iloc[-1]) / float(s.iloc[-2]) - 1.0) * 100
            out["mom_chg_pct"] = round(mom, 2)
        # 同比
        if len(s) >= 12:
            yoy = (float(s.iloc[-1]) / float(s.iloc[-12]) - 1.0) * 100
            out["yoy_chg_pct"] = round(yoy, 2)
        # 信号：融资余额快速上升 = 散户加杠杆（牛市后期）；快速下降 = 强平/恐慌
        if out["yoy_chg_pct"] is not None:
            yoy = out["yoy_chg_pct"]
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
        fred_key = os.environ.get("FRED_API", "")
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
        t = yf.Ticker(symbol)
        expirations = t.options or []
        if not expirations:
            out["error"] = "无可用期权到期日"
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
            out["error"] = "期权链为空"
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
    key = os.environ.get("FINNHUB_API", api_key)
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
    key = os.environ.get("NEWSAPI_KEY", api_key)
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
    东方财富网个股新闻（完全免费，无 key）。
    API: https://np-anotice-stock.eastmoney.com/api/security/ann?cb=&sr=-1&page_size=20&page_index=1&ann_type=A&client_source=web&stock_list=SZ000001
    """
    out: List[Dict[str, Any]] = []
    if not symbol.endswith((".HK", ".SS", ".SZ")):
        return out  # 仅 A 股 / 港股
    try:
        # 简化：转成东方财富内部代码
        if symbol.endswith(".HK"):
            # 港股东财代码不通用，跳过
            return out
        if symbol.endswith(".SS"):
            secid = f"1.{symbol.replace('.SS', '')}"
        elif symbol.endswith(".SZ"):
            secid = f"0.{symbol.replace('.SZ', '')}"
        else:
            return out
        url = "https://np-anotice-stock.eastmoney.com/api/security/ann"
        params = {"sr": -1, "page_size": 10, "page_index": 1, "ann_type": "A", "client_source": "web", "stock_list": secid}
        data = _safe_get_json(url, params=params)
        if not data or "data" not in data:
            return out
        for item in (data.get("data") or {}).get("list", [])[:10]:
            out.append({
                "title": item.get("title", ""),
                "link": item.get("art_code", ""),
                "source": "东方财富网",
                "date": item.get("notice_date", ""),
                "snippet": "",
            })
    except Exception as e:  # noqa: BLE001
        logger.debug("EastMoney %s 失败: %s", symbol, e)
    return out


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

    # 5) 个股新闻
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
        # 东方财富 (A 股)
        if is_cn:
            try:
                per_sym.extend(fetch_eastmoney_stock_news(sym))
            except Exception as e:  # noqa: BLE001
                pass
        payload["stocks"][sym] = _dedup_news(per_sym)[:8]
        if payload["stocks"][sym]:
            has_any = True

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
    api_key: str,
    symbol: str,
    technical: Dict[str, Any],
    news: List[Dict[str, Any]],
    policy_events: List[Dict[str, Any]],
    fundamentals: Dict[str, Any],
    options_data: Dict[str, Any],
) -> Optional[str]:
    """调用 DeepSeek 生成下周走势预测。"""
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        prompt = build_prediction_prompt(symbol, technical, news, policy_events, fundamentals, options_data)
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是专业卖方策略师，输出必须是简体中文，分析风格克制专业。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.55,
            max_tokens=1500,
        )
        return resp.choices[0].message.content
    except Exception as e:  # noqa: BLE001
        logger.error("预测生成失败 %s: %s", symbol, e)
        return None


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
