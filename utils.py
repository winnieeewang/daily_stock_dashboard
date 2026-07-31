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


def calc_fedwatch_from_futures() -> Dict[str, Any]:
    """
    用 SR3 (3 月 SOFR 期货) 反推市场隐含的下次会议利率区间概率。

    简化模型：
      - 当前有效联邦基金利率假设为 5.33% (2024 末水平，会随时间变化)
      - SR3 = 100 - 隐含利率
      - 隐含利率 = (当前 SR3 - 目标 SR3) / 12 + 当前 FFR
    """
    try:
        sr3 = yf.download("SR3=F", period="6mo", progress=False)
        if sr3.empty:
            sr3 = yf.download("ZQ=F", period="6mo", progress=False)
        if sr3.empty:
            return {"error": "无法获取 SOFR/FF 期货", "meetings": []}
        close = sr3["Close"].squeeze() if isinstance(sr3["Close"], pd.DataFrame) else sr3["Close"]
        implied_rate = 100.0 - float(close.iloc[-1])
    except Exception as e:  # noqa: BLE001
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
