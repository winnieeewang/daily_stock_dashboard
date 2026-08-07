# -*- coding: utf-8 -*-
"""
市场底部判断模块 — 底部确信度(0-4) = 宏观环境分(0-2,全市场统一) + 个股结构分(0-2,逐股)

阶段1 已实现（零新数据源，纯计算）：
  · 热力图离散度（累积60天历史后启用分位）
  · 同板块PE相对排名
  · PE历史分位框架（data/pe_history.json 逐日累积，满6个月启用）
  · 政策新闻关键词计数

架构原则：
- 维科夫/多因子保持纯计算独立，底部信号灯并列展示
- 只在第三层(LLM叙事)做交叉引用，不互相修改分数
- 交通灯语义：🔴 0-1分(观望) 🟡 2分(部分确认) 🟢 3-4分(多维度共振)

接口（任务书要求）：
  compute_macro_environment_score() -> dict{"score", "hits", "details"}
  compute_stock_structure_score(symbol, df) -> dict{"score", "hits", "details"}
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
PE_HISTORY_PATH = DATA_DIR / "pe_history.json"
DISPERSION_HISTORY_PATH = DATA_DIR / "dispersion_history.json"
BOTTOM_SCORE_PATH = DATA_DIR / "bottom_scores.json"

# 政策新闻恐慌关键词（维度②）— 任务书精确列表
PANIC_KEYWORDS = [
    "紧急", "平准基金", "国家队", "限制卖空",
    "熔断规则调整", "央行声明",
]


def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return None if f != f else f
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 交通灯
# ---------------------------------------------------------------------------
def traffic_light(score: int) -> Tuple[str, str, str]:
    """返回 (emoji, 中文标签, CSS色)。"""
    if score <= 1:
        return "🔴", "红灯·继续观望", "#dc2626"
    if score == 2:
        return "🟡", "黄灯·部分确认", "#ca8a04"
    return "🟢", "绿灯·多维度共振", "#16a34a"


# ---------------------------------------------------------------------------
# PE 历史累积（维度③）
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_pe_snapshot(symbol: str, pe: Optional[float], date_str: Optional[str] = None) -> None:
    """每日跑批时调用：把当天 PE 追加进历史文件。"""
    if pe is None or pe != pe:
        return
    hist = _load_json(PE_HISTORY_PATH)
    sym_hist = hist.get(symbol, [])
    d = date_str or datetime.now().strftime("%Y-%m-%d")
    sym_hist = [(day, val) for day, val in sym_hist if day != d]
    sym_hist.append((d, round(float(pe), 4)))
    sym_hist = sym_hist[-800:]  # 保留约3年
    hist[symbol] = sym_hist
    _save_json(PE_HISTORY_PATH, hist)


def _pe_history_values(symbol: str, min_days: int = 120) -> Optional[List[float]]:
    """返回 symbol 的 PE 历史值列表；不足 min_days 返回 None。"""
    hist = _load_json(PE_HISTORY_PATH).get(symbol, [])
    vals = [v for _, v in hist]
    if len(vals) < min_days:
        return None
    return vals


def dim3_pe_percentile(symbol: str, pe: Optional[float] = None) -> Optional[bool]:
    """
    PE 处于自己滚动历史(建议3年)的后20分位 → 命中。
    数据不足6个月(~120交易日) → 返回 None（展示层显示"数据积累中"）。
    """
    if pe is None:
        return None
    vals = _pe_history_values(symbol, min_days=120)
    if vals is None:
        return None
    s = pd.Series(vals)
    pct = s.rank(pct=True).iloc[-1]
    return pct <= 0.20


def dim3_erp_high(symbol: str, pe: Optional[float] = None,
                  treasury_yield: Optional[float] = None) -> Optional[bool]:
    """
    ERP = 1/PE − 10年期美债收益率，处于自己历史高位(80分位以上) → 命中。
    数据不足返回 None。
    """
    if pe is None or treasury_yield is None or pe <= 0:
        return None
    vals = _pe_history_values(symbol, min_days=120)
    if vals is None:
        return None
    erps = [(1.0 / v) - (treasury_yield / 100.0) for v in vals if v > 0]
    if len(erps) < 120:
        return None
    s = pd.Series(erps)
    return s.rank(pct=True).iloc[-1] >= 0.80


# ---------------------------------------------------------------------------
# 同板块PE相对排名（维度③）
# ---------------------------------------------------------------------------

def dim3_sector_relative_pe(symbol: str, stocks_df: Optional[pd.DataFrame] = None,
                            sector_map: Optional[Dict[str, str]] = None) -> bool:
    """同板块相对估值排名靠后(组内PE分位排后25%) → 命中。零新数据源。"""
    if stocks_df is None or stocks_df.empty:
        return False
    if sector_map is None:
        sector_map = {
            "MSFT": "科技", "GOOGL": "科技", "NVDA": "科技", "ORCL": "科技",
            "AMZN": "科技", "META": "科技", "AAPL": "科技", "PLTR": "科技",
            "MU": "半导体", "MRVL": "半导体", "LITE": "半导体", "SNDK": "半导体",
            "TSLA": "汽车", "NIO": "汽车", "XPEV": "汽车",
            "0700.HK": "科技", "0883.HK": "能源", "3750.HK": "新能源",
            "01879.HK": "科技", "00700.HK": "科技",
        }
    sector = sector_map.get(symbol)
    if not sector:
        return False
    peers = [s for s, sec in sector_map.items()
             if sec == sector and s in stocks_df["symbol"].values]
    if len(peers) < 3:
        return False
    pe_vals = []
    for p in peers:
        row = stocks_df[stocks_df["symbol"] == p]
        if not row.empty:
            pe = _safe_float(row.iloc[0].get("PE_Ratio"))
            if pe is not None and pe > 0:
                pe_vals.append((p, pe))
    if len(pe_vals) < 3:
        return False
    pe_vals.sort(key=lambda x: x[1])
    rank = next((i for i, (s, _) in enumerate(pe_vals) if s == symbol), len(pe_vals))
    return rank / len(pe_vals) <= 0.25


# ---------------------------------------------------------------------------
# 热力图离散度（维度④辅助 + 维度①拥挤出清辅助）
# ---------------------------------------------------------------------------

def _append_dispersion(market: str, dispersion: float, date_str: Optional[str] = None) -> None:
    """每日跑批时调用：把当天离散度追加进历史。"""
    hist = _load_json(DISPERSION_HISTORY_PATH)
    mkt_hist = hist.get(market, [])
    d = date_str or datetime.now().strftime("%Y-%m-%d")
    mkt_hist = [(day, val) for day, val in mkt_hist if day != d]
    mkt_hist.append((d, round(float(dispersion), 4)))
    mkt_hist = mkt_hist[-80:]  # 保留约4个月交易日
    hist[market] = mkt_hist
    _save_json(DISPERSION_HISTORY_PATH, hist)


def dim1_heatmap_dispersion(market: str = "US") -> Tuple[Optional[bool], str]:
    """
    算当天该市场热力图 change_pct 的标准差，与过去60天分布比较。
    处于后1/3(离散度低=普涨普跌) → 命中"资金系统性进出" → True。
    处于前1/3(离散度高) → 倾向"换仓" → False。
    历史不足60天 → 返回 (None, "数据积累中")。
    """
    try:
        import utils as U
        # 根据 market 选 ticker 池
        if market == "US":
            universe = U.US_HEATMAP_TICKERS
        elif market == "HK":
            universe = U.HK_HEATMAP_TICKERS
        elif market == "CN":
            universe = U.A_SHARE_HEATMAP_TICKERS
        else:
            universe = U.US_HEATMAP_TICKERS

        hm = U.build_heatmap_data(universe)
        if hm is None or hm.empty or "change_pct" not in hm.columns:
            return None, "无数据"
        vals = pd.to_numeric(hm["change_pct"], errors="coerce").dropna()
        if len(vals) < 5:
            return None, "样本不足"
        today_disp = float(vals.std())

        # 读取历史
        hist = _load_json(DISPERSION_HISTORY_PATH).get(market, [])
        hist_vals = [v for _, v in hist]
        # 追加今天（但先不保存，由跑批统一保存）
        all_vals = hist_vals + [today_disp]

        if len(hist_vals) < 60:
            return None, f"离散度 {today_disp:.2f}% (历史 {len(hist_vals)}/60 天，积累中)"

        # 过去60天分布（不含今天）
        past60 = hist_vals[-60:]
        s = pd.Series(past60)
        q33 = s.quantile(0.33)
        q66 = s.quantile(0.66)

        if today_disp <= q33:
            return True, f"离散度 {today_disp:.2f}% (低，系统性进出)"
        if today_disp >= q66:
            return False, f"离散度 {today_disp:.2f}% (高，板块换仓)"
        return False, f"离散度 {today_disp:.2f}% (中性)"
    except Exception as e:
        return None, f"计算失败: {e}"


# ---------------------------------------------------------------------------
# 政策新闻关键词计数（维度②）
# ---------------------------------------------------------------------------

def dim2_policy_news_panic_freq(policy_news: Optional[List[Dict]] = None,
                                threshold_multiplier: float = 2.0) -> bool:
    """
    政策新闻里恐慌关键词频次较过去7天均值明显放大 → 命中。
    关键词表按任务书精确设定。
    """
    try:
        import utils as U
        if policy_news is None:
            policy_news = U.fetch_policy_news_free(top_n=30)
        if not policy_news:
            return False

        daily_counts: Dict[str, int] = {}
        for item in policy_news:
            text = f"{item.get('title', '')} {item.get('summary', '')}"
            day = item.get("date", datetime.now().strftime("%Y-%m-%d"))
            hits = sum(1 for kw in PANIC_KEYWORDS if kw in text)
            if hits:
                daily_counts[day] = daily_counts.get(day, 0) + hits

        if not daily_counts:
            return False

        days = sorted(daily_counts.keys())
        if len(days) < 2:
            return daily_counts.get(days[-1], 0) >= 3

        today = days[-1]
        today_count = daily_counts[today]
        hist_avg = sum(daily_counts[d] for d in days[:-1]) / max(len(days) - 1, 1)
        return today_count >= max(hist_avg * threshold_multiplier, 2)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 维度① · 拥挤交易出清（个股）
# ---------------------------------------------------------------------------

def dim1_wyckoff_sc(df: Optional[pd.DataFrame] = None) -> bool:
    """维科夫事件序列检测到 SC(抛售高潮) → 命中。df 为历史K线。"""
    if df is None or df.empty:
        return False
    try:
        import screener as S
        w = S.detect_wyckoff_events(df)
        if not w.get("ok"):
            return False
        events = w.get("events", [])
        return any(e.get("event") == "SC" or "抛售高潮" in str(e) for e in events)
    except Exception:
        return False


def dim1_oi_pcr_fallback(symbol: str) -> bool:
    """OI PCR 从近20日高点回落超15%。P3实现前返回 False。"""
    return False


# ---------------------------------------------------------------------------
# 宏观环境分 (0-2)
# ---------------------------------------------------------------------------

def compute_macro_environment_score(policy_news: Optional[List[Dict]] = None,
                                    market: str = "US") -> Dict[str, Any]:
    """
    宏观环境分(0-2) = 监管恐慌命中(0/1) + 杠杆去化命中(0/1)
    返回 {"score": int, "hits": List[str], "details": dict}
    """
    details: Dict[str, Any] = {}
    hits: List[str] = []
    score = 0

    # 维度②：监管恐慌（任一命中即+1）
    d2_panic = dim2_policy_news_panic_freq(policy_news)
    d2_hit = d2_panic  # 其他子指标(P3)暂未实现
    if d2_hit:
        score += 1
        hits.append("政策新闻出现恐慌关键词放大")
    details["监管恐慌"] = {
        "命中": d2_hit,
        "政策新闻恐慌": d2_panic,
        "FedWatch紧急降息": False,   # P3
        "VIX期限倒挂": False,         # P3
        "贴现窗口骤增": False,        # P3
    }

    # 维度①宏观：杠杆去化（Margin Debt二阶导等）
    # P3实现前，暂时用热力图离散度低（系统性进出）作为 proxy
    d1_disp, d1_disp_txt = dim1_heatmap_dispersion(market)
    d1_hit = d1_disp if d1_disp is not None else False
    if d1_hit:
        score += 1
        hits.append(f"市场离散度低({d1_disp_txt})，倾向系统性资金进出")
    details["杠杆去化"] = {
        "命中": d1_hit,
        "MarginDebt二阶导": False,    # P3
        "CFTC_COT": False,            # P3
        "热力图离散度低": d1_disp,
        "离散度描述": d1_disp_txt,
    }

    return {"score": score, "hits": hits, "details": details}


# ---------------------------------------------------------------------------
# 个股结构分 (0-2)
# ---------------------------------------------------------------------------

def compute_stock_structure_score(symbol: str,
                                  df: Optional[pd.DataFrame] = None,
                                  pe: Optional[float] = None,
                                  stocks_df: Optional[pd.DataFrame] = None,
                                  treasury_yield: Optional[float] = None,
                                  sector_map: Optional[Dict[str, str]] = None,
                                  ) -> Dict[str, Any]:
    """
    个股结构分(0-2) = 拥挤出清命中(0/1) + 估值资金综合命中(0/1)
    df 为该股票的历史K线 DataFrame（复用 _fetch_price_history）。
    返回 {"score": int, "hits": List[str], "details": dict}
    """
    details: Dict[str, Any] = {}
    hits: List[str] = []
    score = 0

    # 维度①个股：拥挤出清
    d1_sc = dim1_wyckoff_sc(df)
    d1_oi = dim1_oi_pcr_fallback(symbol)
    d1_hit = any([d1_sc, d1_oi])
    if d1_hit:
        score += 1
        if d1_sc:
            hits.append("维科夫检测到抛售高潮(SC)")
        if d1_oi:
            hits.append("OI PCR从高点回落")
    details["拥挤出清"] = {
        "命中": d1_hit,
        "维科夫SC": d1_sc,
        "OI_PCR回落": d1_oi,
    }

    # 维度③④综合：估值跌透 + 资金动向
    d3_pe = dim3_pe_percentile(symbol, pe)
    d3_erp = dim3_erp_high(symbol, pe, treasury_yield)
    d3_sector = dim3_sector_relative_pe(symbol, stocks_df, sector_map)

    # PEG<1 判断（从 yfinance info 取 pegRatio）
    d3_peg = False
    if not symbol.endswith((".HK", ".SS", ".SZ")):
        try:
            import yfinance as yf
            info = yf.Ticker(symbol).info or {}
            peg = info.get("pegRatio")
            if peg is not None and float(peg) < 1.0:
                d3_peg = True
        except Exception:
            d3_peg = False

    d3_hits = []
    if d3_pe is True:
        d3_hits.append("PE处于历史低位(后20分位)")
    if d3_erp is True:
        d3_hits.append("ERP处于历史高位")
    if d3_sector:
        d3_hits.append("同板块相对估值排名靠后")
    if d3_peg:
        d3_hits.append("PEG<1(成长性价比)")

    d3_hit = any(x is True for x in [d3_pe, d3_erp, d3_sector]) or d3_peg

    # 维度④资金：北向（仅A股）
    d4_north = False
    if symbol.endswith((".SS", ".SZ")):
        # 北向资金逻辑保留，现阶段简化
        d4_north = False

    d34_hit = d3_hit or d4_north
    if d34_hit:
        score += 1
        hits.extend(d3_hits)
        if d4_north:
            hits.append("北向资金持续净流出")

    details["估值跌透"] = {
        "命中": d3_hit,
        "PE历史低位": d3_pe,
        "ERP高位": d3_erp,
        "板块相对低位": d3_sector,
        "PEG小于1": d3_peg,
    }
    details["资金动向"] = {
        "命中": d4_north,
        "北向净流出": d4_north,
    }

    return {"score": score, "hits": hits, "details": details}


# ---------------------------------------------------------------------------
# 底部确信度整合 (0-4)
# ---------------------------------------------------------------------------

def calc_bottom_confidence(symbol: str,
                           macro: Optional[Dict[str, Any]] = None,
                           df: Optional[pd.DataFrame] = None,
                           pe: Optional[float] = None,
                           stocks_df: Optional[pd.DataFrame] = None,
                           treasury_yield: Optional[float] = None,
                           ) -> Dict[str, Any]:
    """
    计算单只股票的底部确信度(0-4)。
    macro 建议外层批量算一次复用；df 为历史K线。
    """
    if macro is None:
        macro = compute_macro_environment_score()

    ind = compute_stock_structure_score(
        symbol, df=df, pe=pe, stocks_df=stocks_df, treasury_yield=treasury_yield,
    )
    total = macro["score"] + ind["score"]
    emoji, label, color = traffic_light(total)

    return {
        "symbol": symbol,
        "bottom_score": total,
        "macro_score": macro["score"],
        "individual_score": ind["score"],
        "traffic_light": emoji,
        "label": label,
        "color": color,
        "macro_hits": macro.get("hits", []),
        "individual_hits": ind.get("hits", []),
        "macro_details": macro.get("details", {}),
        "individual_details": ind.get("details", {}),
        "asof": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# ---------------------------------------------------------------------------
# 批量计算 & 持久化（供每日 CI / 验收调用）
# ---------------------------------------------------------------------------

def batch_bottom_signals(symbols: List[str],
                         stocks_df: Optional[pd.DataFrame] = None,
                         policy_news: Optional[List[Dict]] = None,
                         treasury_yield: Optional[float] = None,
                         save: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    批量计算底部确信度。宏观环境分只算一次，逐股复用。
    返回 {symbol: result_dict}；可选保存到 data/bottom_scores.json。
    """
    macro = compute_macro_environment_score(policy_news=policy_news)
    results = {}
    for sym in symbols:
        pe = None
        if stocks_df is not None and not stocks_df.empty:
            row = stocks_df[stocks_df["symbol"] == sym]
            if not row.empty:
                pe = _safe_float(row.iloc[0].get("PE_Ratio"))
        r = calc_bottom_confidence(
            sym, macro=macro, pe=pe,
            stocks_df=stocks_df, treasury_yield=treasury_yield,
        )
        results[sym] = r

    if save:
        _save_json(BOTTOM_SCORE_PATH, {
            "generated_at": datetime.now().isoformat(),
            "scores": results,
        })
    return results


def load_bottom_scores() -> Dict[str, Dict[str, Any]]:
    """加载已保存的底部确信度。"""
    return _load_json(BOTTOM_SCORE_PATH).get("scores", {})
