# -*- coding: utf-8 -*-
"""
市场底部判断模块 — 底部确信度(0-4) = 宏观环境分(0-2) + 个股结构分(0-2)

架构原则：
- 第一层(维科夫)保持纯计算、不受其他信号干扰
- 底部信号灯与维科夫/多因子并列展示，只在第三层(LLM叙事)做交叉引用
- 交通灯语义：🔴 0-1分(观望) 🟡 2分(部分确认) 🟢 3-4分(多维度共振)

实施优先级：
  P1(零新数据源,纯计算): 热力图离散度、同板块PE排名、PE历史分位框架、政策新闻关键词
  P2(小改动,加ticker):  避险资产联动(GLD/TLT/DXY)、PEG
  P3(新开管线):           OI PCR序列化、FedWatch非议息窗口、VIX期限结构倒挂
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# 常量 / 配置
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
PE_HISTORY_PATH = DATA_DIR / "pe_history.json"
BOTTOM_SCORE_PATH = DATA_DIR / "bottom_scores.json"

# 政策新闻恐慌关键词（维度②）
PANIC_KEYWORDS = ["紧急", "平准基金", "国家队", "限制卖空", "熔断", "熔断规则调整",
                  "救市", "系统性风险", "流动性危机", "踩踏"]

# 避险资产 ticker（维度④）
SAFE_HAVEN_TICKERS = {"GLD": "黄金ETF", "TLT": "美债20年ETF", "DXY": "美元指数"}


def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return None if f != f else f  # NaN check
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
# 维度① · 拥挤交易出清
# ---------------------------------------------------------------------------

def dim1_macro_margin_debt() -> bool:
    """
    FINRA Margin Debt 同比增速虽仍为负，但比上月收窄（二阶导转正）→ 命中
    复用 utils.fetch_margin_debt 的输出。
    """
    try:
        import utils as U
        md = U.fetch_margin_debt()
        if not md or "error" in md:
            return False
        # 需要同比序列；utils 目前返回最新值，需要扩展
        # 现阶段：若返回的同比变化率有趋势字段则使用，否则保守返回 False
        yoy = md.get("同比变化率")
        if yoy is None:
            return False
        # TODO: 需要月环比二阶导；当前单点数据无法判断，返回 False 待后续增强
        return False
    except Exception:
        return False


def dim1_macro_cftc_cot() -> bool:
    """CFTC COT 对冲基金净多头周环比大幅下降(>52周90分位) → 命中。优先级低，暂不实现。"""
    return False


def dim1_individual_wyckoff_sc(symbol: str, wyckoff_events: Optional[Dict] = None) -> bool:
    """
    维科夫事件序列检测到 SC(抛售高潮) → 命中。
    复用现有 screener.detect_wyckoff_events 的输出。
    """
    if wyckoff_events is None:
        try:
            import screener as S
            wyckoff_events = S.detect_wyckoff_events(symbol)
        except Exception:
            return False
    if not wyckoff_events:
        return False
    events = wyckoff_events.get("events", [])
    return "SC" in events or "抛售高潮" in str(events)


def dim1_individual_oi_pcr_fallback(symbol: str) -> bool:
    """
    OI PCR 从近20日高点回落超15%。
    目前 options_pcr.json 只有最新快照，需要序列化(P3)。
    现阶段用 Volume PCR 或 OI_PCR 单点值做尽力而为判断。
    """
    # P3 实现前，保守返回 False
    return False


# ---------------------------------------------------------------------------
# 维度② · 监管层恐慌（全部宏观）
# ---------------------------------------------------------------------------

def dim2_fedwatch_emergency_cut() -> bool:
    """
    FedWatch 定价出现"非议息窗口"紧急降息概率。
    需要扩展现有 calc_fedwatch_from_futures：把隐含利率路径变化日期和 FOMC 日历比对。
    P3 实现前返回 False。
    """
    return False


def dim2_policy_news_panic_freq(policy_news: Optional[List[Dict]] = None,
                                threshold_multiplier: float = 2.0) -> bool:
    """
    政策新闻里恐慌关键词频次较过去7天均值明显放大 → 命中。
    纯计算，零新数据源。
    """
    try:
        import utils as U
        if policy_news is None:
            policy_news = U.fetch_policy_news_free(top_n=30)
        if not policy_news:
            return False

        # 按天聚合关键词命中次数
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
            return daily_counts.get(days[-1], 0) >= 3  # 单日≥3次算异常

        today = days[-1]
        today_count = daily_counts[today]
        hist_avg = sum(daily_counts[d] for d in days[:-1]) / max(len(days) - 1, 1)
        return today_count >= max(hist_avg * threshold_multiplier, 2)
    except Exception:
        return False


def dim2_vix_term_structure() -> bool:
    """VIX 期货期限结构倒挂(近月贵于远月)。需要新增 CBOE 数据源。P3。"""
    return False


def dim2_fed_discount_window() -> bool:
    """美联储贴现窗口使用量骤增。FRED 周度数据。P3。"""
    return False


# ---------------------------------------------------------------------------
# 维度③ · 估值跌透（全部个股）
# ---------------------------------------------------------------------------

def _load_pe_history() -> Dict[str, List[Tuple[str, float]]]:
    """加载 PE 历史时间序列：{symbol: [(date, pe), ...]}。"""
    if PE_HISTORY_PATH.exists():
        try:
            with open(PE_HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_pe_history(data: Dict[str, List[Tuple[str, float]]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PE_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_pe_snapshot(symbol: str, pe: Optional[float], date_str: Optional[str] = None) -> None:
    """每日跑批时调用：把当天 PE 追加进历史文件。"""
    if pe is None or pe != pe:  # NaN
        return
    hist = _load_pe_history()
    sym_hist = hist.get(symbol, [])
    d = date_str or datetime.now().strftime("%Y-%m-%d")
    # 去重：同一天只保留最新
    sym_hist = [(day, val) for day, val in sym_hist if day != d]
    sym_hist.append((d, round(float(pe), 4)))
    # 保留最近 3 年 (~750交易日)
    sym_hist = sym_hist[-800:]
    hist[symbol] = sym_hist
    _save_pe_history(hist)


def dim3_pe_percentile(symbol: str, pe: Optional[float] = None,
                       lookback_days: int = 750) -> bool:
    """
    PE 处于自己滚动历史(建议3年)的后20分位 → 命中。
    数据不足时退化为全部历史。
    """
    if pe is None:
        return False
    hist = _load_pe_history().get(symbol, [])
    if not hist:
        return False
    values = [v for _, v in hist[-lookback_days:]]
    if len(values) < 30:
        return False
    pct = pd.Series(values).rank(pct=True).iloc[-1]
    return pct <= 0.20


def dim3_erp_high(symbol: str, pe: Optional[float] = None,
                  treasury_yield: Optional[float] = None) -> bool:
    """
    ERP = 1/PE − 10年期美债收益率，处于自己历史高位(80分位以上) → 命中。
    依赖 PE 历史文件。
    """
    if pe is None or treasury_yield is None:
        return False
    if pe <= 0:
        return False
    erp = (1.0 / pe) - (treasury_yield / 100.0)
    hist = _load_pe_history().get(symbol, [])
    if not hist:
        return False
    # 用已存 PE 反推历史 ERP（近似，未存 treasury_yield 历史）
    # 保守做法：假设收益率变化较慢，直接用当前 treasury_yield 反推
    values = [v for _, v in hist]
    if len(values) < 30:
        return False
    erps = [(1.0 / v) - (treasury_yield / 100.0) for v in values if v > 0]
    if len(erps) < 30:
        return False
    s = pd.Series(erps)
    return s.rank(pct=True).iloc[-1] >= 0.80


def dim3_sector_relative_pe(symbol: str, sector_map: Optional[Dict[str, str]] = None,
                            stocks_df: Optional[pd.DataFrame] = None) -> bool:
    """
    同板块相对估值排名靠后(用 sectors 标签分组，PE 分位在组内排后25%) → 命中。
    零新数据源，纯计算。
    """
    if stocks_df is None or stocks_df.empty:
        return False
    if sector_map is None:
        # 简单硬编码映射（后续可从外部注入）
        sector_map = {
            "MSFT": "科技", "GOOGL": "科技", "NVDA": "科技", "ORCL": "科技",
            "AMZN": "科技", "META": "科技", "AAPL": "科技",
            "MU": "半导体", "MRVL": "半导体", "LITE": "半导体", "SNDK": "半导体",
            "TSLA": "汽车", "NIO": "汽车", "XPEV": "汽车",
            "0700.HK": "科技", "0883.HK": "能源", "3750.HK": "新能源",
        }
    sector = sector_map.get(symbol)
    if not sector:
        return False
    peers = [s for s, sec in sector_map.items() if sec == sector and s in stocks_df["symbol"].values]
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


def dim3_peg_growth(symbol: str, info: Optional[Dict] = None) -> bool:
    """成长股 PEG<1 → 命中。需要 ticker.info 的 earningsGrowth/pegRatio。P2。"""
    if info and info.get("pegRatio") is not None:
        return info["pegRatio"] < 1.0
    return False


# ---------------------------------------------------------------------------
# 维度④ · 资金走了还是换仓
# ---------------------------------------------------------------------------

def dim4_heatmap_dispersion(heatmap_df: Optional[pd.DataFrame] = None) -> Tuple[bool, str]:
    """
    热力图离散度标准差处于过去60天前1/3(高离散)→"换仓"；后1/3→"系统性进出"。
    返回 (是否高离散, 描述文本)。零新数据源。
    """
    try:
        import utils as U
        if heatmap_df is None:
            # 尽力构建当前热力图
            heatmap_df = U.build_heatmap_data()
        if heatmap_df is None or heatmap_df.empty or "涨跌幅" not in heatmap_df.columns:
            return False, "无数据"
        vals = pd.to_numeric(heatmap_df["涨跌幅"], errors="coerce").dropna()
        if len(vals) < 5:
            return False, "样本不足"
        dispersion = float(vals.std())
        # 需要60天历史；目前只有单天，保守返回 False
        # TODO: 累积60天离散度历史后启用分位判断
        return False, f"离散度 {dispersion:.2f}% (待累积60天历史)"
    except Exception:
        return False, "计算失败"


def dim4_north_flow_a_share(north_flow: Optional[float] = None) -> bool:
    """北向资金持续净流出 → 对'资金走了'倾向加权。仅 A股适用。"""
    if north_flow is None:
        return False
    return north_flow < -20  # 净流出超20亿算显著


def dim4_safe_haven_correlation(symbol: str) -> Tuple[bool, str]:
    """
    避险资产联动：GLD/TLT/DXY 与 symbol 的30日滚动相关系数。
    需要新增日常抓取(P2)。现阶段返回 False。
    """
    return False, "待部署避险资产日常抓取(GLD/TLT/DXY)"


# ---------------------------------------------------------------------------
# 综合打分
# ---------------------------------------------------------------------------

def calc_macro_env_score(margin_debt_ok: Optional[bool] = None,
                         policy_news: Optional[List[Dict]] = None,
                         vix_inverted: Optional[bool] = None,
                         fed_window: Optional[bool] = None,
                         fedwatch_emergency: Optional[bool] = None) -> Tuple[int, Dict[str, Any]]:
    """
    宏观环境分(0-2) = 维度②监管恐慌命中(0-1) + 维度①宏观部分(杠杆去化)命中(0-1)
    返回 (分数, 明细字典)。
    """
    detail: Dict[str, Any] = {}
    score = 0

    # 维度②：监管恐慌（任一命中即+1）
    d2_panic = dim2_policy_news_panic_freq(policy_news)
    d2_fed = fedwatch_emergency if fedwatch_emergency is not None else dim2_fedwatch_emergency_cut()
    d2_vix = vix_inverted if vix_inverted is not None else dim2_vix_term_structure()
    d2_window = fed_window if fed_window is not None else dim2_fed_discount_window()
    d2_hit = any([d2_panic, d2_fed, d2_vix, d2_window])
    if d2_hit:
        score += 1
    detail["维度②_监管恐慌"] = {
        "命中": d2_hit,
        "政策新闻恐慌": d2_panic,
        "FedWatch紧急降息": d2_fed,
        "VIX期限倒挂": d2_vix,
        "贴现窗口骤增": d2_window,
    }

    # 维度①宏观：杠杆去化
    d1_margin = margin_debt_ok if margin_debt_ok is not None else dim1_macro_margin_debt()
    d1_cftc = dim1_macro_cftc_cot()
    d1_hit = any([d1_margin, d1_cftc])
    if d1_hit:
        score += 1
    detail["维度①_宏观"] = {
        "命中": d1_hit,
        "MarginDebt二阶导": d1_margin,
        "CFTC_COT": d1_cftc,
    }

    return score, detail


def calc_individual_structure_score(symbol: str,
                                    wyckoff_events: Optional[Dict] = None,
                                    pe: Optional[float] = None,
                                    treasury_yield: Optional[float] = None,
                                    stocks_df: Optional[pd.DataFrame] = None,
                                    sector_map: Optional[Dict] = None,
                                    info: Optional[Dict] = None,
                                    heatmap_df: Optional[pd.DataFrame] = None,
                                    north_flow: Optional[float] = None) -> Tuple[int, Dict[str, Any]]:
    """
    个股结构分(0-2) = 维度①个股部分(0-1) + 维度③④综合(0-1)
    返回 (分数, 明细字典)。
    """
    detail: Dict[str, Any] = {}
    score = 0

    # 维度①个股：拥挤出清
    d1_sc = dim1_individual_wyckoff_sc(symbol, wyckoff_events)
    d1_oi = dim1_individual_oi_pcr_fallback(symbol)
    d1_hit = any([d1_sc, d1_oi])
    if d1_hit:
        score += 1
    detail["维度①_个股"] = {
        "命中": d1_hit,
        "维科夫SC": d1_sc,
        "OI_PCR回落": d1_oi,
    }

    # 维度③④综合：估值跌透 + 资金动向
    d3_pe = dim3_pe_percentile(symbol, pe)
    d3_erp = dim3_erp_high(symbol, pe, treasury_yield)
    d3_sector = dim3_sector_relative_pe(symbol, sector_map, stocks_df)
    d3_peg = dim3_peg_growth(symbol, info)
    d3_hit = any([d3_pe, d3_erp, d3_sector, d3_peg])

    d4_disp, d4_disp_txt = dim4_heatmap_dispersion(heatmap_df)
    d4_north = dim4_north_flow_a_share(north_flow) if symbol.endswith((".SS", ".SZ")) else False
    d4_safe, d4_safe_txt = dim4_safe_haven_correlation(symbol)
    d4_hit = any([d4_disp, d4_north, d4_safe])

    d34_hit = d3_hit or d4_hit
    if d34_hit:
        score += 1

    detail["维度③_估值"] = {
        "命中": d3_hit,
        "PE历史低位": d3_pe,
        "ERP高位": d3_erp,
        "板块相对低位": d3_sector,
        "PEG<1": d3_peg,
    }
    detail["维度④_资金"] = {
        "命中": d4_hit,
        "热力图离散度高": d4_disp,
        "离散度描述": d4_disp_txt,
        "北向净流出": d4_north,
        "避险资产联动": d4_safe,
        "避险资产描述": d4_safe_txt,
    }

    return score, detail


def calc_bottom_confidence(symbol: str,
                           macro_score: Optional[int] = None,
                           macro_detail: Optional[Dict] = None,
                           **individual_kwargs) -> Dict[str, Any]:
    """
    计算单只股票的底部确信度。
    如果 macro_score 未传入，会自动计算（建议外层批量算一次复用）。
    """
    if macro_score is None:
        macro_score, macro_detail = calc_macro_env_score()
    else:
        macro_detail = macro_detail or {}

    ind_score, ind_detail = calc_individual_structure_score(symbol, **individual_kwargs)
    total = macro_score + ind_score
    emoji, label, color = traffic_light(total)

    return {
        "symbol": symbol,
        "bottom_score": total,
        "macro_score": macro_score,
        "individual_score": ind_score,
        "traffic_light": emoji,
        "label": label,
        "color": color,
        "macro_detail": macro_detail,
        "individual_detail": ind_detail,
        "asof": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# ---------------------------------------------------------------------------
# 批量计算 & 持久化（供每日 CI 调用）
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
    macro_score, macro_detail = calc_macro_env_score(policy_news=policy_news)
    results = {}
    for sym in symbols:
        pe = None
        if stocks_df is not None and not stocks_df.empty:
            row = stocks_df[stocks_df["symbol"] == sym]
            if not row.empty:
                pe = _safe_float(row.iloc[0].get("PE_Ratio"))
        r = calc_bottom_confidence(
            sym,
            macro_score=macro_score,
            macro_detail=macro_detail,
            pe=pe,
            treasury_yield=treasury_yield,
            stocks_df=stocks_df,
        )
        results[sym] = r

    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(BOTTOM_SCORE_PATH, "w", encoding="utf-8") as f:
            json.dump({"generated_at": datetime.now().isoformat(), "scores": results},
                      f, ensure_ascii=False, indent=2)
    return results


def load_bottom_scores() -> Dict[str, Dict[str, Any]]:
    """加载已保存的底部确信度。"""
    if BOTTOM_SCORE_PATH.exists():
        try:
            with open(BOTTOM_SCORE_PATH, "r", encoding="utf-8") as f:
                return json.load(f).get("scores", {})
        except Exception:
            pass
    return {}
