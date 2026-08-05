"""
screener.py — 选股三层架构（Stock Screening Pipeline）

第一层 Universe Screening（纯计算，无 LLM）：
  维科夫量价事件序列客观判定吸筹阶段。
  识别 7 类事件：PS / SC / AR / ST / Spring / SOS / LPS，
  按「已识别事件数 / 7」输出吸筹置信度与阶段标签。

第二层 Multi-factor Scoring（纯计算，无 LLM）：
  六因子加权评分：
    - POC（成交量加权最密集价位偏离）
    - 量能（近期量 / 20日均量）
    - 板块联动（与板块指数相关性）
    - 龙头强度（相对板块龙头表现）
    - 相对强度（RS，个股 vs 基准指数）
    - 同板块共振（板块内同涨比例）
  输出 0-100 综合分 + 60 分阈值判断。

第三层叙事生成（LLM，仅基于前两层结构化数据）：
  把第一/二层的结构化结果打包成 Prompt，由 LLM 做解释与综合研判，
  LLM 不做自主判断、不引入未提供的假设。

设计原则：
  - 第一/二层纯 pandas 计算，任何环境可跑、可回测
  - 所有函数对空数据 / 短数据免疫
  - 与 utils.py 的 recommend_stocks 互补：本模块重"吸筹结构"，recommend 重"技术/估值"
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("screener")

# 维科夫事件定义（用于解释输出）
WYCKOFF_EVENTS = {
    "PS": "初步支撑（Preliminary Support）：下跌趋势中首次放量止跌",
    "SC": "卖出高潮（Selling Climax）：恐慌性抛售，巨量长下影",
    "AR": "自动反弹（Automatic Rally）：SC 后快速反弹，量能放大",
    "ST": "二次测试（Secondary Test）：回踩前低，量能显著缩小",
    "Spring": "弹簧效应（Spring）：跌破 ST 低点后快速收回（假突破）",
    "SOS": "强势信号（Sign of Strength）：放量长阳突破盘整上沿",
    "LPS": "最后支撑点（Last Point of Support）：SOS 后缩量回踩不破",
}


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def detect_wyckoff_events(df: pd.DataFrame) -> Dict[str, Any]:
    """
    第一层：维科夫量价事件序列检测（纯计算）。

    输入 df 需含列：Open/High/Low/Close/Volume（按时间升序）。
    返回：
      {
        "ok": bool, "error": str|None,
        "events": [{"event": "SC", "date": "2026-07-15", "desc": "..."}, ...],  # 按时间序
        "event_count": int,
        "confidence": float,        # 已识别事件数 / 7，0-1
        "stage": str,               # 吸筹阶段标签
        "phase": str,               # 趋势阶段：下跌/吸筹/突破/拉升/未知
        "summary": str,             # 一句话结论
      }
    """
    base: Dict[str, Any] = {"ok": False, "events": [], "event_count": 0, "confidence": 0.0,
                            "stage": "数据不足", "phase": "未知", "summary": "", "error": None}
    if df is None or df.empty or len(df) < 120:
        base["error"] = "K线数据不足120根，无法检测维科夫结构"
        return base
    need = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in need):
        base["error"] = f"缺少列: {[c for c in need if c not in df.columns]}"
        return base
    d = df[need].copy()
    d = d.dropna(subset=["Close"])
    if len(d) < 120:
        base["error"] = "清洗后数据不足120根"
        return base

    close = d["Close"]
    vol = d["Volume"].astype(float)
    vol_ma20 = vol.rolling(20).mean()
    # 近期 90 日趋势
    lookback = min(90, len(d) - 1)
    seg = d.iloc[-lookback:]
    seg_close = seg["Close"]
    low_90 = seg_close.min()
    high_90 = seg_close.max()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean() if len(close) >= 200 else None
    events: List[Dict[str, Any]] = []
    found: set = set()

    def add(ev: str, i: int):
        if ev in found:
            return
        found.add(ev)
        events.append({"event": ev, "date": str(d.index[i].date()) if hasattr(d.index[i], "date") else str(d.index[i]),
                       "desc": WYCKOFF_EVENTS[ev]})

    # ---- 事件扫描（在近 90 日窗口内） ----
    n = len(d)
    for i in range(max(1, n - lookback), n):
        v = vol.iloc[i]
        vm = vol_ma20.iloc[i]
        if pd.isna(vm) or vm <= 0:
            continue
        v_ratio = v / vm
        body = abs(d["Close"].iloc[i] - d["Open"].iloc[i])
        rng = d["High"].iloc[i] - d["Low"].iloc[i]
        lower_shadow = min(d["Open"].iloc[i], d["Close"].iloc[i]) - d["Low"].iloc[i]
        upper_shadow = d["High"].iloc[i] - max(d["Open"].iloc[i], d["Close"].iloc[i])
        chg = d["Close"].iloc[i] / d["Close"].iloc[i - 1] - 1
        prev_close = d["Close"].iloc[i - 1]

        # SC：巨量 + 大阴或长下影 + 当日接近窗口低点
        if "SC" not in found and v_ratio >= 1.8 and rng > 0 and lower_shadow >= body * 1.2 and chg < -0.02:
            add("SC", i)
        # PS：SC 前出现（或独立）：下跌中首次放量止跌
        if "PS" not in found and v_ratio >= 1.4 and chg >= -0.01 and seg_close.iloc[-1] < close.iloc[i] * 1.05:
            add("PS", i)
        # AR：SC 后快速反弹
        if "SC" in found and "AR" not in found:
            if v_ratio >= 1.2 and chg > 0.02 and d["Close"].iloc[i] > d["Close"].iloc[i - 1]:
                add("AR", i)
        # ST：回踩前低且量缩
        if "SC" in found and "AR" in found and "ST" not in found:
            near_low = d["Low"].iloc[i] <= d["Low"].iloc[:i].min() * 1.03
            if near_low and v_ratio <= 0.8:
                add("ST", i)
        # Spring：跌破 ST 低点后快速收回
        if "ST" in found and "Spring" not in found:
            st_low = d["Low"].iloc[:i].min()
            broke = d["Low"].iloc[i] < st_low * 0.995
            reclaimed = d["Close"].iloc[i] > st_low
            if broke and reclaimed and lower_shadow >= body * 1.5:
                add("Spring", i)
        # SOS：放量长阳突破盘整上沿
        if "Spring" in found and "SOS" not in found:
            box_high = d["High"].iloc[max(0, i - 60):i].max()
            if v_ratio >= 1.5 and chg > 0.03 and d["Close"].iloc[i] > box_high:
                add("SOS", i)
        # LPS：SOS 后缩量回踩不破
        if "SOS" in found and "LPS" not in found:
            sos_high = d["High"].iloc[max(0, i - 40):i].max()
            if v_ratio <= 0.9 and d["Close"].iloc[i] > d["Close"].iloc[i - 1] * 0.98 and d["Low"].iloc[i] > sos_high * 0.97:
                add("LPS", i)

    # ---- 阶段判定 ----
    order = ["PS", "SC", "AR", "ST", "Spring", "SOS", "LPS"]
    confidence = len(found) / len(order)
    events_sorted = sorted(events, key=lambda e: e["date"])
    # 阶段标签
    if "Spring" in found and "SOS" in found:
        stage = "拉升突破（SOS 已确认）"
    elif "Spring" in found:
        stage = "吸筹后期（Spring 出现，关注 SOS）"
    elif "ST" in found:
        stage = "吸筹中期（二次测试，等待 Spring/SOS）"
    elif "SC" in found:
        stage = "吸筹早期（卖出高潮后，等待 AR/ST）"
    elif "PS" in found:
        stage = "初步支撑（下跌末端迹象）"
    else:
        stage = "无明确吸筹结构"
    # 趋势阶段
    last = close.iloc[-1]
    if ma50 is not None and not pd.isna(ma50.iloc[-1]) and last > ma50.iloc[-1] * 1.02:
        phase = "拉升"
    elif ma50 is not None and not pd.isna(ma50.iloc[-1]) and last < ma50.iloc[-1] * 0.98:
        phase = "下跌"
    else:
        phase = "盘整"
    # 一句话结论
    if confidence >= 0.71:
        summary = f"高置信吸筹结构（{len(found)}/7 事件确认），当前阶段：{stage}；建议关注 SOS 突破后的介入机会。"
    elif confidence >= 0.43:
        summary = f"中置信吸筹结构（{len(found)}/7 事件确认），当前阶段：{stage}；等待后续事件确认。"
    elif confidence >= 0.14:
        summary = f"弱吸筹迹象（{len(found)}/7 事件），阶段：{stage}；趋势尚未扭转，保持观察。"
    else:
        summary = f"未检测到维科夫吸筹结构（{len(found)}/7 事件），阶段：{phase}。"
    base.update(ok=True, events=events_sorted, event_count=len(found),
                confidence=round(confidence, 2), stage=stage, phase=phase, summary=summary)
    return base


def explain_wyckoff(w: Dict[str, Any]) -> Dict[str, Any]:
    """
    把维科夫检测结果翻译成「人话」解释（需求6：内嵌 AI 分析指引）。
    纯基于 detect_wyckoff_events 的结构化输出，不编造价格/消息。
    返回：
      {
        "confidence_meaning": str,   # 置信度数字代表什么
        "sequence_meaning": str,     # 事件序列的语义解释
        "stage_meaning": str,        # 当前阶段含义
        "guidance": str,             # 结合智能荐股/多因子的操作指引
      }
    """
    if not w or not w.get("ok"):
        return {
            "confidence_meaning": "样本不足，无法判定吸筹结构。",
            "sequence_meaning": "—",
            "stage_meaning": "—",
            "guidance": "需至少 120 根日K 才能进入第一层扫描；建议先补充历史数据。",
        }
    conf = w.get("confidence", 0)
    found = [e["event"] for e in w.get("events", [])]
    # 置信度含义
    if conf >= 0.71:
        cm = f"置信度 {conf:.0%}（{len(found)}/7 事件确认）——属于高置信吸筹结构，量价已经走完大部分经典序列。"
    elif conf >= 0.43:
        cm = f"置信度 {conf:.0%}（{len(found)}/7 事件确认）——中置信，已出现核心事件但尚未闭环，需后续事件确认。"
    elif conf >= 0.14:
        cm = f"置信度 {conf:.0%}（{len(found)}/7 事件）——仅弱迹象，趋势尚未扭转，仅作观察。"
    else:
        cm = f"置信度 {conf:.0%}（{len(found)}/7 事件）——未形成吸筹结构，当前更多是 {w.get('phase','未知')} 状态。"

    # 事件序列语义
    seq = " → ".join(found) if found else "（无事件）"
    seq_meaning = (
        f"已识别事件序列：{seq}。\n"
        "语义：下跌末端先有【初步支撑 PS】，恐慌抛售形成【卖出高潮 SC】，"
        "随后【自动反弹 AR】与【二次测试 ST】确认底部区间；若出现【弹簧 Spring】（假跌破后收回）"
        "往往是最佳陷阱破位，接着【强势信号 SOS】放量突破、回踩【最后支撑点 LPS】不破，"
        "吸筹即告完成、进入拉升。事件越靠后、越完整，确定性越高。"
    )

    # 阶段含义
    stage_meaning = (
        f"当前阶段：{w.get('stage','—')}；趋势阶段：{w.get('phase','—')}。"
        "春/弹簧(Spring)与强势信号(SOS)是两条关键确认线——"
        "没有 Spring 的突破易假，没有 SOS 的吸筹未闭环。"
    )

    # 结合智能荐股/多因子的指引
    guidance = (
        "【结合智能荐股的操作指引】\n"
        "① 本扫描为选股三层架构第一层（纯量价结构），只回答『是否在吸筹』，不回答『值不值得买』。\n"
        "② 高置信(≥71%)且处于 SOS/LPS 阶段：进入第二层多因子评分（POC/量能/板块联动/龙头/相对强度），"
        "综合分 ≥60 才视为技术面共振；再用 recommend_stocks 的 PE 行业归一化与买入价位做估值校验。\n"
        "③ 中/低置信：仅作自选观察，不急于介入；等待 Spring→SOS 闭环再加第二层过滤。\n"
        "④ 任何介入都须套用 R 倍数分批止盈（见个股深度分析·交易策略），避免一次性追高卖飞。"
    )
    return {
        "confidence_meaning": cm,
        "sequence_meaning": seq_meaning,
        "stage_meaning": stage_meaning,
        "guidance": guidance,
    }


# ---------------------------------------------------------------------------
# 第二层：多因子评分
# ---------------------------------------------------------------------------
def score_multi_factor(
    df: pd.DataFrame,
    sector_df: Optional[pd.DataFrame] = None,
    sector_leader: Optional[float] = None,
    bench_df: Optional[pd.DataFrame] = None,
    sector_syms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    六因子加权评分（0-100）：
      tech  = 均线结构（MA20>MA50 且价>MA20）
      poc   = 成交量加权密集价位（POC）偏离
      vol   = 量能（近5日均量 / 20日均量）
      sec   = 板块联动（与 sector_df 收益率相关性）
      lead  = 龙头强度（相对 sector_leader 的 20 日表现）
      rs    = 相对强度（相对 bench_df 的 60 日表现）
    综合 = 0.25*tech + 0.15*poc + 0.15*vol + 0.15*sec + 0.15*lead + 0.15*rs
    返回 {ok, score, bias, threshold_pass, factors:{...}, detail}
    """
    base: Dict[str, Any] = {"ok": False, "score": 0, "bias": "数据不足", "threshold_pass": False,
                            "factors": {}, "detail": "", "error": None}
    if df is None or df.empty or len(df) < 30:
        base["error"] = "个股 K 线不足 30 根"
        return base
    need = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in need):
        base["error"] = "K线缺少必要列"
        return base
    d = df[need].dropna(subset=["Close"]).copy()
    if len(d) < 30:
        base["error"] = "清洗后不足 30 根"
        return base
    close = d["Close"]
    vol = d["Volume"].astype(float)
    last = float(close.iloc[-1])
    ma5 = float(close.rolling(5).mean().iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else ma20

    # tech：均线结构
    tech = 50.0
    if last > ma20:
        tech += 25
    if ma20 > ma50:
        tech += 15
    if last > ma5:
        tech += 10
    tech = min(100.0, max(0.0, tech))

    # POC：最近 20 日成交量加权均价，价在其上则偏多
    w = d.tail(20)
    poc = float((w["Close"] * w["Volume"]).sum() / max(w["Volume"].sum(), 1))
    poc_dev = (last - poc) / poc * 100 if poc > 0 else 0
    poc_score = min(100.0, max(0.0, 50 + poc_dev * 3))

    # vol：量能扩张度
    v5 = float(vol.tail(5).mean())
    v20 = float(vol.tail(20).mean())
    vol_score = min(100.0, max(0.0, 50 + (v5 / max(v20, 1) - 1) * 60)) if v20 > 0 else 50.0

    # sec：板块联动（收益率相关性）
    sec_score = 50.0
    if sector_df is not None and not sector_df.empty:
        try:
            a = close.pct_change().dropna()
            b = sector_df["Close"].pct_change().dropna() if "Close" in sector_df.columns else None
            if b is not None:
                a, b = a.align(b, join="inner")
                if len(a) > 10:
                    corr = float(a.corr(b))
                    sec_score = min(100.0, max(0.0, 50 + corr * 35))
        except Exception as e:  # noqa: BLE001
            logger.debug("板块联动计算失败: %s", e)

    # lead：龙头强度（相对板块龙头 20 日超额）
    lead_score = 50.0
    if sector_leader is not None:
        try:
            lead_ret = close.iloc[-1] / close.iloc[-21] - 1
            lead_diff = (lead_ret - sector_leader) * 100
            lead_score = min(100.0, max(0.0, 50 + lead_diff * 3))
        except Exception:  # noqa: BLE001
            pass

    # rs：相对强度（相对基准 60 日超额）
    rs_score = 50.0
    if bench_df is not None and not bench_df.empty and len(close) >= 61:
        try:
            bench = bench_df["Close"].dropna()
            my_ret = close.iloc[-1] / close.iloc[-61] - 1
            bench_ret = bench.iloc[-1] / bench.iloc[-61] - 1
            rs_score = min(100.0, max(0.0, 50 + (my_ret - bench_ret) * 150))
        except Exception:  # noqa: BLE001
            pass

    score = round(0.25 * tech + 0.15 * poc_score + 0.15 * vol_score + 0.15 * sec_score + 0.15 * lead_score + 0.15 * rs_score, 1)
    bias = "看多" if score >= 60 else ("看空" if score <= 40 else "震荡")
    base.update(
        ok=True, score=score, bias=bias, threshold_pass=score >= 60,
        factors={
            "tech": round(tech), "poc": round(poc_score), "vol": round(vol_score),
            "sector": round(sec_score), "leader": round(lead_score), "rs": round(rs_score),
            "poc_price": round(poc, 2), "poc_dev_pct": round(poc_dev, 2),
        },
        detail=f"综合 {score} 分（60 分阈值{'通过' if score >= 60 else '未过'}）· {bias}",
    )
    return base


# ---------------------------------------------------------------------------
# 第三层：叙事生成（LLM 仅解释结构化数据）
# ---------------------------------------------------------------------------
def build_screener_narrative_prompt(wyckoff: Dict[str, Any], factors: Dict[str, Any], symbol: str, name: str = "") -> str:
    """
    把第一/二层结构化结果打包为 Prompt。规则：LLM 只能解释输入，
    不能自行引入价格/新闻/预测等未提供信息。
    """
    ev_lines = "\n".join([f"  - {e['date']} {e['event']}: {e['desc']}" for e in wyckoff.get("events", [])]) or "  （无）"
    f = factors.get("factors", {})
    return f"""你是量化研究助理。请仅基于以下【结构化事实】为 {name or symbol} 写一段 150 字以内的投资研判，分三句：
1) 维科夫吸筹结构判断；2) 多因子强弱拆解；3) 综合结论与关注点。

【规则】只能解释下列数据，不得编造价格、消息、目标价或任何未提供的信息；若数据不足，如实说明。

—— 第一层·维科夫事件序列（置信度 {wyckoff.get('confidence')}）——
阶段: {wyckoff.get('stage')}；趋势阶段: {wyckoff.get('phase')}
事件明细:
{ev_lines}
结论: {wyckoff.get('summary')}

—— 第二层·多因子评分（总分 {factors.get('score')}，{factors.get('bias')}，60 分阈值{'通过' if factors.get('threshold_pass') else '未通过'}）——
tech(均线)= {f.get('tech')}  poc(POC偏离 {f.get('poc_dev_pct')}%)= {f.get('poc')}  vol(量能)= {f.get('vol')}
sector(板块联动)= {f.get('sector')}  leader(龙头强度)= {f.get('leader')}  rs(相对强度)= {f.get('rs')}

请输出研判（中文，150 字内）。"""


def call_narrative_llm(prompt: str, llm_fn=None) -> str:
    """
    执行叙事生成。llm_fn(prompt)->str 由调用方注入（避免本模块依赖 openai/streamlit）。
    无 LLM 时返回空串，由上层降级展示结构化结论。
    """
    if llm_fn is None:
        return ""
    try:
        text = llm_fn(prompt)
        return str(text).strip()
    except Exception as e:  # noqa: BLE001
        logger.warning("叙事生成失败: %s", e)
        return ""
