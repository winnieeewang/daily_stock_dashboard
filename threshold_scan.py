"""
threshold_scan.py — 多因子评分阈值扫描校准（40-60分，步长5分）

对全量自选股做滚动回测，统计各阈值（≥T）下信号的：
  - 信号数量（样本量）
  - 20日胜率 / 平均收益率 / 盈亏比
  - 持有期最大回撤（信号后20日窗口）
统一参数与费用假设（等权、无费用、HOLD_DAYS=20），保证各阈值可比。

输出：data/threshold_scan_report.json
CLI: python threshold_scan.py [--years 3] [--step 5] [--top 43]
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
REPORT_PATH = DATA_DIR / "threshold_scan_report.json"

DEFAULT_YEARS = 3
DEFAULT_STEP = 5
SCORE_WINDOW = 250
HOLD_DAYS = 20
THRESHOLDS = [40, 45, 50, 55, 60]
MIN_SAMPLE = 30  # 低于该信号数标注统计显著性不足


def load_symbols(top_n: int = 43) -> List[str]:
    import pandas as pd
    p = DATA_DIR / "stocks.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    return df["symbol"].tolist()[:top_n]


def fetch_history(symbol: str, years: int = DEFAULT_YEARS):
    try:
        import yfinance as yf
        df = yf.download(symbol, period=f"{years}y", progress=False, auto_adjust=True)
        if df is None or df.empty or len(df) < SCORE_WINDOW:
            return None
        if isinstance(df.columns, __import__("pandas").MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception as e:  # noqa: BLE001
        logger.debug("%s 历史数据下载失败: %s", symbol, e)
        return None


def scan_symbol(symbol: str, df, step: int = DEFAULT_STEP) -> List[Dict[str, Any]]:
    """对单只标的滚动评分，返回每次评分的记录列表。"""
    import screener as S
    close = df["Close"]
    n = len(df)
    records = []
    i = SCORE_WINDOW
    while i <= n - HOLD_DAYS:
        window = df.iloc[i - SCORE_WINDOW:i]
        try:
            r = S.score_multi_factor(window)
        except Exception:  # noqa: BLE001
            r = {}
        score = r.get("score")
        if score is None:
            i += step
            continue
        entry = float(close.iloc[i])
        if entry <= 0:
            i += step
            continue
        # 未来20日收益序列（用于胜率/均收/回撤/盈亏比）
        future = close.iloc[i + 1:i + 1 + HOLD_DAYS]
        if len(future) < HOLD_DAYS:
            i += step
            continue
        rets = future.pct_change().fillna(0).cumsum().tolist()  # 近似累计收益序列
        ret_20 = (float(future.iloc[-1]) - entry) / entry * 100
        # 持有期最大回撤（用累计收益序列）
        peak = 0.0
        mdd = 0.0
        for r_ in rets:
            peak = max(peak, r_)
            mdd = min(mdd, r_ - peak)
        records.append({
            "symbol": symbol,
            "date": str(df.index[i].date()) if hasattr(df.index[i], "date") else str(df.index[i]),
            "score": score,
            "ret_20": round(ret_20, 2),
            "mdd_20": round(mdd * 100, 2),
        })
        i += step
    return records


def aggregate(records: List[Dict[str, Any]], threshold: int) -> Dict[str, Any]:
    """按阈值聚合：≥threshold 的所有信号。"""
    hits = [r for r in records if r["score"] >= threshold]
    n = len(hits)
    if n == 0:
        return {"threshold": threshold, "signals": 0, "sample_sufficient": False,
                "note": "无信号"}
    rets = [r["ret_20"] for r in hits]
    mdds = [r["mdd_20"] for r in hits]
    wins = sum(1 for x in rets if x > 0)
    losses = [x for x in rets if x < 0]
    gains = [x for x in rets if x > 0]
    avg_win = sum(gains) / len(gains) if gains else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    profit_loss_ratio = round(avg_win / avg_loss, 2) if avg_loss > 0 else (None if not gains else 999.0)
    return {
        "threshold": threshold,
        "signals": n,
        "sample_sufficient": n >= MIN_SAMPLE,
        "win_rate": round(wins / n * 100, 1),
        "avg_return": round(sum(rets) / n, 2),
        "avg_mdd": round(sum(mdds) / n, 2),
        "profit_loss_ratio": profit_loss_ratio,
        "best_return": round(max(rets), 2),
        "worst_return": round(min(rets), 2),
        "median_return": round(sorted(rets)[n // 2], 2),
    }


def run_scan(years: int = DEFAULT_YEARS, step: int = DEFAULT_STEP,
             top_n: int = 43, thresholds: Optional[List[int]] = None) -> Dict[str, Any]:
    symbols = load_symbols(top_n)
    if not symbols:
        return {"error": "无自选股数据（先运行 stock_dashboard.py）", "by_threshold": {}}
    th = thresholds or THRESHOLDS

    all_records: List[Dict[str, Any]] = []
    per_symbol: Dict[str, Dict[str, Any]] = {}
    for sym in symbols:
        logger.info("扫描 %s ...", sym)
        df = fetch_history(sym, years)
        if df is None:
            per_symbol[sym] = {"error": "历史数据不足"}
            continue
        recs = scan_symbol(sym, df, step)
        all_records.extend(recs)
        per_symbol[sym] = {"scored_signals": len(recs),
                           "avg_score": round(sum(r["score"] for r in recs) / len(recs), 1) if recs else None,
                           "gte60": sum(1 for r in recs if r["score"] >= 60)}

    by_threshold = {str(t): aggregate(all_records, t) for t in th}

    # 每只标的在各阈值下的表现（供前端钻取）
    per_symbol_threshold: Dict[str, Dict[str, Any]] = {}
    for sym, recs in [(s, [r for r in all_records if r["symbol"] == s]) for s in per_symbol if "error" not in per_symbol[s]]:
        per_symbol_threshold[sym] = {str(t): aggregate(recs, t) for t in th}

    # 推荐阈值：综合 胜率 + 均收 + 盈亏比（样本充足前提下）
    def _score_threshold(t: int) -> Optional[float]:
        a = by_threshold.get(str(t), {})
        if not a.get("sample_sufficient") or a.get("signals", 0) == 0:
            return None
        # 归一化打分：胜率(0-1)*40 + 均收(每1%计5分,封顶30) + 盈亏比(每0.5计10分,封顶30)
        s = (a["win_rate"] / 100) * 40
        s += min(max(a["avg_return"], 0), 6) * 5
        plr = a.get("profit_loss_ratio") or 0
        s += min(plr / 0.5, 3) * 10
        return round(s, 1)

    ranked = [(t, _score_threshold(t)) for t in th]
    ranked = [(t, s) for t, s in ranked if s is not None]
    ranked.sort(key=lambda x: -x[1])
    recommendation = {
        "best_threshold": ranked[0][0] if ranked else None,
        "ranking": ranked,
        "note": (f"推荐阈值 {ranked[0][0]}：样本充足且综合胜率/收益/盈亏比最优。"
                 if ranked else "所有阈值样本量均不足，无法给出可靠推荐"),
    }

    report = {
        "generated_at": datetime.now().isoformat(),
        "params": {"years": years, "step": step, "hold_days": HOLD_DAYS,
                   "score_window": SCORE_WINDOW, "thresholds": th,
                   "cost_assumption": "无费用/等权", "symbols_n": len(symbols)},
        "by_threshold": by_threshold,
        "per_symbol": per_symbol,
        "per_symbol_threshold": per_symbol_threshold,
        "recommendation": recommendation,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("✅ 阈值扫描报告已保存: %s", REPORT_PATH)
    return report


def load_report() -> Dict[str, Any]:
    if REPORT_PATH.exists():
        try:
            return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="多因子评分阈值扫描校准")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument("--top", type=int, default=43)
    parser.add_argument("--thresholds", type=int, nargs="*", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_scan(years=args.years, step=args.step, top_n=args.top,
             thresholds=args.thresholds)
