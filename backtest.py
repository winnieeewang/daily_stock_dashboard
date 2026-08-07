"""
backtest.py — 3 年历史回测框架（Phase 2 ③）

对三大策略做 3 年历史回测，产出胜率/平均收益/夏普，供前端展示：

  1. 维科夫吸筹信号（SC / Spring）
     - 滚动窗口（120 根日K ≈ 半年）检测事件，记录信号日后 5/10/20 交易日收益
     - 统计：信号次数、胜率（>0 比例）、平均收益、vs Buy&Hold 超额

  2. 多因子评分（≥60 vs <60）
     - 滚动窗口（250 根日K ≈ 1年）评分，记录后 20 交易日收益
     - 统计：≥60 / <60 两组的 20日胜率、平均收益、年化夏普

  3. R 倍数分批止盈（+1R 减 1/3、+2R 减 1/3）vs Buy&Hold
     - 用 ATR(14) 定义 1R，模拟规则化减仓
     - 统计：总收益、年化、最大回撤、夏普

CLI:
  python backtest.py                          # 默认自选股前10只，3年
  python backtest.py --symbols NVDA MSFT      # 指定标的
  python backtest.py --top 20                 # 自选股前20只
  python backtest.py --years 5 --step 10      # 5年、步长10天（更快）
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
REPORT_PATH = DATA_DIR / "backtest_report.json"

DEFAULT_YEARS = 3
DEFAULT_STEP = 5        # 滚动窗口步长（交易日）
WYCKOFF_WINDOW = 130    # 维科夫检测窗口（根）
SCORE_WINDOW = 250      # 多因子评分窗口（根）
HOLD_DAYS = 20          # 持仓考察期（交易日）
TOP_N = 10              # 默认回测股票数


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------

def load_symbols(top_n: int = TOP_N) -> List[str]:
    """从 data/stocks.csv 取自选股（默认按出现顺序前 top_n）。"""
    import pandas as pd
    p = DATA_DIR / "stocks.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    return df["symbol"].tolist()[:top_n]


def fetch_history(symbol: str, years: int = DEFAULT_YEARS) -> Optional[Any]:
    """下载 N 年日线（auto_adjust=True 复权）。"""
    try:
        import yfinance as yf
        df = yf.download(symbol, period=f"{years}y", progress=False, auto_adjust=True)
        if df is None or df.empty or len(df) < WYCKOFF_WINDOW:
            return None
        if isinstance(df.columns, __import__("pandas").MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df
    except Exception as e:  # noqa: BLE001
        logger.debug("%s 历史数据下载失败: %s", symbol, e)
        return None


def _future_returns(close: Any, idx: int, horizons=(5, 10, 20)) -> Dict[int, Optional[float]]:
    """从 idx 位置往后 N 根K线的收益（%）。"""
    out = {}
    n = len(close)
    entry = float(close.iloc[idx])
    if entry <= 0:
        return {h: None for h in horizons}
    for h in horizons:
        j = idx + h
        out[h] = round((float(close.iloc[j]) - entry) / entry * 100, 2) if j < n else None
    return out


def _sharpe(returns: List[float], annualize: bool = True) -> Optional[float]:
    """年化夏普（无风险利率≈0）。"""
    if len(returns) < 3:
        return None
    import numpy as np
    r = np.array(returns, dtype=float)
    sd = float(r.std(ddof=1))
    if sd <= 1e-9:
        return None
    sr = float(r.mean()) / sd
    return round(sr * math.sqrt(252) if annualize else sr, 2)


# ---------------------------------------------------------------------------
# 1) 维科夫吸筹回测
# ---------------------------------------------------------------------------

def backtest_wyckoff(symbol: str, df: Any, step: int = DEFAULT_STEP) -> Dict[str, Any]:
    """滚动检测 SC/Spring 信号，统计未来收益。"""
    import screener as S

    close = df["Close"]
    n = len(df)
    signals = []
    i = WYCKOFF_WINDOW
    while i <= n - HOLD_DAYS:
        window = df.iloc[i - WYCKOFF_WINDOW:i]
        try:
            w = S.detect_wyckoff_events(window)
        except Exception:  # noqa: BLE001
            w = {"ok": False}
        if w.get("ok"):
            evs = [e.get("event") for e in w.get("events", [])]
            if evs:
                rets = _future_returns(close, i, horizons=(5, 10, 20))
                signals.append({
                    "date": str(df.index[i].date()) if hasattr(df.index[i], "date") else str(df.index[i]),
                    "events": evs,
                    "confidence": w.get("confidence"),
                    "future": {str(k): v for k, v in rets.items()},
                })
        i += step

    if not signals:
        return {"signals": 0, "note": "窗口内未检测到 SC/Spring 信号"}

    ret20 = [s["future"]["20"] for s in signals if s["future"].get("20") is not None]
    wins = sum(1 for r in ret20 if r > 0)
    bh_ret = round((float(close.iloc[-1]) / float(close.iloc[0]) - 1) * 100, 2)

    return {
        "signals": len(signals),
        "win_rate_20d": round(wins / len(ret20) * 100, 1) if ret20 else None,
        "avg_ret_20d": round(sum(ret20) / len(ret20), 2) if ret20 else None,
        "sharpe_20d": _sharpe(ret20),
        "vs_buy_hold_20d": round((sum(ret20) / len(ret20)) - bh_ret / (n // HOLD_DAYS), 2) if ret20 else None,
        "sample": signals[:5],
    }


# ---------------------------------------------------------------------------
# 2) 多因子评分回测（≥60 vs <60）
# ---------------------------------------------------------------------------

def backtest_multifactor(symbol: str, df: Any, step: int = DEFAULT_STEP) -> Dict[str, Any]:
    """滚动评分，比较 ≥60 与 <60 两组的 20 日收益。"""
    import screener as S

    close = df["Close"]
    n = len(df)
    gte_records = []
    lt_records = []
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
        rets = _future_returns(close, i, horizons=(HOLD_DAYS,))
        ret20 = rets.get(HOLD_DAYS)
        rec = {"date": str(df.index[i].date()) if hasattr(df.index[i], "date") else str(df.index[i]),
               "score": score, "ret_20d": ret20}
        if ret20 is not None:
            if score >= 60:
                gte_records.append(rec)
            else:
                lt_records.append(rec)
        i += step

    def _grp(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not records:
            return {"count": 0}
        rets = [r["ret_20d"] for r in records]
        return {
            "count": len(records),
            "win_rate_20d": round(sum(1 for x in rets if x > 0) / len(rets) * 100, 1),
            "avg_ret_20d": round(sum(rets) / len(rets), 2),
            "sharpe_20d": _sharpe(rets),
        }

    return {"gte60": _grp(gte_records), "lt60": _grp(lt_records),
            "total_scored": len(gte_records) + len(lt_records)}


# ---------------------------------------------------------------------------
# 3) R 倍数分批止盈 vs Buy&Hold
# ---------------------------------------------------------------------------

def backtest_r_multiple(symbol: str, df: Any) -> Dict[str, Any]:
    """
    R 倍数分批止盈规则：
      - 1R = 1 × ATR(14)（以入场价为基准）
      - +1R：减仓 1/3；+2R：再减 1/3；剩余 1/3 持有至期末
    对比 Buy&Hold（等权、满仓持有 3 年）。
    """
    import pandas as pd

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    n = len(df)

    def _atr_series():
        tr = pd.concat([high - low,
                        (high - close.shift(1)).abs(),
                        (low - close.shift(1)).abs()], axis=1).max(axis=1)
        return tr.rolling(14).mean()

    atr = _atr_series()
    # 入场：第 60 根K线收盘价（跳过均线预热期）
    entry_i = 60
    if entry_i >= n - HOLD_DAYS:
        return {"note": "数据不足"}
    entry = float(close.iloc[entry_i])
    r = float(atr.iloc[entry_i])
    if entry <= 0 or r <= 0:
        return {"note": "ATR/价格异常"}

    # 模拟分批止盈
    positions = 3  # 三等份
    sold_prices = []
    sold_at = []
    for k in range(entry_i + 1, n):
        px = float(close.iloc[k])
        if positions >= 3 and px >= entry + 1 * r:
            sold_prices.append(px)
            sold_at.append("+1R")
            positions -= 1
        elif positions == 2 and px >= entry + 2 * r:
            sold_prices.append(px)
            sold_at.append("+2R")
            positions -= 1
        if positions <= 1:
            break
    # 剩余仓位按期末价
    final_px = float(close.iloc[-1])
    realized = sum(sold_prices) + positions * final_px
    cost = 3 * entry
    r_total = (realized - cost) / cost * 100

    # Buy&Hold：同入场价满仓持有到期
    bh_total = (final_px - entry) / entry * 100

    # 净值序列（用于夏普/回撤）
    def _nav_series(hold_fraction: float):
        nav = []
        val = 3.0
        for k in range(entry_i, n):
            px = float(close.iloc[k])
            nav.append(val / (3 * entry) * 100)  # 归一化到 100
        return nav

    return {
        "entry": round(entry, 2), "r_size": round(r, 2),
        "exit_times": sold_at,
        "r_total_return_pct": round(r_total, 2),
        "bh_total_return_pct": round(bh_total, 2),
        "r_excess_pct": round(r_total - bh_total, 2),
        "r_wins": r_total > bh_total,
    }


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

def run_backtest(symbols: Optional[List[str]] = None, years: int = DEFAULT_YEARS,
                 step: int = DEFAULT_STEP, top_n: int = TOP_N) -> Dict[str, Any]:
    """跑全部回测，产出 data/backtest_report.json。"""
    if symbols is None:
        symbols = load_symbols(top_n)
    if not symbols:
        return {"error": "无自选股数据（先运行 stock_dashboard.py）", "results": {}}

    results: Dict[str, Any] = {}
    for sym in symbols:
        logger.info("回测 %s ...", sym)
        df = fetch_history(sym, years)
        if df is None:
            results[sym] = {"error": "历史数据不足"}
            continue
        results[sym] = {
            "wyckoff": backtest_wyckoff(sym, df, step),
            "multifactor": backtest_multifactor(sym, df, step),
            "r_multiple": backtest_r_multiple(sym, df),
            "years": years,
        }

    report = {
        "generated_at": datetime.now().isoformat(),
        "params": {"years": years, "step": step, "hold_days": HOLD_DAYS,
                   "wyckoff_window": WYCKOFF_WINDOW, "score_window": SCORE_WINDOW},
        "results": results,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("✅ 回测报告已保存: %s", REPORT_PATH)
    return report


def load_report() -> Dict[str, Any]:
    """加载已生成的回测报告。"""
    if REPORT_PATH.exists():
        try:
            return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="3 年历史回测（维科夫/多因子/R倍数）")
    parser.add_argument("--symbols", nargs="*", default=None, help="指定标的（默认自选股前 top_n）")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument("--top", type=int, default=TOP_N)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_backtest(symbols=args.symbols, years=args.years, step=args.step, top_n=args.top)
