"""
backtest_threshold.py — 回测验证智能荐股 60 分阈值

读取 AI 炒手的 trades.jsonl，对每笔买入交易：
  1. 回溯当时的多因子评分（候选池来源已有评分；模型自选事后补算）
  2. 计算买入后 N 个交易日（默认 20 日）的收益率
  3. 统计：评分 ≥60  vs  <60 的平均收益率、胜率、最大回撤

产出：data/threshold_validation_report.json
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
AI_TRADERS_DIR = DATA_DIR / "ai_traders"
REPORT_PATH = DATA_DIR / "threshold_validation_report.json"

HOLD_DAYS = 20  # 默认持有 20 个交易日
THRESHOLD = 60  # 智能荐股阈值


# ---------------------------------------------------------------------------
# 1. 读取交易记录
# ---------------------------------------------------------------------------

def load_trades(model_id: str) -> List[Dict[str, Any]]:
    """读取指定模型的 trades.jsonl，只返回 BUY 交易。"""
    p = AI_TRADERS_DIR / model_id / "trades.jsonl"
    trades: List[Dict[str, Any]] = []
    if not p.exists():
        return trades
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
                if t.get("action", "").upper() == "BUY":
                    trades.append(t)
            except Exception:
                continue
    return trades


# ---------------------------------------------------------------------------
# 2. 事后补算多因子评分（模型自选时）
# ---------------------------------------------------------------------------

def _fetch_score_at_date(symbol: str, date_str: str) -> Optional[int]:
    """
    事后补算某股票在某日期的多因子评分。
    策略：用该日期前 3 个月的日线数据跑 screener.score_multi_factor。
    """
    try:
        import yfinance as yf
        import screener as S
        # 下载日期前 3 个月的数据（确保有足够历史）
        end = datetime.strptime(date_str, "%Y-%m-%d")
        start = end - timedelta(days=100)
        df = yf.download(
            symbol, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
            progress=False, auto_adjust=False,
        )
        if df is None or df.empty or len(df) < 30:
            return None
        res = S.score_multi_factor(df)
        return res.get("score")
    except Exception as e:  # noqa: BLE001
        logger.debug("补算评分失败 %s@%s: %s", symbol, date_str, e)
        return None


# ---------------------------------------------------------------------------
# 3. 计算持有期收益率
# ---------------------------------------------------------------------------

def _holding_return(symbol: str, buy_date: str, hold_days: int = HOLD_DAYS) -> Optional[Dict[str, Any]]:
    """
    计算买入后 hold_days 个交易日的收益率。
    返回 {"return_pct": float, "max_drawdown_pct": float, "exit_price": float, "bars": int} 或 None。
    """
    try:
        import yfinance as yf
        buy_dt = datetime.strptime(buy_date, "%Y-%m-%d")
        # 多下一些数据确保覆盖 hold_days 个交易日
        end = buy_dt + timedelta(days=hold_days + 10)
        df = yf.download(
            symbol, start=buy_date, end=end.strftime("%Y-%m-%d"),
            progress=False, auto_adjust=False,
        )
        if df is None or df.empty:
            return None
        # 取收盘价
        closes = df["Close"].dropna()
        if len(closes) < 2:
            return None
        entry = float(closes.iloc[0])
        # 取第 hold_days 个交易日的收盘价（或最后一个可用）
        exit_idx = min(hold_days, len(closes) - 1)
        exit_price = float(closes.iloc[exit_idx])
        ret = round((exit_price - entry) / entry * 100, 2)
        # 最大回撤
        cummax = closes.iloc[:exit_idx + 1].cummax()
        dd = ((closes.iloc[:exit_idx + 1] - cummax) / cummax).min()
        return {
            "return_pct": ret,
            "max_drawdown_pct": round(float(dd) * 100, 2),
            "exit_price": round(exit_price, 2),
            "bars": exit_idx,
        }
    except Exception as e:  # noqa: BLE001
        logger.debug("持有期收益计算失败 %s@%s: %s", symbol, buy_date, e)
        return None


# ---------------------------------------------------------------------------
# 4. 核心回测逻辑
# ---------------------------------------------------------------------------

def backtest_model(model_id: str, hold_days: int = HOLD_DAYS) -> Dict[str, Any]:
    """
    对单个模型的 BUY 交易做回测。
    返回 {
        "model_id": str,
        "total_trades": int,
        "scored_trades": int,
        "gte60": {"count": int, "avg_return": float, "win_rate": float, "avg_mdd": float, "trades": [...]},
        "lt60":  {"count": int, "avg_return": float, "win_rate": float, "avg_mdd": float, "trades": [...]},
    }
    """
    trades = load_trades(model_id)
    gte60_records: List[Dict[str, Any]] = []
    lt60_records: List[Dict[str, Any]] = []

    for t in trades:
        sym = t.get("symbol", "")
        date = t.get("date", "")
        source = t.get("source", "")
        score = t.get("score_at_trade")

        # 如果没有评分，尝试补算
        if score is None:
            score = _fetch_score_at_date(sym, date)

        if score is None:
            continue

        # 计算持有期收益
        hold = _holding_return(sym, date, hold_days)
        if hold is None:
            continue

        record = {
            "symbol": sym,
            "date": date,
            "source": source,
            "score": score,
            "entry_price": t.get("price"),
            "hold_days": hold["bars"],
            "return_pct": hold["return_pct"],
            "max_drawdown_pct": hold["max_drawdown_pct"],
            "exit_price": hold["exit_price"],
        }

        if score >= THRESHOLD:
            gte60_records.append(record)
        else:
            lt60_records.append(record)

    def _stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not records:
            return {"count": 0, "avg_return": None, "win_rate": None, "avg_mdd": None, "trades": []}
        returns = [r["return_pct"] for r in records]
        mdds = [r["max_drawdown_pct"] for r in records]
        wins = sum(1 for r in returns if r > 0)
        return {
            "count": len(records),
            "avg_return": round(sum(returns) / len(returns), 2),
            "win_rate": round(wins / len(records) * 100, 1),
            "avg_mdd": round(sum(mdds) / len(mdds), 2),
            "trades": records,
        }

    return {
        "model_id": model_id,
        "total_trades": len(trades),
        "scored_trades": len(gte60_records) + len(lt60_records),
        "threshold": THRESHOLD,
        "hold_days": hold_days,
        "gte60": _stats(gte60_records),
        "lt60": _stats(lt60_records),
    }


# ---------------------------------------------------------------------------
# 5. 汇总报告
# ---------------------------------------------------------------------------

def generate_report(model_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    为所有模型生成回测报告，写入 data/threshold_validation_report.json。
    """
    if model_ids is None:
        model_ids = ["kimi", "deepseek"]

    results = {}
    for mid in model_ids:
        try:
            results[mid] = backtest_model(mid)
        except Exception as e:  # noqa: BLE001
            logger.warning("回测 %s 失败: %s", mid, e)
            results[mid] = {"error": str(e)}

    report = {
        "generated_at": datetime.now().isoformat(),
        "threshold": THRESHOLD,
        "hold_days": HOLD_DAYS,
        "models": results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("✅ 阈值验证报告已保存: %s", REPORT_PATH)
    return report


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="回测验证智能荐股 60 分阈值")
    parser.add_argument("--hold-days", type=int, default=HOLD_DAYS, help="持有天数")
    parser.add_argument("--threshold", type=int, default=THRESHOLD, help="评分阈值")
    args = parser.parse_args()
    HOLD_DAYS = args.hold_days
    THRESHOLD = args.threshold
    generate_report()
