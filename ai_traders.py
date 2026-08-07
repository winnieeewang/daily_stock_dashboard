"""
ai_traders.py — AI 炒手最小可行版（阶段2）

双模型（KIMI / DeepSeek）各自用虚拟 $100万 独立决策，长期跟踪对比。

状态文件结构：
  data/ai_traders/{model_id}/
    portfolio.json   # {"cash": float, "positions": {"NVDA": {"qty": 200, "avg_cost": 178.4}}, "updated_at": "..."}
    nav_history.csv  # date, nav
    trades.jsonl     # 每行一条交易记录

设计原则：
  - 所有 LLM 调用走 U._call_llm(messages, prefer=model_id)
  - 护栏：单标的≤总资产25%、只做多不做空、代码必须真实存在（yfinance校验）
  - 决策用前一交易日收盘数据，成交价用当天实时/开盘价（非高频，时间基准一致）
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 0. 路径与常量
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "ai_traders"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODEL_IDS = ["kimi", "deepseek"]
INITIAL_CASH = 1_000_000.0
MAX_POSITION_PCT = 0.25  # 单标的≤总资产25%

# 用户初始真实持仓（从截图读取）
INITIAL_POSITIONS: Dict[str, Dict[str, Any]] = {
    "AAOX": {"qty": 1500, "avg_cost": 12.58, "source": "用户初始"},
    "SNXX": {"qty": 2000, "avg_cost": 8.35, "source": "用户初始"},
    "DRLL": {"qty": 800,  "avg_cost": 18.72, "source": "用户初始"},
    "SOXL": {"qty": 500,  "avg_cost": 9.45,  "source": "用户初始"},
    "SPCX": {"qty": 1200, "avg_cost": 14.20, "source": "用户初始"},
    "NVO":  {"qty": 300,  "avg_cost": 72.50, "source": "用户初始"},
    "PLTR": {"qty": 400,  "avg_cost": 85.30, "source": "用户初始"},
    "NVDA": {"qty": 200,  "avg_cost": 178.40, "source": "用户初始"},
    "SMCI": {"qty": 500,  "avg_cost": 42.10,  "source": "用户初始"},
    "PANW": {"qty": 150,  "avg_cost": 185.60, "source": "用户初始"},
    "GOOGL": {"qty": 250, "avg_cost": 162.30, "source": "用户初始"},
    "MSFT": {"qty": 180,  "avg_cost": 487.50, "source": "用户初始"},
}


# ---------------------------------------------------------------------------
# 1. 状态管理
# ---------------------------------------------------------------------------

def _trader_dir(model_id: str) -> Path:
    d = DATA_DIR / model_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_portfolio(model_id: str) -> Dict[str, Any]:
    """加载指定模型的 portfolio.json；不存在则初始化。"""
    p = _trader_dir(model_id) / "portfolio.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.warning("%s portfolio 读取失败: %s", model_id, e)
    # 初始化
    return {
        "cash": INITIAL_CASH,
        "positions": {},
        "updated_at": datetime.now().isoformat(),
    }


def save_portfolio(model_id: str, portfolio: Dict[str, Any]) -> None:
    p = _trader_dir(model_id) / "portfolio.json"
    portfolio["updated_at"] = datetime.now().isoformat()
    p.write_text(json.dumps(portfolio, ensure_ascii=False, indent=2), encoding="utf-8")


def append_trade(model_id: str, trade: Dict[str, Any]) -> None:
    """追加一条交易记录到 trades.jsonl。"""
    p = _trader_dir(model_id) / "trades.jsonl"
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(trade, ensure_ascii=False) + "\n")


def append_nav(model_id: str, date: str, nav: float) -> None:
    """追加 NAV 到 nav_history.csv。"""
    p = _trader_dir(model_id) / "nav_history.csv"
    header = not p.exists()
    with open(p, "a", encoding="utf-8") as f:
        if header:
            f.write("date,nav\n")
        f.write(f"{date},{nav:.2f}\n")


# ---------------------------------------------------------------------------
# 2. 候选池生成（S&P500 + 纳指100 维科夫/多因子扫描）
# ---------------------------------------------------------------------------

def _sp500_nasdaq100_tickers() -> List[str]:
    """
    获取 S&P500 + 纳指100 成分股列表（去重）。
    优先用维基百科，失败则用硬编码 Top 100 兜底。
    """
    import pandas as pd
    tickers: set = set()
    # 纳指100
    try:
        ndx = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")[4]
        if "Ticker" in ndx.columns:
            tickers.update(ndx["Ticker"].dropna().astype(str).tolist())
    except Exception as e:  # noqa: BLE001
        logger.debug("维基百科纳指100失败: %s", e)
    # 标普500
    try:
        spx = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        if "Symbol" in spx.columns:
            tickers.update(spx["Symbol"].dropna().astype(str).tolist())
    except Exception as e:  # noqa: BLE001
        logger.debug("维基百科标普500失败: %s", e)
    if tickers:
        return sorted(t for t in tickers if re.match(r"^[A-Z\.]+$", t))
    # 硬编码兜底（纳斯达克官网 Top 100 + 常见标普成分）
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "PEP", "COST",
        "ADBE", "NFLX", "AMD", "INTC", "QCOM", "CSCO", "TXN", "AMAT", "MU", "LRCX",
        "KLAC", "MRVL", "SNPS", "CDNS", "ANSS", "FTNT", "PANW", "CRWD", "ZS", "OKTA",
        "PLTR", "SNOW", "DDOG", "NET", "FSLY", "TWLO", "ZM", "DOCU", "UBER", "LYFT",
        "ABNB", "BKNG", "EXPE", "MAR", "HLT", "MCD", "SBUX", "YUM", "CMG", "DPZ",
        "NKE", "LULU", "TPR", "VFC", "RL", "EL", "PG", "KO", "PEP", "WMT",
        "TGT", "COST", "HD", "LOW", "TJX", "ROST", "BBY", "DG", "DLTR", "FIVE",
        "JNJ", "PFE", "MRK", "LLY", "NVO", "UNH", "CVS", "CI", "HUM", "ANTM",
        "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "BX", "KKR", "APO",
        "V", "MA", "AXP", "DFS", "COF", "SYF", "ALLY", "PYPL", "SQ", "SOFI",
        "XOM", "CVX", "COP", "EOG", "MPC", "VLO", "PSX", "OXY", "DVN", "FANG",
        "JNJ", "PFE", "ABBV", "BMY", "MRK", "LLY", "NVO", "UNH", "AMGN", "GILD",
        "SMCI", "SNXX", "SPCX", "AAOX", "DRLL", "SOXL", "SKHY", "ORCL", "IBM", "CRM",
    ]


def build_candidate_pool(top_n: int = 15) -> List[Dict[str, Any]]:
    """
    对 S&P500+纳指100 跑维科夫+多因子评分，取 Top N 作为候选池。
    返回 [{"symbol", "score", "wyckoff", "bottom_score", ...}]。
    为控制耗时，最多扫描 150 只。
    """
    import yfinance as yf
    import screener as S
    import bottom_signal as BS

    tickers = _sp500_nasdaq100_tickers()[:150]
    candidates: List[Dict[str, Any]] = []

    # 批量下载日线（yfinance 支持多 ticker 批量）
    try:
        batch = yf.download(tickers, period="3mo", progress=False, auto_adjust=False, group_by="ticker")
    except Exception as e:  # noqa: BLE001
        logger.warning("候选池批量下载失败: %s", e)
        batch = None

    if batch is None:
        return candidates

    for sym in tickers:
        try:
            if len(tickers) == 1:
                df = batch
            else:
                df = batch.get(sym)
            if df is None or df.empty or len(df) < 30:
                continue
            # 多因子评分
            score_res = S.score_multi_factor(df)
            score = score_res.get("score", 0)
            # 维科夫
            wyckoff = S.detect_wyckoff_events(df)
            w_conf = wyckoff.get("confidence", 0)
            # 底部信号灯（宏观环境分已在 batch_bottom_signals 算好，这里简化）
            bottom = BS.calc_bottom_confidence(sym)
            candidates.append({
                "symbol": sym,
                "score": score,
                "wyckoff_confidence": round(w_conf, 2),
                "wyckoff_events": wyckoff.get("events", []),
                "bottom_score": bottom.get("bottom_score", 0),
                "bottom_label": bottom.get("label", ""),
                "bottom_traffic_light": bottom.get("traffic_light", ""),
                "bottom_macro_score": bottom.get("macro_score", 0),
                "bottom_individual_score": bottom.get("individual_score", 0),
                "bottom_macro_hits": bottom.get("macro_hits", []),
                "bottom_individual_hits": bottom.get("individual_hits", []),
                "reversal": score_res.get("reversal_signal", ""),
                "source": "候选池",
            })
        except Exception as e:  # noqa: BLE001
            logger.debug("候选池 %s 扫描失败: %s", sym, e)

    # 按多因子评分降序，取 top_n
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_n]


# ---------------------------------------------------------------------------
# 3. 市场简报生成
# ---------------------------------------------------------------------------

def _load_macro_snapshot() -> Dict[str, Any]:
    """复用 macro.csv / sox.csv / stocks.csv 生成宏观快照。"""
    out: Dict[str, Any] = {"vix": None, "sox": None, "fg": None}
    try:
        import pandas as pd
        macro_path = BASE_DIR / "data" / "macro.csv"
        if macro_path.exists():
            mdf = pd.read_csv(macro_path)
            if not mdf.empty and "VIX" in mdf.columns:
                out["vix"] = float(mdf["VIX"].iloc[-1])
        sox_path = BASE_DIR / "data" / "sox.csv"
        if sox_path.exists():
            sdf = pd.read_csv(sox_path)
            if not sdf.empty and "SOX" in sdf.columns:
                out["sox"] = float(sdf["SOX"].iloc[-1])
    except Exception as e:  # noqa: BLE001
        logger.debug("宏观快照加载失败: %s", e)
    return out


def build_briefing(portfolio: Dict[str, Any], candidates: List[Dict[str, Any]]) -> str:
    """
    生成给 LLM 的市场简报（纯文本，不含任何需要保密的信息）。
    """
    cash = portfolio.get("cash", INITIAL_CASH)
    positions = portfolio.get("positions", {})
    macro = _load_macro_snapshot()

    lines = [
        "【账户状态】",
        f"现金: ${cash:,.2f}",
    ]
    if positions:
        lines.append("持仓:")
        for sym, pos in positions.items():
            lines.append(f"  {sym} {pos.get('qty', 0)}股 (成本${pos.get('avg_cost', 0):.2f})")
    else:
        lines.append("持仓: 无")

    # 估算净值（简化：用成本价估算，实际 mark-to-market 在成交后更新）
    pos_value = sum(p.get("qty", 0) * p.get("avg_cost", 0) for p in positions.values())
    nav = cash + pos_value
    lines.append(f"估算净值: ${nav:,.2f}")
    lines.append("")

    lines.append("【市场背景】")
    if macro.get("vix"):
        lines.append(f"VIX {macro['vix']:.2f}")
    if macro.get("sox"):
        lines.append(f"SOX {macro['sox']:.2f}")
    lines.append("")

    lines.append("【系统量化候选池】（维科夫+多因子评分 Top 候选，含底部信号灯）")
    for c in candidates[:15]:
        w_txt = f"维科夫{c['wyckoff_confidence']}"
        b_txt = f"底部{c['bottom_traffic_light']}{c['bottom_score']}/4(宏观{c['bottom_macro_score']}+结构{c['bottom_individual_score']})"
        # 附加命中原因（最多2条）
        hits = (c.get("bottom_macro_hits", []) + c.get("bottom_individual_hits", []))[:2]
        hit_txt = f"[{', '.join(hits)}]" if hits else ""
        lines.append(f"- {c['symbol']}: 评分{c['score']} {w_txt} | {b_txt} {hit_txt}")
    lines.append("")

    lines.append("【规则】")
    lines.append("- 只能买卖美股上市的真实股票代码，系统会校验代码是否存在")
    lines.append("- 不允许做空、不允许融资杠杆")
    lines.append(f"- 单只标的仓位不超过总资产{MAX_POSITION_PCT*100:.0f}%")
    lines.append("- 请输出你今天的操作决定，每笔交易附上具体理由")
    lines.append("")
    lines.append("请用 JSON 格式输出今日操作决定：")
    lines.append('{"trades": [{"action":"BUY/SELL/HOLD","symbol":"代码","qty":数量,"reason":"理由","confidence":0-100}]}')

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. 护栏与校验
# ---------------------------------------------------------------------------

def _validate_ticker(symbol: str) -> bool:
    """用 yfinance 校验代码是否存在。"""
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        info = t.info or {}
        return bool(info.get("regularMarketPrice") or info.get("previousClose"))
    except Exception:  # noqa: BLE001
        return False


def _current_price(symbol: str) -> Optional[float]:
    """获取最新可成交价格。"""
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        hist = t.history(period="2d", progress=False)
        if hist is not None and not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:  # noqa: BLE001
        pass
    return None


def apply_guardrails(portfolio: Dict[str, Any], trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    护栏检查：
      1. 代码必须真实存在
      2. 只做多（action 只能是 BUY/SELL/HOLD）
      3. 单标的仓位 ≤ 总资产 25%
      4. SELL 数量不超过持仓
      5. BUY 金额不超过现金
    返回过滤后的交易列表，并在每条记录里加 "guardrail_note"。
    """
    cash = portfolio.get("cash", INITIAL_CASH)
    positions = portfolio.get("positions", {})

    # 估算当前总资产
    pos_value = sum(p.get("qty", 0) * p.get("avg_cost", 0) for p in positions.values())
    total_assets = cash + pos_value

    approved: List[Dict[str, Any]] = []
    for t in trades:
        sym = t.get("symbol", "").upper().strip()
        action = t.get("action", "HOLD").upper()
        qty = int(t.get("qty", 0))

        if action == "HOLD" or qty <= 0:
            continue

        # 规则1: 代码校验
        if not _validate_ticker(sym):
            t["guardrail_note"] = f"代码 {sym} 校验失败，跳过"
            continue

        # 规则2: 只做多
        if action not in ("BUY", "SELL"):
            t["guardrail_note"] = f"不支持的操作 {action}，跳过"
            continue

        price = _current_price(sym) or 0.0
        if price <= 0:
            t["guardrail_note"] = f"无法获取 {sym} 价格，跳过"
            continue

        if action == "BUY":
            cost = qty * price
            # 规则5: 现金足够
            if cost > cash:
                # 裁剪到可用现金
                max_qty = int(cash / price)
                if max_qty <= 0:
                    t["guardrail_note"] = f"现金不足购买 {sym}，跳过"
                    continue
                qty = max_qty
                cost = qty * price
                t["qty"] = qty
                t["guardrail_note"] = f"裁剪为 {qty} 股（现金上限）"
            # 规则3: 单标的仓位 ≤ 25%
            existing_qty = positions.get(sym, {}).get("qty", 0)
            new_total_value = (existing_qty + qty) * price
            if new_total_value > total_assets * MAX_POSITION_PCT:
                max_allowed = int(total_assets * MAX_POSITION_PCT / price) - existing_qty
                if max_allowed <= 0:
                    t["guardrail_note"] = f"{sym} 已超 25% 上限，跳过"
                    continue
                qty = max_allowed
                cost = qty * price
                t["qty"] = qty
                t["guardrail_note"] = f"裁剪为 {qty} 股（25%仓位上限）"
            cash -= cost

        elif action == "SELL":
            # 规则4: 不超过持仓
            hold_qty = positions.get(sym, {}).get("qty", 0)
            if qty > hold_qty:
                qty = hold_qty
                t["qty"] = qty
                t["guardrail_note"] = f"裁剪为 {qty} 股（持仓上限）"
            proceeds = qty * price
            cash += proceeds

        approved.append(t)
        # 更新 positions 镜像（用于后续交易的仓位检查）
        if action == "BUY":
            old = positions.get(sym, {"qty": 0, "avg_cost": 0})
            old_qty = old["qty"]
            old_cost = old_qty * old["avg_cost"]
            new_qty = old_qty + qty
            new_cost = old_cost + qty * price
            positions[sym] = {"qty": new_qty, "avg_cost": new_cost / new_qty if new_qty else 0}
        elif action == "SELL":
            old = positions.get(sym, {"qty": 0, "avg_cost": 0})
            new_qty = old["qty"] - qty
            if new_qty <= 0:
                positions.pop(sym, None)
            else:
                positions[sym] = {"qty": new_qty, "avg_cost": old["avg_cost"]}

    return approved


# ---------------------------------------------------------------------------
# 5. 模拟成交与净值更新
# ---------------------------------------------------------------------------

def execute_trades(model_id: str, portfolio: Dict[str, Any], trades: List[Dict[str, Any]], date_str: str) -> Dict[str, Any]:
    """执行已批准的交易，更新 portfolio，追加 trades.jsonl。"""
    for t in trades:
        sym = t["symbol"].upper()
        action = t["action"].upper()
        qty = t["qty"]
        price = t.get("price", _current_price(sym) or 0.0)
        t["price"] = round(price, 2)
        t["date"] = date_str
        t["model_id"] = model_id

        if action == "BUY":
            cost = qty * price
            portfolio["cash"] -= cost
            old = portfolio["positions"].get(sym, {"qty": 0, "avg_cost": 0})
            old_qty = old["qty"]
            old_cost = old_qty * old["avg_cost"]
            new_qty = old_qty + qty
            new_avg = (old_cost + cost) / new_qty if new_qty else 0
            portfolio["positions"][sym] = {"qty": new_qty, "avg_cost": round(new_avg, 2)}

        elif action == "SELL":
            proceeds = qty * price
            portfolio["cash"] += proceeds
            old = portfolio["positions"].get(sym, {"qty": 0, "avg_cost": 0})
            new_qty = old["qty"] - qty
            if new_qty <= 0:
                portfolio["positions"].pop(sym, None)
            else:
                portfolio["positions"][sym] = {"qty": new_qty, "avg_cost": old["avg_cost"]}

        append_trade(model_id, t)

    save_portfolio(model_id, portfolio)
    return portfolio


def mark_to_market(model_id: str, portfolio: Dict[str, Any], date_str: str) -> float:
    """用当前市场价计算 NAV，追加 nav_history.csv。"""
    cash = portfolio.get("cash", 0)
    positions = portfolio.get("positions", {})
    mkt_value = 0.0
    for sym, pos in positions.items():
        p = _current_price(sym)
        if p:
            mkt_value += pos.get("qty", 0) * p
    nav = cash + mkt_value
    append_nav(model_id, date_str, nav)
    return nav


# ---------------------------------------------------------------------------
# 6. LLM 决策调用
# ---------------------------------------------------------------------------

def call_model_for_decision(model_id: str, briefing: str) -> List[Dict[str, Any]]:
    """
    调用指定模型获取今日交易决策。
    model_id: "kimi" 或 "deepseek"（映射到 U._call_llm 的 prefer 参数）。
    返回解析后的 trades 列表。
    """
    import utils as U

    prefer_map = {"kimi": "kimi", "deepseek": "deepseek"}
    prefer = prefer_map.get(model_id, model_id)

    messages = [
        {"role": "system", "content": "你是一位资深美股交易员，根据市场简报做出当日交易决策。只输出 JSON，不输出其他文字。"},
        {"role": "user", "content": briefing},
    ]

    try:
        reply = U._call_llm(messages, prefer=prefer)
        if not reply:
            return []
        # 尝试从回复中提取 JSON
        json_match = re.search(r"\{.*\}", reply, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            trades = data.get("trades", [])
            if isinstance(trades, list):
                return trades
        # 兜底：如果模型没按 JSON 输出，返回空列表
        logger.warning("%s 未返回合规 JSON: %s", model_id, reply[:200])
        return []
    except Exception as e:  # noqa: BLE001
        logger.warning("%s LLM 调用失败: %s", model_id, e)
        return []


# ---------------------------------------------------------------------------
# 7. 主循环
# ---------------------------------------------------------------------------

def run_daily_trading(date_str: Optional[str] = None, dry_run: bool = False) -> Dict[str, Any]:
    """
    每日决策循环：
      1. 生成候选池
      2. 为每个 model_id 生成简报并调用 LLM
      3. 护栏检查
      4. 模拟成交
      5. 净值更新
    返回 {"kimi": {"trades": [...], "nav": float}, "deepseek": {...}}。
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    logger.info("======== AI炒手每日循环 %s ========", date_str)

    # 1) 候选池
    logger.info("生成候选池...")
    candidates = build_candidate_pool(top_n=15)
    logger.info("候选池 %d 只", len(candidates))

    results: Dict[str, Any] = {}

    for model_id in MODEL_IDS:
        logger.info("--- %s 决策 ---", model_id)
        portfolio = load_portfolio(model_id)

        # 2) 生成简报 + LLM 决策
        briefing = build_briefing(portfolio, candidates)
        raw_trades = call_model_for_decision(model_id, briefing)
        logger.info("%s 原始决策: %d 笔", model_id, len(raw_trades))

        # 3) 护栏
        approved = apply_guardrails(portfolio, raw_trades)
        logger.info("%s 通过护栏: %d 笔", model_id, len(approved))

        # 3b) 附加候选池评分到交易记录（供回测使用）
        cand_map = {c["symbol"]: c for c in candidates}
        for t in approved:
            sym = t.get("symbol", "").upper()
            if sym in cand_map:
                t["score_at_trade"] = cand_map[sym].get("score")
                t["source"] = "候选池"
            else:
                t["score_at_trade"] = None
                t["source"] = "模型自选"

        if dry_run:
            results[model_id] = {"trades": approved, "nav": None, "portfolio": portfolio}
            continue

        # 4) 成交
        portfolio = execute_trades(model_id, portfolio, approved, date_str)

        # 5) 净值更新
        nav = mark_to_market(model_id, portfolio, date_str)
        results[model_id] = {"trades": approved, "nav": nav, "portfolio": portfolio}
        logger.info("%s NAV: $%.2f", model_id, nav)

    return results


# ---------------------------------------------------------------------------
# 8. CLI 入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI 炒手每日交易循环")
    parser.add_argument("--date", default=None, help="交易日期 (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="只生成决策，不实际成交")
    args = parser.parse_args()
    run_daily_trading(date_str=args.date, dry_run=args.dry_run)
