"""
risk.py — 风险管理层（Risk Management Layer）

功能：
  1. positions.json：记录用户实际持仓（买入价 / 仓位 / 成本 / 建仓日期）
  2. R 倍数框架：以入场风险 R 为单位，生成 +1R / +2R 分批减仓的个性化止盈计划
  3. 个股杠杆强平线：给定杠杆倍数 + 维持保证金率，计算强平价与距强平 ATR 倍数
  4. 宏观斩杀线：基于 FINRA 融资余额 YoY% 判定宏观杠杆环境，输出仓位上限建议

设计原则：
  - positions.json 由用户在页面手动维护（增删改），本模块只负责读写与计算
  - 所有函数对脏数据免疫（缺字段 / 非数字 / 空文件均安全返回）
  - 不依赖任何付费 API
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("risk")


# ---------------------------------------------------------------------------
# positions.json 读写
# ---------------------------------------------------------------------------
# 格式：
# {
#   "updated_at": "2026-08-05 14:00",
#   "positions": {
#     "0700.HK": {
#         "buy_price": 320.5,        # 建仓均价
#         "shares": 1000,            # 股数
#         "entry_date": "2026-08-01",
#         "note": "腾讯控股，中线持有"
#     }
#   }
# }


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def load_positions(path: Optional[Path] = None) -> Dict[str, Any]:
    """读取 positions.json，返回 {"updated_at":..., "positions":{...}}。"""
    p = Path(path) if path else Path("data") / "positions.json"
    if not p.exists():
        return {"updated_at": "", "positions": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"updated_at": "", "positions": {}}
        pos = data.get("positions") or {}
        if not isinstance(pos, dict):
            pos = {}
        return {"updated_at": data.get("updated_at", ""), "positions": pos}
    except Exception as e:  # noqa: BLE001
        logger.warning("读取 positions.json 失败: %s", e)
        return {"updated_at": "", "positions": {}}


def save_position(
    symbol: str,
    buy_price: float,
    shares: float,
    entry_date: str = "",
    note: str = "",
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """新增/更新一条持仓。返回更新后的完整数据。"""
    data = load_positions(path)
    data["positions"][symbol] = {
        "buy_price": round(float(buy_price), 4),
        "shares": float(shares),
        "entry_date": entry_date or datetime.now().strftime("%Y-%m-%d"),
        "note": note,
    }
    data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    p = Path(path) if path else Path("data") / "positions.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def delete_position(symbol: str, path: Optional[Path] = None) -> Dict[str, Any]:
    """删除一条持仓。返回更新后的完整数据。"""
    data = load_positions(path)
    data["positions"].pop(symbol, None)
    data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    p = Path(path) if path else Path("data") / "positions.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


# ---------------------------------------------------------------------------
# R 倍数框架：+1R / +2R 分批减仓
# ---------------------------------------------------------------------------
def r_multiple_plan(
    buy_price: float,
    stop_loss: float,
    current_price: Optional[float] = None,
    atr: Optional[float] = None,
    shares: float = 0.0,
) -> Dict[str, Any]:
    """
    生成 R 倍数分批止盈计划（垫厚利润思路）。

    规则：
      - R = 单股风险 = |buy_price - stop_loss|
      - +1R = buy + 1×R  → 减仓 1/3（保本止损上移到 buy）
      - +2R = buy + 2×R  → 再减仓 1/3（止损上移到 +1R）
      - 剩余 1/3 用移动止损追踪，直到趋势破坏
    buy_price / stop_loss / atr 任一不可用则返回 error。
    """
    bp = _num(buy_price)
    sl = _num(stop_loss)
    if bp is None or sl is None or sl >= bp:
        return {"ok": False, "error": "buy_price 与 stop_loss 需为正数且 stop_loss < buy_price"}
    r = bp - sl
    plan = {
        "ok": True,
        "buy_price": round(bp, 4),
        "stop_loss": round(sl, 4),
        "risk_r": round(r, 4),
        "risk_pct": round(r / bp * 100, 2),
        "stages": [
            {
                "stage": "+1R",
                "price": round(bp + 1 * r, 4),
                "price_pct": round((bp + 1 * r) / bp * 100 - 100, 2),
                "action": "减仓 1/3，止损上移至成本价（保本）",
            },
            {
                "stage": "+2R",
                "price": round(bp + 2 * r, 4),
                "price_pct": round((bp + 2 * r) / bp * 100 - 100, 2),
                "action": "再减 1/3，止损上移至 +1R 价位（锁定 +1R 利润）",
            },
            {
                "stage": "剩余1/3",
                "price": round(bp + r, 4),  # 移动止损触发位 = +1R 价位（锁定已兑现 +1R 利润，回撤至此即清仓）
                "price_pct": round((bp + r) / bp * 100 - 100, 2),
                "action": f"移动止损追踪，回撤至 +1R 价位（{round(bp + r, 4)}）或跌破 MA20 时清仓剩余 1/3",
            },
        ],
    }
    # 移动止损的具体触发价（与第三批共用的清晰价位点）
    plan["trail_trigger_price"] = round(bp + r, 4)
    cp = _num(current_price)
    if cp is not None:
        pnl_r = (cp - bp) / r
        plan["current_price"] = round(cp, 4)
        plan["current_pnl_pct"] = round((cp - bp) / bp * 100, 2)
        plan["current_pnl_R"] = round(pnl_r, 2)
        if pnl_r >= 2:
            plan["current_stage"] = "已到 +2R，执行第三阶段移动止损"
        elif pnl_r >= 1:
            plan["current_stage"] = "已到 +1R，执行第二阶段：再减 1/3，止损上移至 +1R"
        elif pnl_r > 0:
            plan["current_stage"] = "浮盈中，等待 +1R 触发第一批减仓"
        else:
            plan["current_stage"] = "浮亏中，执行原止损纪律（触及止损位即离场）"
    if atr is not None:
        atr_v = _num(atr)
        if atr_v is not None and atr_v > 0:
            plan["stop_in_atr"] = round(r / atr_v, 2)  # 止损距入场几倍 ATR
            plan["r1_in_atr"] = round((bp + r) / atr_v, 2)
    if shares and shares > 0:
        plan["stage_shares"] = {
            "at_1R": round(shares / 3, 0),
            "at_2R": round(shares / 3, 0),
            "trail": round(shares - shares / 3 * 2, 0),
        }
    return plan


# ---------------------------------------------------------------------------
# 个股杠杆强平线
# ---------------------------------------------------------------------------
def liquidation_line(entry: float, leverage: float, maintenance: float = 0.3) -> Optional[float]:
    """
    杠杆强平价公式：
      margin_call_price = entry × (leverage - 1) / (leverage × (1 - maintenance))
    示例：2x 杠杆 + 30% 维持保证金 → 强平价 ≈ entry × 0.714（跌 ~28.6% 触发强平）
    """
    e = _num(entry)
    if e is None or leverage is None or leverage <= 1 or maintenance is None or maintenance <= 0:
        return None
    return round(e * (leverage - 1) / (leverage * (1 - maintenance)), 2)


def leverage_risk_plan(
    entry: float,
    current: float,
    atr: Optional[float] = None,
    leverage_levels: tuple = (1.5, 2.0, 3.0),
    maintenance: float = 0.3,
) -> Dict[str, Any]:
    """按多个杠杆档位给出强平价 + 距强平 ATR 倍数 + 风险等级。"""
    out: Dict[str, Any] = {"ok": False, "details": {}, "error": ""}
    e = _num(entry)
    c = _num(current)
    if e is None or c is None:
        out["error"] = "entry / current 需为正数"
        return out
    out["ok"] = True
    out["entry"] = round(e, 4)
    out["current"] = round(c, 4)
    for lev in leverage_levels:
        mc = liquidation_line(e, lev, maintenance)
        atr_mult = None
        if mc is not None and atr is not None:
            atr_v = _num(atr)
            if atr_v is not None and atr_v > 0:
                atr_mult = round((c - mc) / atr_v, 2)
        risk = "高危"
        if mc is not None:
            dd = (c - mc) / c * 100
            risk = "高危" if dd < 10 else ("中危" if dd < 25 else "低危")
        out["details"][f"{lev}x"] = {"强平价": mc, "距强平ATR倍数": atr_mult, "风险等级": risk}
    return out


# ---------------------------------------------------------------------------
# 宏观斩杀线：FINRA 融资余额环境 → 仓位上限建议
# ---------------------------------------------------------------------------
def macro_kill_line(finra_margin_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    基于 FINRA 保证金债务 YoY% 与 NFCI 杠杆子指数输出宏观杠杆环境与仓位上限建议。

    规则（历史顶部区间参考 2000 / 2007 / 2021 峰值）：
      - 融资 YoY > 60%            → 极度危险，仓位上限 ≤ 30%（宏观斩杀线）
      - 40% < YoY ≤ 60%           → 警戒，上限 ≤ 50%
      - 25% < YoY ≤ 40%           → 偏高，上限 ≤ 70%
      - 融资增速回落（YoY 下行）   → 视为去杠杆信号，降档处理
    无数据时返回保守默认（上限 60%）。
    """
    out: Dict[str, Any] = {
        "ok": False,
        "level": "未知",
        "max_position_pct": 60,
        "advice": "FINRA 数据不可用，按保守上限 60% 执行",
        "source": "",
    }
    finra = finra_margin_data or {}
    yoy = _num(finra.get("yoy_chg_pct") or finra.get("FINRA保证金债务YoY%"))
    rolled = bool(finra.get("FINRA增速回落"))
    if yoy is None:
        return out
    out["ok"] = True
    out["yoy_pct"] = round(yoy, 1)
    out["rolled_over"] = rolled
    out["source"] = str(finra.get("source", "FINRA"))
    if yoy > 60:
        out["level"], out["max_position_pct"] = "极度危险", 30
    elif yoy > 40:
        out["level"], out["max_position_pct"] = "警戒", 50
    elif yoy > 25:
        out["level"], out["max_position_pct"] = "偏高", 70
    else:
        out["level"], out["max_position_pct"] = "正常", 100
    if rolled and out["max_position_pct"] < 100:
        out["max_position_pct"] = min(out["max_position_pct"] + 10, 100)
        out["advice"] = f"融资增速已回落（去杠杆中），上限上调至 {out['max_position_pct']}%"
    else:
        out["advice"] = f"融资 YoY {out['yoy_pct']}% → {out['level']}，建议总仓位上限 {out['max_position_pct']}%"
    return out


# ---------------------------------------------------------------------------
# 组合级：合并所有持仓的 R 计划
# ---------------------------------------------------------------------------
def build_portfolio_plan(
    positions: Dict[str, Dict[str, Any]],
    quote_map: Optional[Dict[str, Dict[str, Any]]] = None,
    atr_map: Optional[Dict[str, Optional[float]]] = None,
) -> List[Dict[str, Any]]:
    """
    遍历 positions 生成每只的 R 倍数计划。
    quote_map: {sym: {last, ...}}；atr_map: {sym: atr}；均缺省时跳过实时部分。
    """
    plans: List[Dict[str, Any]] = []
    for sym, pos in (positions or {}).items():
        bp = _num(pos.get("buy_price"))
        if bp is None:
            continue
        # 止损默认取 2.2×ATR（与荐股逻辑一致），无 ATR 时退化为 7%
        atr = (atr_map or {}).get(sym) if atr_map else None
        atr_v = _num(atr)
        sl = None
        if atr_v is not None:
            sl = bp - max(atr_v * 2.2, 0.01)
        else:
            sl = bp * 0.93
        cp = None
        if quote_map:
            q = quote_map.get(sym) or {}
            cp = _num(q.get("last"))
        plan = r_multiple_plan(bp, sl, current_price=cp, atr=atr_v, shares=_num(pos.get("shares")) or 0)
        if plan.get("ok"):
            plan["symbol"] = sym
            plan["note"] = pos.get("note", "")
            plan["entry_date"] = pos.get("entry_date", "")
            plans.append(plan)
    return plans
