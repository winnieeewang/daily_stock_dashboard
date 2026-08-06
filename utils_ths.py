"""
同花顺 Financial-API 适配层（HiThink-Tech/Financial-API，仅行情/数据只读）。

设计原则：
  1. **零破坏**：未设置 HITHINK_FINANCE_API_KEY 时，本模块所有函数返回空/None，
     调用方应主动检测 ths_available，按需降级到现有 utils.py 数据源。
  2. **代码格式**：本仓库 thscode 形如 "600519.SH"；本模块同时支持 thscode 与
     yfinance 写法（"600519.SS" / "AAPL"），yf_to_ths / ths_to_yf 自动转换。
  3. **限流与重试**：复用官方 SDK 的指数退避（网络/限流错误）；本模块增加进程级
     1s 间隔（合规要求：同花顺 API 高频调用会被封禁）。
  4. **不混入交易**：本仓库不提供交易接口；本适配层亦不引入任何下单/撤单方法。

使用前：
  - 注册并开通：https://fuyao.aicubes.cn/admin/
  - 创建 API Key
  - 设置环境变量 HITHINK_FINANCE_API_KEY=<your_key>
"""
from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, Iterable, List, Optional

import requests
import pandas as pd

logger = logging.getLogger("winnie.ths")

THS_BASE_URL: str = os.getenv("HITHINK_FINANCE_BASE_URL", "https://fuyao.aicubes.cn")
THS_API_KEY: str = os.getenv("HITHINK_FINANCE_API_KEY", "") or os.getenv("FUYAO_TOKEN", "") or os.getenv("API_KEY", "")
THS_TIMEOUT: int = int(os.getenv("HITHINK_FINANCE_TIMEOUT", "10"))
THS_MAX_RETRIES: int = int(os.getenv("HITHINK_FINANCE_MAX_RETRIES", "3"))

ths_available: bool = bool(THS_API_KEY)

_LAST_CALL_TS: float = 0.0
_MIN_INTERVAL: float = 1.0  # 秒；同花顺公共 API 限流较严


# ---------------------------------------------------------------------------
# 工具：代码格式转换
# ---------------------------------------------------------------------------

def yf_to_ths(symbol: str) -> str:
    """
    yfinance 代码 → 同花顺 thscode。
      "600519.SS" → "600519.SH"
      "000001.SZ" → "000001.SZ"
      "AAPL"      → "AAPL"  （非 A 股保持原样，但本仓库仅支持 A 股）
    """
    if not symbol:
        return symbol
    if symbol.endswith(".SS"):
        return symbol[:-3] + ".SH"
    return symbol


def ths_to_yf(symbol: str) -> str:
    """
    同花顺 thscode → yfinance 代码（用于把同花顺结果喂回现有 utils.py）。
      "600519.SH" → "600519.SS"
    """
    if not symbol:
        return symbol
    if symbol.endswith(".SH"):
        return symbol[:-3] + ".SS"
    return symbol


# ---------------------------------------------------------------------------
# 底层：带重试的 GET（脱胎自官方 SDK 的 fuyao_client._get，简化）
# ---------------------------------------------------------------------------

def _throttle() -> None:
    """进程级节流：相邻请求至少间隔 _MIN_INTERVAL 秒。"""
    global _LAST_CALL_TS
    now = time.time()
    wait = _MIN_INTERVAL - (now - _LAST_CALL_TS)
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL_TS = time.time()


def _resolve_key(api_key: Optional[str] = None) -> str:
    """解析同花顺 key：调用方传入优先，其次模块级 THS_API_KEY 缓存。"""
    return (api_key or THS_API_KEY or "").strip()


def _ths_get(path: str, params: Optional[Dict[str, Any]] = None, api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    GET 同花顺 REST，返回 payload["data"]，出错返回 None（不抛）。
    错误写入 logger 而非传播——调用方应通过返回值判断。
    api_key 为空时回退模块级 THS_API_KEY；两者皆空则直接返回 None。
    """
    key = _resolve_key(api_key)
    if not key:
        return None
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    url = f"{THS_BASE_URL}{path}"
    headers = {"X-api-key": key, "User-Agent": "Winnie-Dashboard/1.0"}
    last_exc: Optional[Exception] = None
    for attempt in range(THS_MAX_RETRIES):
        try:
            _throttle()
            resp = requests.get(url, params=clean, headers=headers, timeout=THS_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            time.sleep(0.5 * (2**attempt))
            continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("THS GET %s 解析失败: %s", path, exc)
            return None
        if not isinstance(payload, dict):
            logger.warning("THS GET %s 返回非字典: %r", path, type(payload))
            return None
        code = payload.get("code", -1)
        if code == 0:
            return payload.get("data") or {}
        # 业务错误：401/403/429 重试
        if code in (401, 403, 429) and attempt < THS_MAX_RETRIES - 1:
            time.sleep(0.5 * (2**attempt))
            continue
        logger.warning("THS GET %s 业务错误 code=%s msg=%s", path, code, payload.get("message", ""))
        return None
    if last_exc:
        logger.warning("THS GET %s 网络失败: %s", path, last_exc)
    return None


# ---------------------------------------------------------------------------
# 高层封装
# ---------------------------------------------------------------------------

def fetch_ths_quote(symbols: Iterable[str], api_key: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    A 股实时行情快照（多标的）。
    输入 yfinance 代码列表（"600519.SS" / "000001.SZ"），返回 {yf_code: {price, open, high, low, prev_close, chg, chg_pct, volume, turnover, ts}}。
    api_key 可选：调用方传入时优先于模块级 THS_API_KEY（便于 Streamlit Cloud 运行时注入）；
    无 key / 失败 → 返回空 dict。
    """
    if not _resolve_key(api_key):
        return {}
    sym_list = list(dict.fromkeys(symbols))  # 保序去重
    if not sym_list:
        return {}
    ths_codes = [yf_to_ths(s) for s in sym_list]
    joined = ",".join(ths_codes)
    data = _ths_get("/api/a-share/prices/snapshot", {"thscodes": joined}, api_key=api_key)
    if not isinstance(data, dict) or not data:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    # 返回结构：{ thscode: {...} }
    for ths, raw in data.items():
        if not isinstance(raw, dict):
            continue
        yf = ths_to_yf(ths)
        out[yf] = {
            "price": raw.get("price") or raw.get("last") or raw.get("close"),
            "open": raw.get("open"),
            "high": raw.get("high"),
            "low": raw.get("low"),
            "prev_close": raw.get("prev_close") or raw.get("pre_close"),
            "chg": raw.get("chg") or raw.get("change"),
            "chg_pct": raw.get("chg_pct") or raw.get("change_percent"),
            "volume": raw.get("volume"),
            "turnover": raw.get("turnover") or raw.get("amount"),
            "ts": raw.get("ts") or raw.get("timestamp"),
        }
    return out


def fetch_ths_history(
    symbol: str,
    days: int = 60,
    adjust: str = "none",
) -> pd.DataFrame:
    """
    A 股历史 K 线（日频）。
      symbol: yfinance 代码（"600519.SS"）
      days:   取最近 N 个交易日
      adjust: "none" / "qfq" / "hfq"
    返回 DataFrame(columns=[date, open, high, low, close, volume, turnover])；
    无 key / 失败 / 非 A 股 → 返回空 DataFrame。
    """
    if not ths_available or not symbol or not (symbol.endswith((".SS", ".SZ"))):
        return pd.DataFrame()
    ths = yf_to_ths(symbol)
    end_ms = int(time.time() * 1000)
    # 多取 ~30% 以覆盖节假日
    start_ms = end_ms - int(days * 1.5 * 24 * 3600 * 1000)
    data = _ths_get(
        "/api/a-share/prices/historical",
        {"thscode": ths, "period": "daily", "adjust": adjust, "start_ms": start_ms, "end_ms": end_ms},
    )
    if not data or not isinstance(data, (dict, list)):
        return pd.DataFrame()
    # 形态兼容：data 可能为 {records: [...]} 或直接 list
    rows = data.get("records") if isinstance(data, dict) and data.get("records") else (data if isinstance(data, list) else [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # 字段名归一
    rename = {
        "date": "date", "trade_date": "date", "ts": "date",
        "open": "open", "high": "high", "low": "low", "close": "close",
        "volume": "volume", "vol": "volume",
        "turnover": "turnover", "amount": "turnover",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    for c in ("open", "high", "low", "close", "volume", "turnover"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], unit="ms" if df["date"].max() > 1e11 else None, errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df.tail(days).reset_index(drop=True) if len(df) > days else df


def fetch_ths_valuation(symbol: str) -> Dict[str, Any]:
    """
    A 股估值快照：PE / PB / PS / PCF。
    无 key / 失败 / 非 A 股 → 返回空 dict。
    """
    if not ths_available or not symbol or not symbol.endswith((".SS", ".SZ")):
        return {}
    ths = yf_to_ths(symbol)
    data = _ths_get("/api/a-share/valuations/snapshot", {"thscode": ths})
    if not isinstance(data, dict) or not data:
        return {}
    return {
        "pe_ttm": data.get("pe_ttm") or data.get("pe"),
        "pb": data.get("pb"),
        "ps_ttm": data.get("ps_ttm") or data.get("ps"),
        "pcf": data.get("pcf"),
        "total_mv": data.get("total_mv") or data.get("mkt_cap"),
        "circ_mv": data.get("circ_mv"),
        "ts": data.get("ts") or data.get("timestamp"),
    }


def fetch_ths_corp_actions(symbol: str, days: int = 365) -> List[Dict[str, Any]]:
    """
    公司行动（分红/送转/复权因子）。返回 [{date, type, amount, ...}]。
    """
    if not ths_available or not symbol or not symbol.endswith((".SS", ".SZ")):
        return []
    ths = yf_to_ths(symbol)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 3600 * 1000
    data = _ths_get(
        "/api/a-share/corporate-actions/adjustment-factors",
        {"thscode": ths, "start_ms": start_ms, "end_ms": end_ms},
    )
    if not data:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("records", "list", "items"):
            v = data.get(k)
            if isinstance(v, list):
                return v
    return []


def ths_status() -> Dict[str, Any]:
    """诊断用：返回同花顺适配层当前状态（key 是否设置、调用计数等）。"""
    return {
        "available": ths_available,
        "base_url": THS_BASE_URL,
        "key_set": bool(THS_API_KEY),
        "key_masked": (THS_API_KEY[:4] + "***" + THS_API_KEY[-2:]) if THS_API_KEY and len(THS_API_KEY) > 6 else "",
        "timeout": THS_TIMEOUT,
        "max_retries": THS_MAX_RETRIES,
    }
