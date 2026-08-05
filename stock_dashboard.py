"""
stock_dashboard.py — 数据采集 & AI 报告生成 (v2.0 升级版)

职责：
  1. 拉取自选股行情 + 技术指标
  2. 拉取宏观指数 + FRED 利率/杠杆数据
  3. 通过 SerpApi 抓宏观/政策/个股新闻（替代旧版只能抓百度个股新闻）
  4. DeepSeek 生成：
      - AI 大盘总览 (data/report.md)
      - 个股决策卡片 (data/cards.json)
      - 每周总结 (data/weekly_report.md)
      - Morning Brief (data/morning_brief.md)
      - Evening Recap (data/evening_recap.md)

升级点（vs initial version）：
  - 新增 Morning Brief / Evening Recap 自动输出（盘中可手动跑、每日可定时跑）
  - 新闻抓取改用 utils.fetch_all_news()，覆盖宏观/政策/个股三档
  - 修复原版的几处小 bug（PE 字段、Alpha Vantage PCR 接口）
  - 保留原有所有功能
"""
from __future__ import annotations

import json
import logging
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
import requests
import ta
import yfinance as yf
from fredapi import Fred

import utils as U  # 新增：所有新闻/FedWatch/Calendar/Heatmap 走这里

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("quant_report")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    fred_api_key: str = field(default_factory=lambda: U._get_secret("FRED_API"))
    alpha_vantage_key: str = field(default_factory=lambda: U._get_secret("ALPHA_API"))
    telegram_bot_token: str = field(default_factory=lambda: U._get_secret("TELEGRAM_BOT_TOKEN"))
    telegram_chat_id: str = field(default_factory=lambda: U._get_secret("TELEGRAM_CHAT_ID"))
    deepseek_api_key: str = field(default_factory=lambda: U._get_secret("DEEPSEEK_API_KEY"))
    openrouter_api_key: str = field(default_factory=lambda: U._get_secret("OPENROUTER_API_KEY"))
    serpapi_key: str = field(default_factory=lambda: U._get_secret("SERPAPI"))

    leverage_levels: Tuple[float, ...] = (1.5, 2.0, 3.0)
    maintenance_margin: float = 0.3
    lookback_days: int = 60
    output_dir: Path = field(default_factory=lambda: Path("data"))

    vix_alert_threshold: float = 25.0
    sox_support_level: float = 11200.0
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    volume_surge_ratio: float = 1.2
    volume_contract_ratio: float = 0.8

    stocks: Tuple[str, ...] = (
        # 美股
        "MU", "AAOI", "GOOGL", "MSFT", "AMZN", "MRVL", "LITE",
        "SNDK", "NVDA", "ORCL", "SPCX", "SKHY", "TSLA",
        # 港股
        "0700.HK", "0883.HK", "3750.HK", "07709.HK", "00981.HK",
        # A 股（强一股份 / 三环集团 / 电连技术 等）
        "688809.SS", "300408.SZ", "300679.SZ", "000426.SZ",
        "002624.SZ", "601872.SS", "601975.SS", "002258.SZ",
        "001331.SZ", "600150.SS",
        # v2.5 新增：港股 + 美股
        "00293.HK", "03690.HK", "01138.HK", "03968.HK",
        "EUV", "RKLB", "GEV", "FUTU", "UNH", "NVO", "NFLX", "JNJ", "INTU",
    )

    macro_indices: Dict[str, str] = field(default_factory=lambda: {
        "VIX": "^VIX",
        "美元指数": "DX-Y.NYB",
        "标普500": "^GSPC",
        "纳斯达克100": "^NDX",
        "黄金": "GC=F",
        "WTI原油": "CL=F",
        "10年期美债收益率": "^TNX",
    })

    def __post_init__(self):
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# TypedDicts
# ---------------------------------------------------------------------------
class StockTechData(TypedDict, total=False):
    symbol: str
    收盘价: float
    涨跌幅: float
    成交量: int
    量比状态: str
    RSI_14: float
    MACD: float
    MACD信号: float
    MACD柱: float
    前日MACD柱: float
    MA5: float
    MA20: float
    MA50: float
    MA200: float
    布林上轨: float
    布林中轨: float
    布林下轨: float
    ATR: float
    PE_Ratio: Optional[float]
    资金集中价位: Optional[float]
    集中度: float
    抄底评分: int
    抄底依据: List[str]
    反弹反转信号: str
    反弹反转置信度: str
    反弹反转描述: str
    杠杆风险等级: str


class SOXSignals(TypedDict, total=False):
    最新价: float
    回撤: float
    RSI: float
    MA20: float
    MA50: float
    MA200: float
    MACD: float
    MACD信号: float
    支撑11200: bool
    技术性熊市: bool
    信号列表: List[str]


class SentimentResult(TypedDict):
    情绪指数: float
    标签: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def retry_on_error(max_retries: int = 2, exceptions: Tuple = (Exception,)):
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < max_retries:
                        logger.warning(f"{func.__name__} 第{attempt + 1}次失败: {e}")
            logger.error(f"{func.__name__} 最终失败: {last_exc}")
            return None
        return wrapper
    return decorator


def safe_float(series: pd.Series, default: float = 0.0) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:  # noqa: BLE001
        return default


def safe_fetch(ticker: str, period: str = "3mo") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# DataFetcher
# ---------------------------------------------------------------------------
class DataFetcher:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @retry_on_error(max_retries=2, exceptions=(Exception,))
    def fetch_yf(self, ticker: str, period: str = "3mo") -> pd.DataFrame:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"{ticker} 返回空数据")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    def fetch_hk(self, symbol: str) -> pd.DataFrame:
        try:
            import akshare as ak
            code = symbol.replace(".HK", "").zfill(5)
            # 用未复权实际成交价（qfq 对港股杠杆ETF 等会产生错乱）
            df = ak.stock_hk_daily(symbol=code, adjust="")
            if df.empty:
                raise ValueError("AKShare 返回空数据")
            rename_map = {
                "日期": "Date", "date": "Date",
                "开盘": "Open", "open": "Open",
                "收盘": "Close", "close": "Close",
                "最高": "High", "high": "High",
                "最低": "Low", "low": "Low",
                "成交量": "Volume", "volume": "Volume",
            }
            existing = {k: v for k, v in rename_map.items() if k in df.columns}
            df = df.rename(columns=existing)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")
            elif not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            required = ["Open", "High", "Low", "Close", "Volume"]
            for req in required:
                if req not in df.columns and req.lower() in df.columns:
                    df[req] = df[req.lower()]
            df = df[[c for c in required if c in df.columns]]
            if len(df.columns) < len(required):
                missing = set(required) - set(df.columns)
                raise ValueError(f"AKShare 返回数据缺少列: {missing}")
            return df.apply(pd.to_numeric, errors="coerce").dropna()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"AKShare {symbol} 失败({e})，回退 Yahoo Finance")
            return self.fetch_yf(symbol, period=f"{self.cfg.lookback_days}d")

    def fetch_a_share(self, symbol: str) -> pd.DataFrame:
        try:
            import akshare as ak
            code = symbol.replace(".SS", "").replace(".SZ", "")
            # 用未复权实际成交价，避免复权算法在个别标的上的偏差
            df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="")
            if df is None or df.empty:
                raise ValueError("AKShare A股 返回空")
            rename_map = {
                "日期": "Date", "开盘": "Open", "收盘": "Close",
                "最高": "High", "最低": "Low", "成交量": "Volume",
            }
            existing = {k: v for k, v in rename_map.items() if k in df.columns}
            df = df.rename(columns=existing)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")
            elif not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            required = ["Open", "High", "Low", "Close", "Volume"]
            df = df[[c for c in required if c in df.columns]]
            if len(df.columns) < len(required):
                raise ValueError("AKShare A股 数据列缺失")
            return df.apply(pd.to_numeric, errors="coerce").dropna().tail(self.cfg.lookback_days)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"AKShare A股 {symbol} 失败({e})，回退 Yahoo Finance")
            return self.fetch_yf(symbol, period=f"{self.cfg.lookback_days}d")

    def get_stock_df(self, symbol: str) -> pd.DataFrame:
        if symbol.endswith(".HK"):
            return self.fetch_hk(symbol)
        if symbol.endswith((".SS", ".SZ")):
            return self.fetch_a_share(symbol)
        return self.fetch_yf(symbol, period=f"{self.cfg.lookback_days}d")


# ---------------------------------------------------------------------------
# TechnicalAnalyzer
# ---------------------------------------------------------------------------
class TechnicalAnalyzer:
    @staticmethod
    def safe_float(series: pd.Series, default: float = 0.0) -> float:
        try:
            val = series.iloc[-1]
            return float(val) if pd.notna(val) else default
        except Exception:  # noqa: BLE001
            return default

    @classmethod
    def volume_profile_poc(cls, df: pd.DataFrame, bins: int = 20) -> Dict[str, Any]:
        if df.empty or len(df) < 5:
            return {"资金集中价位": None, "集中度": 0.0}
        low, high = df["Low"].min(), df["High"].max()
        if low >= high or pd.isna(low) or pd.isna(high):
            return {"资金集中价位": float(df["Close"].iloc[-1]), "集中度": 100.0}
        edges = np.linspace(low, high, bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        vols = np.zeros(bins)
        for i in range(bins):
            if i == bins - 1:
                mask = (tp >= edges[i]) & (tp <= edges[i + 1])
            else:
                mask = (tp >= edges[i]) & (tp < edges[i + 1])
            vols[i] = df.loc[mask, "Volume"].sum()
        total = vols.sum()
        if total == 0:
            return {"资金集中价位": float(df["Close"].iloc[-1]), "集中度": 0.0}
        max_idx = int(np.argmax(vols))
        return {
            "资金集中价位": round(float(centers[max_idx]), 2),
            "集中度": round(float(vols[max_idx] / total * 100), 2),
        }

    @classmethod
    def analyze(cls, df: pd.DataFrame, cfg: Config, symbol: str = None) -> Optional[StockTechData]:
        if len(df) < 20:
            logger.warning("数据不足20条，跳过计算")
            return None
        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        volume = df["Volume"].astype(float)

        latest_close = cls.safe_float(close)
        prev_close = cls.safe_float(close.shift(1), latest_close)
        change = (latest_close - prev_close) / prev_close * 100 if prev_close else 0.0

        vol_ma5 = volume.rolling(5).mean()
        ma5_vol = cls.safe_float(vol_ma5, 0.0)
        latest_vol = cls.safe_float(volume, 0.0)
        if ma5_vol > 0:
            ratio = latest_vol / ma5_vol
            vol_status = "放量" if ratio > cfg.volume_surge_ratio else "缩量" if ratio < cfg.volume_contract_ratio else "持平"
        else:
            vol_status = "持平"

        try:
            rsi = float(ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1])
        except Exception:  # noqa: BLE001
            rsi = 50.0

        try:
            macd_obj = ta.trend.MACD(close)
            macd_line = cls.safe_float(macd_obj.macd())
            macd_signal = cls.safe_float(macd_obj.macd_signal())
            macd_diff = cls.safe_float(macd_obj.macd_diff())
            prev_macd_diff = cls.safe_float(macd_obj.macd_diff().shift(1), macd_diff)
        except Exception:  # noqa: BLE001
            macd_line = macd_signal = macd_diff = prev_macd_diff = 0.0

        ma5 = cls.safe_float(close.rolling(5).mean())
        ma20 = cls.safe_float(close.rolling(20).mean())
        ma50 = cls.safe_float(close.rolling(50).mean())
        ma200 = cls.safe_float(close.rolling(200).mean()) if len(close) >= 200 else 0.0

        try:
            bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
            bb_high = cls.safe_float(bb.bollinger_hband())
            bb_mid = cls.safe_float(bb.bollinger_mavg())
            bb_low = cls.safe_float(bb.bollinger_lband())
        except Exception:  # noqa: BLE001
            bb_high = bb_mid = bb_low = 0.0

        try:
            atr = float(ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1])
        except Exception:  # noqa: BLE001
            atr = 0.0

        pe = cls._fetch_pe(symbol)
        poc = cls.volume_profile_poc(df)
        return {
            "收盘价": latest_close,
            "涨跌幅": change,
            "成交量": int(latest_vol),
            "量比状态": vol_status,
            "RSI_14": rsi,
            "MACD": macd_line,
            "MACD信号": macd_signal,
            "MACD柱": macd_diff,
            "前日MACD柱": prev_macd_diff,
            "MA5": ma5,
            "MA20": ma20,
            "MA50": ma50,
            "MA200": ma200,
            "布林上轨": bb_high,
            "布林中轨": bb_mid,
            "布林下轨": bb_low,
            "ATR": atr,
            "PE_Ratio": pe,
            "资金集中价位": poc["资金集中价位"],
            "集中度": poc["集中度"],
        }

    @staticmethod
    def _fetch_pe(symbol: str) -> Optional[float]:
        if not symbol:
            return None
        try:
            info = yf.Ticker(symbol).info
            pe = info.get("trailingPE") or info.get("forwardPE")
            return float(pe) if pe else None
        except Exception:  # noqa: BLE001
            return None


# ---------------------------------------------------------------------------
# Macro / Sentiment / Risk Engines
# ---------------------------------------------------------------------------
def get_sp500_signals() -> Dict[str, Any]:
    sp500 = safe_fetch("^GSPC", period="3mo")
    if sp500.empty:
        return {"error": "无法获取 标普500 数据"}
    close = sp500["Close"].astype(float)
    latest_close = float(close.iloc[-1])
    peak = float(close.max())
    drawdown = (latest_close - peak) / peak * 100
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else ma50
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    try:
        rs_last = float((gain / loss).iloc[-1])
    except Exception:  # noqa: BLE001
        rs_last = 1.0
    rsi_val = float(100 - (100 / (1 + rs_last))) if rs_last != 0 else 50.0
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd_line = exp12 - exp26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_val = float(macd_line.iloc[-1])
    macd_sig_val = float(macd_signal.iloc[-1])
    support_4800 = latest_close > 4800
    above_ma200 = latest_close > ma200
    bear_market = drawdown < -20
    signals = []
    if not support_4800:
        signals.append("⚠️ 标普500 跌破4800关键心理位")
    else:
        signals.append("✅ 标普500 站上4800")
    if not above_ma200:
        signals.append("🔻 价格跌破200日均线（长期趋势转弱）")
    if ma20 < ma50:
        signals.append("🔻 20/50日均线死叉")
    if rsi_val < 30:
        signals.append("🟢 RSI超卖（<30），可能反弹")
    elif rsi_val > 70:
        signals.append("🔴 RSI超买（>70），警惕回调")
    if bear_market:
        signals.append(f"🐻 技术性熊市（回撤{drawdown:.1f}%）")
    if macd_val < macd_sig_val:
        signals.append("🔻 MACD卖出信号")
    return {
        "最新价": latest_close,
        "52周高点": peak,
        "回撤%": drawdown,
        "MA20": ma20,
        "MA50": ma50,
        "MA200": ma200,
        "RSI": rsi_val,
        "MACD": macd_val,
        "MACD信号": macd_sig_val,
        "支撑4800": support_4800,
        "站上MA200": above_ma200,
        "技术性熊市": bear_market,
        "信号列表": signals,
    }


class ReversalAnalyzer:
    @staticmethod
    def analyze(data) -> Dict[str, str]:
        signals = []
        confidence = 0
        macd = data.get("MACD", 0)
        macd_sig = data.get("MACD信号", 0)
        macd_hist = data.get("MACD柱", 0)
        prev_hist = data.get("前日MACD柱", macd_hist)
        if macd > macd_sig and macd_hist > 0:
            signals.append("MACD金叉"); confidence += 25
        elif macd_hist > prev_hist and macd_hist < 0:
            signals.append("MACD绿柱收敛"); confidence += 15
        elif macd_hist > 0 and macd_hist > prev_hist:
            signals.append("MACD红柱放大"); confidence += 15
        rsi = data.get("RSI_14", 50)
        if 30 < rsi < 45:
            signals.append("RSI从超卖区回升"); confidence += 20
        elif rsi < 30:
            signals.append("RSI深度超卖"); confidence += 10
        elif rsi > 55:
            signals.append("RSI进入强势区"); confidence += 10
        close = data.get("收盘价", 0)
        ma5 = data.get("MA5", 0)
        ma20 = data.get("MA20", 0)
        if close > ma5 > ma20:
            signals.append("站上短期均线"); confidence += 20
        elif close > ma5:
            signals.append("站上MA5"); confidence += 10
        vol_status = data.get("量比状态", "持平")
        if vol_status == "放量":
            signals.append("放量确认"); confidence += 15
        elif vol_status == "缩量" and close > ma5:
            signals.append("缩量企稳"); confidence += 10
        bb_low = data.get("布林下轨", 0)
        if bb_low > 0 and close <= bb_low * 1.01:
            signals.append("触及布林下轨"); confidence += 10
        if confidence >= 65 and len(signals) >= 3:
            label, conf_level = "反转", "高"
        elif confidence >= 40 and len(signals) >= 2:
            label = "反弹"
            conf_level = "中" if confidence >= 55 else "低"
        else:
            label, conf_level = "无", "低"
        desc = f"{' | '.join(signals)}（置信度{confidence}分）" if signals else "暂无明确信号"
        return {"信号": label, "置信度": conf_level, "描述": desc}


class BottomFishingEngine:
    @staticmethod
    def score(data) -> Dict[str, Any]:
        score = 0
        reasons: List[str] = []
        current = data.get("收盘价", 0)
        poc = data.get("资金集中价位")
        rsi = data.get("RSI_14", 50)
        if rsi < 30:
            score += 30; reasons.append("RSI超卖")
        elif rsi < 40:
            score += 15; reasons.append("RSI偏低")
        bb_low = data.get("布林下轨", 0)
        if bb_low > 0 and current <= bb_low * 1.02:
            score += 25; reasons.append("触及/逼近布林下轨")
        if data.get("量比状态") == "缩量":
            score += 15; reasons.append("恐慌盘衰竭（缩量企稳）")
        if poc and poc > 0 and current > 0:
            dist = abs(current - poc) / poc * 100
            if dist < 3:
                score += 30; reasons.append(f"接近资金集中区（{poc:.2f}，主力成本支撑）")
        return {"抄底评分": score, "抄底依据": reasons}


class RiskEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @staticmethod
    def margin_call_price(entry: float, leverage: float, maintenance: float) -> Optional[float]:
        if entry <= 0 or leverage <= 1 or maintenance <= 0:
            return None
        return round(entry * (leverage - 1) / (leverage * (1 - maintenance)), 2)

    def analyze(self, data) -> Dict[str, Any]:
        price = data.get("收盘价", 0)
        atr = data.get("ATR", 0)
        result: Dict[str, Any] = {"当前价格": price, "ATR": atr, "details": {}}
        max_risk = "低"
        for lev in self.cfg.leverage_levels:
            mc = self.margin_call_price(price, lev, self.cfg.maintenance_margin)
            if mc is None:
                continue
            dist = price - mc
            atr_mult = round(dist / atr, 2) if atr > 0 else 999
            if atr_mult < 3:
                risk = "高"
            elif atr_mult < 6:
                risk = "中"
            else:
                risk = "低"
            if risk == "高":
                max_risk = "高"
            elif risk == "中" and max_risk != "高":
                max_risk = "中"
            result["details"][f"{lev}x"] = {"强平价": mc, "距强平ATR倍数": atr_mult, "风险等级": risk}
        result["综合风险等级"] = max_risk
        last_lev = f"{self.cfg.leverage_levels[-1]}x"
        result["描述"] = f"当前价距{last_lev}强平约{result['details'].get(last_lev, {}).get('距强平ATR倍数', 'N/A')}倍ATR"
        return result


def get_stock_put_call_ratio(symbol: str, api_key: str) -> Tuple[Optional[float], Optional[float]]:
    """
    个股级 PCR（HISTORICAL_OPTIONS）。
    注意：Alpha Vantage HISTORICAL_OPTIONS 仅覆盖美股个股期权；
    港股/A股个股期权无免费数据源（HKEX 期权数据不公开免费分发），
    因此对 .HK/.SS/.SZ 直接短路返回 None，由上层标注"无免费源"。
    """
    if not api_key or symbol.endswith((".HK", ".SS", ".SZ")):
        return None, None
    url = f"https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol={symbol}&apikey={api_key}"
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        contracts = data.get("data", [])
        if not contracts:
            return None, None
        call_vol = sum(float(c.get("volume", 0) or 0) for c in contracts if c.get("type") == "call")
        put_vol = sum(float(c.get("volume", 0) or 0) for c in contracts if c.get("type") == "put")
        call_oi = sum(float(c.get("open_interest", 0) or 0) for c in contracts if c.get("type") == "call")
        put_oi = sum(float(c.get("open_interest", 0) or 0) for c in contracts if c.get("type") == "put")
        vol_pcr = round(put_vol / call_vol, 3) if call_vol else None
        oi_pcr = round(put_oi / call_oi, 3) if call_oi else None
        return vol_pcr, oi_pcr
    except Exception as e:  # noqa: BLE001
        logger.warning(f"{symbol} 期权PCR获取失败: {e}")
        return None, None


class MacroCollector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.fetcher = DataFetcher(cfg)

    def collect_all(self) -> Dict[str, Any]:
        macro: Dict[str, Any] = {}
        for name, sym in self.cfg.macro_indices.items():
            val = self._get_macro_value(sym)
            macro[name] = val if val is not None else "无数据"
        if self.cfg.fred_api_key:
            macro.update(self._fetch_fred_batch())
        else:
            macro["美国国债规模"] = "未配置FRED Key"
            macro["芝加哥联储杠杆指数"] = "未配置FRED Key"
            macro["2年期实际利率（近似）"] = "未配置FRED Key"
        macro.update(self._fetch_finra_margin())
        return macro

    def _get_macro_value(self, symbol: str) -> Optional[float]:
        df = self.fetcher.fetch_yf(symbol, period="5d")
        if df.empty:
            return None
        val = df["Close"].iloc[-1]
        return float(val) if pd.notna(val) else None

    def _fetch_fred_batch(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        try:
            fred = Fred(api_key=self.cfg.fred_api_key)
            debt = fred.get_series("GFDEBTN")
            result["美国国债规模"] = float(debt.iloc[-1]) if not debt.empty else None
            lev = fred.get_series("NFCILEVERAGE")
            result["芝加哥联储杠杆指数"] = float(lev.iloc[-1]) if not lev.empty else None
            dgs2 = fred.get_series("DGS2")
            t5yie = fred.get_series("T5YIE")
            if not dgs2.empty and not t5yie.empty:
                result["2年期实际利率（近似）"] = f"{float(dgs2.iloc[-1] - t5yie.iloc[-1]):.2f}%"
        except Exception as e:  # noqa: BLE001
            logger.error(f"FRED 批量失败: {e}")
        return result

    def _fetch_finra_margin(self) -> Dict[str, Any]:
        """
        FINRA 保证金债务（融资余额）自动获取：
          1. 优先 FRED MDEBT（结构化、历史长，单位百万美元 → 十亿）
          2. 无 FRED key 时自动抓 FINRA 官网 margin-statistics 页面表格
          3. 本地 finra_margin.csv 作为历史补充（可选，不再强制要求手动下载）
        返回 YoY%、杠杆区间、增速是否回落等衍生信号。
        """
        # ---- 获取最新值与历史 ----
        values: List[Tuple[str, float]] = []  # (asof, value_billion)
        source_label = "FRED MDEBT"
        try:
            fred_key = self.cfg.fred_api_key
            if fred_key:
                fred = Fred(api_key=fred_key)
                s = fred.get_series("MDEBT", observation_start=(datetime.now() - timedelta(days=1000)))
                if s is not None and len(s.dropna()) > 0:
                    s = s.dropna()
                    values = [(str(d.date()), float(v) / 1000.0) for d, v in s.items()]  # 百万 → 十亿
            if not values:
                web = U.fetch_finra_margin_web()
                if web.get("value_billion") is not None:
                    values = [(web["asof"], float(web["value_billion"]))]
                    source_label = "FINRA官网直抓"
        except Exception as e:  # noqa: BLE001
            logger.warning("FINRA margin 获取失败: %s", e)

        # 本地 CSV 历史补充（如存在）
        csv_path = self.cfg.output_dir / "finra_margin.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                df["date"] = pd.to_datetime(df["date"])
                for _, r in df.sort_values("date").iterrows():
                    values.append((str(pd.to_datetime(r["date"]).date()), float(r["margin_debt"])))
            except Exception as e:  # noqa: BLE001
                logger.warning("读取 finra_margin.csv 失败: %s", e)

        if not values:
            return {"FINRA保证金债务": "数据源不可用（无 FRED key 且 FINRA 官网抓取失败）"}

        # 去重 + 排序
        seen: set = set()
        dedup: List[Tuple[str, float]] = []
        for d, v in values:
            if d not in seen and v > 0:
                seen.add(d)
                dedup.append((d, v))
        dedup.sort(key=lambda x: x[0])
        if len(dedup) < 13:
            return {
                "FINRA保证金债务": f"{dedup[-1][1]:.0f}亿美元（{dedup[-1][0]}，{source_label}）",
                "FINRA数据点": len(dedup),
                "FINRA杠杆区间": "数据不足13个月，无法计算YoY",
            }
        series = pd.Series([v for _, v in dedup], index=pd.to_datetime([d for d, _ in dedup]))
        yoy = series.pct_change(periods=12) * 100
        latest_yoy = float(yoy.iloc[-1])
        prev_yoy = float(yoy.iloc[-2])
        rolled_over = latest_yoy < prev_yoy
        if latest_yoy > 60:
            zone = "极度危险（历史顶部区间）"
        elif latest_yoy > 40:
            zone = f"警戒区（历史前兆区间）{'，且已开始回落⚠️' if rolled_over else '，仍在加速'}"
        elif latest_yoy > 25:
            zone = "偏高（需关注）"
        else:
            zone = "正常"
        return {
            "FINRA保证金债务": f"{series.iloc[-1]:.0f}亿美元（{dedup[-1][0]}，{source_label}）",
            "FINRA保证金债务YoY%": round(latest_yoy, 1),
            "FINRA杠杆区间": zone,
            "FINRA增速回落": rolled_over,
            "FINRA数据源": source_label,
        }

    def get_sox(self) -> SOXSignals:
        df = self.fetcher.fetch_yf("^SOX", period="3mo")
        if df.empty:
            return {"最新价": None, "回撤": None, "技术性熊市": False, "RSI": None, "信号列表": ["无法获取 SOX 数据"]}
        close = df["Close"].astype(float)
        latest = float(close.iloc[-1])
        peak = float(close.max())
        drawdown = (latest - peak) / peak * 100
        ma20 = safe_float(close.rolling(20).mean())
        ma50 = safe_float(close.rolling(50).mean())
        ma200 = safe_float(close.rolling(200).mean()) if len(close) >= 200 else ma50
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        try:
            rsi_val = float(100 - (100 / (1 + rs.iloc[-1])))
        except Exception:  # noqa: BLE001
            rsi_val = 50.0
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd_line = exp12 - exp26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        signals: List[str] = []
        if latest < self.cfg.sox_support_level:
            signals.append(f"⚠️ SOX 跌破{self.cfg.sox_support_level:.0f}")
        else:
            signals.append(f"✅ SOX 站上{self.cfg.sox_support_level:.0f}")
        if ma20 < ma50:
            signals.append("🔻 20/50死叉")
        if rsi_val < self.cfg.rsi_oversold:
            signals.append("🟢 RSI超卖")
        elif rsi_val > self.cfg.rsi_overbought:
            signals.append("🔴 RSI超买")
        if drawdown < -20:
            signals.append(f"🐻 技术性熊市（{drawdown:.1f}%）")
        if float(macd_line.iloc[-1]) < float(macd_signal.iloc[-1]):
            signals.append("🔻 MACD卖出")
        return {
            "最新价": latest, "回撤": drawdown, "RSI": rsi_val, "MA20": ma20, "MA50": ma50, "MA200": ma200,
            "MACD": float(macd_line.iloc[-1]), "MACD信号": float(macd_signal.iloc[-1]),
            "支撑11200": latest > self.cfg.sox_support_level, "技术性熊市": drawdown < -20, "信号列表": signals,
        }


class SentimentEngine:
    @staticmethod
    def calculate(vix: Any, vol_pcr: Any, sox_rsi: Any) -> SentimentResult:
        try:
            vix_f = float(vix) if vix not in (None, "无数据") else 18.0
        except (TypeError, ValueError):
            vix_f = 18.0
        try:
            pcr_f = float(vol_pcr) if vol_pcr not in (None, "无数据") else None
        except (TypeError, ValueError):
            pcr_f = None
        try:
            rsi_f = float(sox_rsi) if sox_rsi not in (None, "无数据") else 50.0
        except (TypeError, ValueError):
            rsi_f = 50.0
        vix_score = max(0, min(100, 100 - (vix_f - 12) * 4))
        pcr_score = max(0, min(100, 100 - (pcr_f - 0.5) * 80)) if pcr_f is not None else 50
        composite = round(vix_score * 0.4 + pcr_score * 0.3 + rsi_f * 0.3, 1)
        label = (
            "极度贪婪" if composite > 75
            else "贪婪" if composite > 55
            else "中性" if composite > 45
            else "恐惧" if composite > 25
            else "极度恐惧"
        )
        return {"情绪指数": composite, "标签": label}


class AlertService:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def send_telegram(self, message: str) -> None:
        if not self.cfg.telegram_bot_token or not self.cfg.telegram_chat_id:
            return
        try:
            url = f"https://api.telegram.org/bot{self.cfg.telegram_bot_token}/sendMessage"
            requests.post(url, json={"chat_id": self.cfg.telegram_chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Telegram 失败: {e}")

    def check_and_send(self, macro: Dict, stock_dict: Dict[str, Any], sox: SOXSignals) -> List[str]:
        alerts: List[str] = []
        if sox.get("技术性熊市"):
            alerts.append(f"🐻 SOX 技术性熊市（{sox.get('回撤', 0):.1f}%）")
        if (sox.get("最新价") or 0) < self.cfg.sox_support_level:
            alerts.append(f"⚠️ SOX 跌破 {self.cfg.sox_support_level:.0f}")
        for sym, data in stock_dict.items():
            if not data:
                continue
            rsi = data.get("RSI_14", 50)
            if rsi > self.cfg.rsi_overbought:
                alerts.append(f"🔴 {sym} RSI={rsi:.1f} 超买")
            elif rsi < self.cfg.rsi_oversold:
                alerts.append(f"🟢 {sym} RSI={rsi:.1f} 超卖")
        try:
            if float(macro.get("VIX", 0)) > self.cfg.vix_alert_threshold:
                alerts.append(f"🌪️ VIX 超 {self.cfg.vix_alert_threshold}，当前 {macro['VIX']}")
        except (TypeError, ValueError):
            pass
        if alerts:
            self.send_telegram("📢 <b>市场预警</b>\n" + "\n".join(alerts))
        return alerts


def load_futu_data() -> Dict[str, Dict]:
    path = Path("data/futu_data.json")
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("stocks", {})
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# AI 报告生成器
# ---------------------------------------------------------------------------
class AIReportGenerator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = None
        if cfg.deepseek_api_key:
            from openai import OpenAI
            self.client = OpenAI(api_key=cfg.deepseek_api_key, base_url="https://api.deepseek.com/v1")

    def _call(self, prompt: str, system: str = "专业金融分析师", max_tokens: int = 1000) -> Optional[str]:
        if not self.client:
            return None
        try:
            resp = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=0.7, max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:  # noqa: BLE001
            logger.error(f"AI 失败: {e}")
            return None

    @staticmethod
    def extract_json(raw: str) -> Any:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            s, e = raw.find("["), raw.rfind("]")
            if s != -1 and e != -1 and e > s:
                return json.loads(raw[s:e + 1])
            raise

    def generate_overview(self, macro: Dict, sox: SOXSignals, sentiment: SentimentResult, adv_dec: Optional[float], news: Dict) -> Optional[str]:
        prompt = f"""你是专业股票分析师，请生成简洁大盘总览。
日期：{datetime.now().strftime('%Y-%m-%d %H:%M')}
宏观：{json.dumps(macro, indent=2, ensure_ascii=False)}
SPY涨跌：{adv_dec if adv_dec is not None else '无'}%
SOX：最新{sox.get('最新价','N/A')} 回撤{sox.get('回撤','N/A')}% 熊市{'是' if sox.get('技术性熊市') else '否'}
情绪：{sentiment['情绪指数']}（{sentiment['标签']}）
新闻：{json.dumps(news, indent=2, ensure_ascii=False) if news else '无'}
要求：1.宏观判断 2.SOX解读 3.情绪资金 4.操作建议 5.风险提示。500字内，专业简洁，用emoji。不展开个股点位。
"""
        return self._call(prompt, max_tokens=1200)

    def generate_cards(self, stock_dict: Dict[str, StockTechData], macro: Dict, sox: SOXSignals, leverage_dict: Dict[str, Any]) -> Optional[List[Dict]]:
        valid = {s: d for s, d in stock_dict.items() if d}
        if not valid:
            return None
        facts = []
        for sym, data in valid.items():
            market = "港股" if sym.endswith(".HK") else ("A股" if sym.endswith((".SS", ".SZ")) else "美股")
            lev = leverage_dict.get(sym, {})
            mc_str = "；".join([f"{k}强平${v['强平价']:.2f}({v['距强平ATR倍数']}xATR)" for k, v in lev.get("details", {}).items()])
            bf = BottomFishingEngine.score(data)
            rev = ReversalAnalyzer.analyze(data)
            pcr_str = f"VolumePCR {data.get('Volume_PCR','N/A')} OI_PCR {data.get('OI_PCR','N/A')}"
            facts.append(f"""
{sym}（{market}）
收盘 {data.get('收盘价',0):.2f} 涨跌{data.get('涨跌幅',0):.2f}% 量比{data.get('量比状态','N/A')} 成交量{data.get('成交量',0)}
RSI {data.get('RSI_14',0):.1f} MACD {data.get('MACD',0):.3f} PE {data.get('PE_Ratio','N/A')} {pcr_str}
MA5 {data.get('MA5',0):.2f} MA20 {data.get('MA20',0):.2f} MA50 {data.get('MA50',0):.2f}
布林 {data.get('布林上轨',0):.2f}/{data.get('布林中轨',0):.2f}/{data.get('布林下轨',0):.2f}
POC {data.get('资金集中价位','N/A')} 集中度{data.get('集中度',0):.1f}%
抄底评分 {bf['抄底评分']} 依据：{', '.join(bf['抄底依据'])}
反弹反转：{rev['信号']}（{rev['置信度']}）{rev['描述']}
ATR {data.get('ATR',0):.2f}
强平线：{mc_str or 'N/A'}
""")
        prompt = f"""你是专业美股/港股分析师。针对以下股票生成结构化决策卡片。
市场：VIX={macro.get('VIX','N/A')} 标普={macro.get('标普500','N/A')} SOX回撤={sox.get('回撤','N/A')}%
个股：
{''.join(facts)}
严格只输出 JSON 数组，无前后缀、无 markdown。每只股票字段：
[
  {{
    "symbol": "代码（如 0700.HK）",
    "score": 0-100整数,
    "operation": "买入/观望/卖出",
    "trend": "看多/看空/震荡",
    "core_view": "50字内核心判断",
    "catalysts": ["利好1","利好2"],
    "risks": ["风险1","风险2"],
    "sniper": {{"ideal_buy": "理想买入位","second_buy": "二次加仓位","stop_loss": "止损位","target": "止盈目标"}},
    "sectors": ["板块1","板块2"],
    "bottom_fishing": {{"score": 0-100, "reasons": ["依据1"]}},
    "reversal": {{"signal": "反弹/反转/无", "confidence": "高/中/低"}}
  }}
]
"""
        raw = self._call(prompt, system="只输出严格合法 JSON 数组，不输出其他文字。", max_tokens=4000)
        if not raw:
            return None
        try:
            return self.extract_json(raw)
        except Exception as e:  # noqa: BLE001
            logger.error(f"卡片解析失败: {e}")
            return None

    def generate_weekly_outlook(self, macro: Dict, sox: SOXSignals, sp500: Dict, stock_dict: Dict[str, StockTechData],
                                 sentiment: SentimentResult, news: Dict) -> Optional[str]:
        tech_summary = []
        for sym, data in stock_dict.items():
            if not data:
                continue
            tech_summary.append(
                f"{sym}: 收盘{data.get('收盘价',0):.2f} 周期涨跌{data.get('涨跌幅',0):.2f}% "
                f"RSI{data.get('RSI_14',0):.1f} PE{data.get('PE_Ratio','N/A')} "
                f"均线结构{'多头' if data.get('MA5',0)>data.get('MA20',0)>data.get('MA50',0) else '非多头'} "
                f"反弹反转:{data.get('反弹反转信号','无')}"
            )
        prompt = f"""你是专业美股/港股基金经理，请撰写一份「每周总结 + 下周展望」报告。
日期：{datetime.now().strftime('%Y-%m-%d')}（周报）

【基本面/宏观】
{json.dumps(macro, indent=2, ensure_ascii=False)}
标普500：{json.dumps(sp500, indent=2, ensure_ascii=False)}
SOX：回撤{sox.get('回撤','N/A')}% 技术性熊市{'是' if sox.get('技术性熊市') else '否'}
市场情绪：{sentiment['情绪指数']}（{sentiment['标签']}）

【消息面】
{json.dumps(news, indent=2, ensure_ascii=False) if news else '本周无抓取到的新闻数据'}

【技术面（个股）】
{chr(10).join(tech_summary)}

请输出：
1. 本周复盘：市场整体表现、驱动本周走势的核心因素（宏观事件/个股消息/资金面）
2. 板块/个股结构变化：哪些股票走强、走弱，均线结构是否发生转变
3. 下周展望：基于当前宏观、技术面、消息面综合研判，下周可能的走势区间和关键观察点位
4. 下周需要重点关注的风险事件（如有已知的财报/议息会议等，若不确定请明确说明"需自行查证日历"，不要编造具体日期）
总字数800-1200字，分点清晰，专业但不夸张，不做确定性预测（用"可能""倾向于"这类措辞）。
"""
        return self._call(prompt, max_tokens=2000)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class ReportOrchestrator:
    def __init__(self):
        self.cfg = Config()
        self.fetcher = DataFetcher(self.cfg)
        self.macro_collector = MacroCollector(self.cfg)
        self.risk_engine = RiskEngine(self.cfg)
        self.alert_svc = AlertService(self.cfg)
        self.ai = AIReportGenerator(self.cfg)
        self.futu_data = load_futu_data()

    def _fetch_stock_parallel(self) -> Tuple[Dict[str, StockTechData], Dict[str, Any]]:
        results: Dict[str, StockTechData] = {}
        leverage_results: Dict[str, Any] = {}

        def worker(sym: str) -> Tuple[str, Optional[StockTechData], Optional[Dict]]:
            try:
                df = self.fetcher.get_stock_df(sym)
                data = TechnicalAnalyzer.analyze(df, self.cfg, symbol=sym)
                if not data:
                    return sym, None, None
                data["symbol"] = sym

                # 实时报价覆盖：HK / A股 用 东方财富→腾讯→新浪 的实时价与涨跌幅，
                # 避免 akshare 日线（尤其 qfq / 杠杆ETF）失真导致头条价与涨跌幅错误。
                if sym.endswith((".HK", ".SS", ".SZ")):
                    q = U.fetch_realtime_quote(sym)
                    if q.get("ok"):
                        data["收盘价"] = q["last"]
                        data["涨跌幅"] = q["pct"]

                futu_pe = self.futu_data.get(sym, {}).get("pe_ratio")
                if data.get("PE_Ratio") is None and futu_pe is not None:
                    data["PE_Ratio"] = float(futu_pe)

                bf = BottomFishingEngine.score(data)
                data["抄底评分"] = bf["抄底评分"]
                data["抄底依据"] = bf["抄底依据"]

                rev = ReversalAnalyzer.analyze(data)
                data["反弹反转信号"] = rev["信号"]
                data["反弹反转置信度"] = rev["置信度"]
                data["反弹反转描述"] = rev["描述"]

                vol_pcr, oi_pcr = get_stock_put_call_ratio(sym, self.cfg.alpha_vantage_key)
                data["Volume_PCR"] = vol_pcr
                data["OI_PCR"] = oi_pcr

                lev = self.risk_engine.analyze(data)
                data["杠杆风险等级"] = lev["综合风险等级"]
                leverage_results[sym] = lev

                return sym, data, lev
            except Exception as e:  # noqa: BLE001
                logger.error(f"{sym} 失败: {e}")
                return sym, None, None

        hk = [s for s in self.cfg.stocks if s.endswith(".HK")]
        a_share = [s for s in self.cfg.stocks if s.endswith((".SS", ".SZ"))]
        us = [s for s in self.cfg.stocks if not s.endswith((".HK", ".SS", ".SZ"))]

        with ThreadPoolExecutor(max_workers=4) as ex:
            for fut in as_completed({ex.submit(worker, s): s for s in us + a_share}):
                sym, data, lev = fut.result()
                results[sym] = data
                if lev:
                    leverage_results[sym] = lev

        for sym in hk:
            sym, data, lev = worker(sym)
            results[sym] = data
            if lev:
                leverage_results[sym] = lev

        return results, leverage_results

    def _persist(self, macro: Dict, stock_dict: Dict, sox: SOXSignals, leverage_dict: Dict,
                 cards: Optional[List], report_md: Optional[str], weekly_md: Optional[str] = None):
        out = self.cfg.output_dir
        pd.DataFrame([macro]).to_csv(out / "macro.csv", index=False)

        records = []
        for sym, data in stock_dict.items():
            if not data:
                continue
            row = {"symbol": sym, **data}
            records.append(row)
        if records:
            pd.DataFrame(records).to_csv(out / "stocks.csv", index=False)

        sox_row = {k: v for k, v in sox.items() if k != "信号列表"}
        sox_row["信号列表"] = "；".join(sox.get("信号列表", []))
        pd.DataFrame([sox_row]).to_csv(out / "sox.csv", index=False)

        if leverage_dict:
            payload = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "stocks": leverage_dict}
            (out / "leverage_risk.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        if report_md:
            (out / "report.md").write_text(report_md, encoding="utf-8")

        if weekly_md:
            (out / "weekly_report.md").write_text(weekly_md, encoding="utf-8")

        if cards:
            payload = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "stocks": cards}
            (out / "cards.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ===== 升级点 1: Morning Brief =====
    def _generate_morning_brief(self, macro: Dict, sox: SOXSignals, sentiment: SentimentResult) -> Optional[str]:
        """
        盘前早报：基于隔夜行情 + 宏观/政策新闻 + 经济日程 + FedWatch
        """
        if not self.cfg.serpapi_key:
            logger.info("未配置 SERPAPI，跳过 Morning Brief")
            return None
        # 拉宏观 / 政策新闻
        macro_news = U.fetch_macro_news(self.cfg.serpapi_key, top_n=5)
        policy_news = U.fetch_policy_news(self.cfg.serpapi_key, top_n=3)
        calendar = U.fetch_economic_calendar(self.cfg.serpapi_key)[:6]
        fw = U.calc_fedwatch_from_futures()

        # 隔夜行情（用 1d 数据近似）
        try:
            spx = safe_fetch("^GSPC", period="5d")
            ndx = safe_fetch("^NDX", period="5d")
            dji = safe_fetch("^DJI", period="5d")
            sox = safe_fetch("^SOX", period="5d")
            spx_chg = (float(spx["Close"].iloc[-1]) / float(spx["Close"].iloc[-2]) - 1) * 100 if len(spx) >= 2 else 0
            ndx_chg = (float(ndx["Close"].iloc[-1]) / float(ndx["Close"].iloc[-2]) - 1) * 100 if len(ndx) >= 2 else 0
            dji_chg = (float(dji["Close"].iloc[-1]) / float(dji["Close"].iloc[-2]) - 1) * 100 if len(dji) >= 2 else 0
            sox_chg = (float(sox["Close"].iloc[-1]) / float(sox["Close"].iloc[-2]) - 1) * 100 if len(sox) >= 2 else 0
        except Exception:  # noqa: BLE001
            spx_chg = ndx_chg = dji_chg = sox_chg = 0

        fg = U.calculate_fear_greed()

        ctx = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "market_status": f"US {U.market_status_now()['us']} / HK {U.market_status_now()['hk']}",
            "fear_greed_score": fg.get("score", 50),
            "fear_greed_label": fg.get("label", "中性"),
            "sentiment_score": sentiment["情绪指数"],
            "sentiment_label": sentiment["标签"],
            "fedwatch_verdict": fw.get("verdict", "N/A"),
            "fedwatch_cut": fw.get("prob_cut", "—"),
            "fedwatch_hold": fw.get("prob_hold", "—"),
            "fedwatch_hike": fw.get("prob_hike", "—"),
            "spx_chg": f"{spx_chg:+.2f}",
            "ndx_chg": f"{ndx_chg:+.2f}",
            "dji_chg": f"{dji_chg:+.2f}",
            "sox_chg": f"{sox_chg:+.2f}",
            "macro_news": "\n".join([f"• {n.get('title','')}（{n.get('source','')}）" for n in macro_news[:5]]),
            "policy_news": "\n".join([f"• {n.get('title','')}（{n.get('source','')}）" for n in policy_news[:3]]),
            "calendar": "\n".join([f"• {e.get('date','')} {e.get('event','')}" for e in calendar[:6]]),
        }
        brief = U.render_morning_brief(self.cfg.deepseek_api_key, ctx)
        if brief:
            (self.cfg.output_dir / "morning_brief.md").write_text(brief, encoding="utf-8")
            logger.info("Morning Brief 已写入 data/morning_brief.md")
        return brief

    # ===== 升级点 2: Evening Recap =====
    def _generate_evening_recap(self, macro: Dict, sox: SOXSignals, sentiment: SentimentResult, stock_dict: Dict) -> Optional[str]:
        """
        盘后总结：今日收盘 + 涨跌归因 + 资金轮动 + 未来 3 天风险
        """
        if not self.cfg.serpapi_key:
            logger.info("未配置 SERPAPI，跳过 Evening Recap")
            return None
        macro_news = U.fetch_macro_news(self.cfg.serpapi_key, top_n=5)
        calendar = U.fetch_economic_calendar(self.cfg.serpapi_key)[:5]
        fg = U.calculate_fear_greed()

        # 今日美股收盘
        try:
            spx = safe_fetch("^GSPC", period="5d")
            ndx = safe_fetch("^NDX", period="5d")
            dji = safe_fetch("^DJI", period="5d")
            sox = safe_fetch("^SOX", period="3mo")
            vix_df = safe_fetch("^VIX", period="5d")
            spx_chg = (float(spx["Close"].iloc[-1]) / float(spx["Close"].iloc[-2]) - 1) * 100 if len(spx) >= 2 else 0
            ndx_chg = (float(ndx["Close"].iloc[-1]) / float(ndx["Close"].iloc[-2]) - 1) * 100 if len(ndx) >= 2 else 0
            dji_chg = (float(dji["Close"].iloc[-1]) / float(dji["Close"].iloc[-2]) - 1) * 100 if len(dji) >= 2 else 0
            sox_chch = (float(sox["Close"].iloc[-1]) / float(sox["Close"].iloc[-2]) - 1) * 100 if len(sox) >= 2 else 0
            sox_peak = float(sox["Close"].max())
            sox_dd = (float(sox["Close"].iloc[-1]) / sox_peak - 1) * 100
            vix = float(vix_df["Close"].iloc[-1]) if not vix_df.empty else 0
            vix_chg = ((float(vix_df["Close"].iloc[-1]) / float(vix_df["Close"].iloc[-2]) - 1) * 100) if len(vix_df) >= 2 else 0
        except Exception:  # noqa: BLE001
            spx_chg = ndx_chg = dji_chg = sox_chch = sox_dd = 0
            vix = vix_chg = 0

        # 自选股今日表现 Top 5
        watchlist_perf = []
        for sym, data in list(stock_dict.items())[:5]:
            if not data:
                continue
            watchlist_perf.append(
                f"• {sym}: 收 {data.get('收盘价',0):.2f} 涨跌 {data.get('涨跌幅',0):+.2f}% RSI {data.get('RSI_14',0):.1f}"
            )

        # 个股新闻
        stock_news = []
        for sym in list(stock_dict.keys())[:3]:
            try:
                news = U.fetch_stock_news(sym, self.cfg.serpapi_key, is_hk=sym.endswith(".HK"), top_n=2)
                for n in news:
                    stock_news.append(f"• {sym}: {n.get('title','')}")
            except Exception:  # noqa: BLE001
                pass

        ctx = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "market_status": f"US {U.market_status_now()['us']}",
            "fear_greed_score": fg.get("score", 50),
            "fear_greed_label": fg.get("label", "中性"),
            "spx_chg": f"{spx_chg:+.2f}",
            "spx_close": f"{float(spx['Close'].iloc[-1]):.2f}" if 'spx' in dir() else "—",
            "ndx_chg": f"{ndx_chg:+.2f}",
            "ndx_close": f"{float(ndx['Close'].iloc[-1]):.2f}" if 'ndx' in dir() else "—",
            "dji_chg": f"{dji_chg:+.2f}",
            "dji_close": f"{float(dji['Close'].iloc[-1]):.2f}" if 'dji' in dir() else "—",
            "sox_chch": f"{sox_chch:+.2f}",
            "sox_drawdown": f"{sox_dd:.1f}",
            "vix": f"{vix:.2f}",
            "vix_chg": f"{vix_chg:+.2f}",
            "watchlist_perf": "\n".join(watchlist_perf) or "无",
            "macro_news": "\n".join([f"• {n.get('title','')}" for n in macro_news[:5]]),
            "calendar": "\n".join([f"• {e.get('date','')} {e.get('event','')}" for e in calendar[:5]]),
            "stock_news": "\n".join(stock_news) or "无",
        }
        recap = U.render_evening_recap(self.cfg.deepseek_api_key, ctx)
        if recap:
            (self.cfg.output_dir / "evening_recap.md").write_text(recap, encoding="utf-8")
            logger.info("Evening Recap 已写入 data/evening_recap.md")
        return recap

    # ===== v2.1 升级点 5: 4 张新 Dashboard 指标卡 =====
    def _fetch_extra_indicators_and_save(self):
        """
        拉取并保存 4 张新指标：
          - 2-Year Real-Time Scorecard (^TNX + FRED DGS2)
          - U.S. National Debt (FRED GFDEBTN)
          - FINRA Retail Margin Debt (FRED MDEBT)
          - Chicago Fed NFCI Leverage Subindex (FRED NFCILEVERAGE)
        """
        logger.info("📊 拉取 4 张新指标卡...")
        try:
            data = U.fetch_extra_indicators()
            (self.cfg.output_dir / "extra_indicators.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(f"✅ 4 张指标卡已写入 data/extra_indicators.json")
        except Exception as e:  # noqa: BLE001
            logger.error(f"❌ 4 张指标卡失败: {e}")

    # ===== v2.1 升级点 6: Vol/OI PCR（每只个股）=====
    def _fetch_options_pcr_and_save(self):
        """并发拉每只自选股的 Vol/OI PCR。"""
        logger.info("📈 拉取 Vol/OI PCR...")
        try:
            result = U.fetch_all_pcr(list(self.cfg.stocks), out_path=self.cfg.output_dir / "options_pcr.json")
            valid = sum(1 for v in result.values() if v.get("vol_pcr") is not None)
            logger.info(f"✅ Vol/OI PCR 已写入 data/options_pcr.json ({valid}/{len(result)} 有效)")
        except Exception as e:  # noqa: BLE001
            logger.error(f"❌ Vol/OI PCR 失败: {e}")

    # ===== v2.1 升级点 7: 下周走势预测（综合四维）=====
    def _generate_predictions_and_save(self):
        """
        为每只自选股生成下周走势预测，结合：
          - 技术面 (RSI/MACD/MA/ATR)
          - 消息面 (该股最近新闻)
          - 政策面 (未来 1-2 周事件)
          - 基本面 (PE/财报)
          - 期权 (Vol/OI PCR)
        """
        if not self.cfg.deepseek_api_key:
            logger.info("未配置 DEEPSEEK_API_KEY，跳过预测")
            return
        logger.info("🎯 生成下周走势预测...")

        # 读已有数据
        try:
            news_data = json.loads((self.cfg.output_dir / "news.json").read_text(encoding="utf-8")) if (self.cfg.output_dir / "news.json").exists() else {}
        except Exception:  # noqa: BLE001
            news_data = {}
        try:
            pcr_data = json.loads((self.cfg.output_dir / "options_pcr.json").read_text(encoding="utf-8")) if (self.cfg.output_dir / "options_pcr.json").exists() else {}
        except Exception:  # noqa: BLE001
            pcr_data = {}
        try:
            cards_data = json.loads((self.cfg.output_dir / "cards.json").read_text(encoding="utf-8")) if (self.cfg.output_dir / "cards.json").exists() else {"stocks": []}
        except Exception:  # noqa: BLE001
            cards_data = {"stocks": []}

        calendar = U.fetch_economic_calendar(self.cfg.serpapi_key)[:8]
        predictions: Dict[str, Any] = {"asof": datetime.now().strftime("%Y-%m-%d %H:%M"), "stocks": {}}

        for sym in self.cfg.stocks:
            try:
                # 技术面（重新拉一次确保最新）
                hist = yf.download(sym, period="3mo", progress=False, auto_adjust=True)
                if hist is None or hist.empty or len(hist) < 20:
                    logger.warning(f"{sym} 数据不足，跳过预测")
                    continue
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.get_level_values(0)
                close = hist["Close"].dropna()
                close_last = float(close.iloc[-1])
                prev_close = float(close.iloc[-2])
                change_pct = (close_last / prev_close - 1) * 100
                ma20 = float(close.rolling(20).mean().iloc[-1])
                ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else ma20
                delta = close.diff()
                gain = delta.clip(lower=0).rolling(14).mean()
                loss = (-delta.clip(upper=0)).rolling(14).mean()
                rs = gain / loss.replace(0, 1e-9)
                rsi = float(100 - 100 / (1 + rs.iloc[-1])) if pd.notna(rs.iloc[-1]) else 50
                ema12 = close.ewm(span=12).mean()
                ema26 = close.ewm(span=26).mean()
                macd = float((ema12 - ema26).iloc[-1])
                macd_signal = float((ema12 - ema26).ewm(span=9).mean().iloc[-1])
                # ATR
                high = hist["High"]
                low = hist["Low"]
                tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
                atr = float(tr.rolling(14).mean().iloc[-1])

                technical = {
                    "close": round(close_last, 2),
                    "change_pct": round(change_pct, 2),
                    "rsi": round(rsi, 1),
                    "macd": round(macd, 3),
                    "macd_signal": round(macd_signal, 3),
                    "ma20": round(ma20, 2),
                    "ma50": round(ma50, 2),
                    "atr": round(atr, 2),
                }

                # 消息面
                news = (news_data.get("stocks", {}) or {}).get(sym, [])[:5]

                # 基本面 (从 cards.json 取)
                card = next((c for c in (cards_data.get("stocks") or []) if c.get("symbol") == sym), {})

                # 期权
                opt = pcr_data.get(sym, {})

                # 生成
                pred_md = U.predict_next_week(
                    self.cfg.deepseek_api_key,
                    sym,
                    technical=technical,
                    news=news,
                    policy_events=calendar,
                    fundamentals={
                        "pe_ratio": card.get("PE_Ratio", "N/A"),
                        "last_earnings": card.get("last_earnings", "—"),
                        "sector": card.get("sector", "—"),
                    },
                    options_data=opt,
                )
                if pred_md:
                    predictions["stocks"][sym] = {
                        "prediction": pred_md,
                        "technical": technical,
                        "vol_pcr": opt.get("vol_pcr"),
                        "oi_pcr": opt.get("oi_pcr"),
                        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    }
                    logger.info(f"  ✅ {sym} 预测完成")
                time.sleep(0.3)  # 避免 DeepSeek 限流
            except Exception as e:  # noqa: BLE001
                logger.warning(f"  ⚠️ {sym} 预测失败: {e}")
                continue

        (self.cfg.output_dir / "predictions.json").write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"✅ 预测已写入 data/predictions.json ({len(predictions['stocks'])} 只)")

    def run(self, generate_brief: bool = True, generate_recap: bool = True):
        logger.info("📊 报告生成流程启动")

        macro = self.macro_collector.collect_all()
        logger.info("宏观完成")

        stock_dict, leverage_dict = self._fetch_stock_parallel()
        valid = sum(1 for v in stock_dict.values() if v)
        logger.info(f"个股完成，有效 {valid}/{len(self.cfg.stocks)}")

        sox = self.macro_collector.get_sox()
        logger.info("SOX 完成")

        sp500 = get_sp500_signals()
        if "error" in sp500:
            sp500 = {
                "最新价": "N/A", "回撤%": "N/A", "技术性熊市": False,
                "RSI": "N/A", "MA20": "N/A", "MA50": "N/A", "MA200": "N/A",
                "信号列表": ["无法获取"],
            }
        sp500_row = {k: v for k, v in sp500.items() if k != "信号列表"}
        sp500_row["信号列表"] = "；".join(sp500.get("信号列表", []))
        pd.DataFrame([sp500_row]).to_csv(self.cfg.output_dir / "sp500.csv", index=False)
        logger.info("✅ 标普500 数据已保存")

        try:
            spy = self.fetcher.fetch_yf("SPY", period="2d")
            adv_dec = float((spy["Close"].iloc[-1] - spy["Close"].iloc[-2]) / spy["Close"].iloc[-2] * 100)
        except Exception:  # noqa: BLE001
            adv_dec = None

        sentiment = SentimentEngine.calculate(macro.get("VIX", 0), None, sox.get("RSI", 50))
        logger.info(f"情绪: {sentiment['情绪指数']} ({sentiment['标签']})")

        # ===== 升级点 3: 新闻多源聚合（SerpAPI 可选，免费源始终兜底）=====
        try:
            U.fetch_all_news_multi_source(
                list(self.cfg.stocks),
                serpapi_key=self.cfg.serpapi_key,
                finnhub_key=U._get_secret("FINNHUB_API"),
                newsapi_key=U._get_secret("NEWSAPI_KEY"),
                out_path=self.cfg.output_dir / "news.json",
            )
            with open(self.cfg.output_dir / "news.json", encoding="utf-8") as f:
                news = json.load(f)
            logger.info("新闻完成 (多源聚合，含免费兜底)")
        except Exception as e:  # noqa: BLE001
            logger.error(f"新闻抓取失败: {e}")
            news = {}

        alerts = self.alert_svc.check_and_send(macro, stock_dict, sox)
        if alerts:
            logger.info(f"触发 {len(alerts)} 条预警")

        report_md = None
        cards = None
        weekly_md = None
        morning_md = None
        evening_md = None
        if self.ai.client:
            report_md = self.ai.generate_overview(macro, sox, sentiment, adv_dec, news)
            if report_md:
                logger.info("大盘报告完成")
            cards = self.ai.generate_cards(stock_dict, macro, sox, leverage_dict)
            if cards:
                logger.info(f"决策卡片完成，共 {len(cards)} 只")

            if datetime.now().weekday() == 4:  # 周五生成周报
                weekly_md = self.ai.generate_weekly_outlook(macro, sox, sp500, stock_dict, sentiment, news)
                if weekly_md:
                    logger.info("每周总结完成")

            # 升级点 4: Morning Brief / Evening Recap
            if generate_brief:
                morning_md = self._generate_morning_brief(macro, sox, sentiment)
            if generate_recap:
                evening_md = self._generate_evening_recap(macro, sox, sentiment, stock_dict)
        else:
            logger.info("跳过 AI 报告（DEEPSEEK 未配置）")

        self._persist(macro, stock_dict, sox, leverage_dict, cards, report_md, weekly_md)
        logger.info("✅ 全部完成")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]

    # === CLI 模式分发（v2.1）===
    #  --morning      只跑 Morning Brief + 数据刷新
    #  --evening      只跑 Evening Recap + 数据刷新
    #  --predictions  只生成下周走势预测
    #  --extras       只刷新 4 张新指标卡 + Vol/OI PCR
    #  --all          全部（默认）
    orch = ReportOrchestrator()

    if "--morning" in args:
        # 美股盘前模式：轻量级，只更新关键数据
        logger.info("🌅 盘前模式 (Morning Brief)")
        macro = orch.macro_collector.collect_all()
        sox = orch.macro_collector.get_sox()
        sentiment = SentimentEngine.calculate(macro.get("VIX", 0), None, sox.get("RSI", 50))
        orch._fetch_extra_indicators_and_save()
        orch._generate_morning_brief(macro, sox, sentiment)
        orch._fetch_options_pcr_and_save()
        logger.info("✅ 盘前模式完成")
    elif "--evening" in args:
        # 美股盘后模式：跑全部
        logger.info("🌙 盘后模式 (Evening Recap)")
        orch.run(generate_brief=False, generate_recap=True)
        orch._fetch_extra_indicators_and_save()
        orch._fetch_options_pcr_and_save()
        orch._generate_predictions_and_save()
        logger.info("✅ 盘后模式完成")
    elif "--predictions" in args:
        logger.info("🎯 仅生成下周走势预测")
        orch._generate_predictions_and_save()
    elif "--extras" in args:
        logger.info("📊 仅刷新 4 张新指标卡 + Vol/OI PCR")
        orch._fetch_extra_indicators_and_save()
        orch._fetch_options_pcr_and_save()
    elif "--news" in args:
        # 只刷新多源新闻（最轻量）
        logger.info("📰 仅刷新多源新闻")
        from utils import fetch_all_news_multi_source
        symbols = list(orch.cfg.stocks)
        result = fetch_all_news_multi_source(
            symbols=symbols,
            serpapi_key=orch.cfg.serpapi_key,
            finnhub_key=U._get_secret("FINNHUB_API"),
            newsapi_key=U._get_secret("NEWSAPI_KEY"),
            out_path=orch.cfg.output_dir / "news.json",
        )
        logger.info(f"✅ 新闻刷新完成: sources={result.get('sources_used', [])} stocks={sum(1 for v in result.get('stocks', {}).values() if v)}/{len(symbols)}")
    else:
        # 默认全跑
        gen_brief = "--no-brief" not in args
        gen_recap = "--no-recap" not in args
        orch.run(generate_brief=gen_brief, generate_recap=gen_recap)
        orch._fetch_extra_indicators_and_save()
        orch._fetch_options_pcr_and_save()
        orch._generate_predictions_and_save()
