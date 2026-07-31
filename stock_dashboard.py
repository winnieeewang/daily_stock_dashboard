from __future__ import annotations

import json
import logging
import os
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

warnings.filterwarnings("ignore")

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("quant_report")


# ==================== 配置层 ====================
@dataclass(frozen=True)
class Config:
    fred_api_key: str = field(default_factory=lambda: os.environ.get("FRED_API", ""))
    alpha_vantage_key: str = field(default_factory=lambda: os.environ.get("ALPHA_API", ""))
    telegram_bot_token: str = field(default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.environ.get("TELEGRAM_CHAT_ID", ""))
    deepseek_api_key: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY", ""))
    serpapi_key: str = field(default_factory=lambda: os.environ.get("SERPAPI", ""))

    leverage_levels: Tuple[float, ...] = (1.5, 2.0, 3.0)
    maintenance_margin: float = 0.3  # 美股/港股监管最低标准，富途实际比例可能更高
    lookback_days: int = 60
    output_dir: Path = field(default_factory=lambda: Path("data"))

    vix_alert_threshold: float = 25.0
    sox_support_level: float = 11200.0
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    volume_surge_ratio: float = 1.2
    volume_contract_ratio: float = 0.8

    stocks: Tuple[str, ...] = (
        "MU", "AAOI", "GOOGL", "MSFT", "AMZN", "MRVL", "LITE",
        "SNDK", "NVDA", "ORCL", "SPCX", "SKHY", "TSLA",
        "0700.HK", "0883.HK", "3750.HK",
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
        object.__setattr__(self, 'output_dir', Path(self.output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ==================== 数据结构 ====================
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


# ==================== 装饰器 ====================
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


# ==================== 数据获取层 ====================
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
            df = ak.stock_hk_daily(symbol=code, adjust="qfq")
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
        except Exception as e:
            logger.warning(f"AKShare {symbol} 失败({e})，回退 Yahoo Finance")
            return self.fetch_yf(symbol, period=f"{self.cfg.lookback_days}d")

    def get_stock_df(self, symbol: str) -> pd.DataFrame:
        if symbol.endswith(".HK"):
            return self.fetch_hk(symbol)
        return self.fetch_yf(symbol, period=f"{self.cfg.lookback_days}d")


# ==================== 技术指标层（含 POC） ====================
class TechnicalAnalyzer:
    @staticmethod
    def safe_float(series: pd.Series, default: float = 0.0) -> float:
        try:
            val = series.iloc[-1]
            return float(val) if pd.notna(val) else default
        except Exception:
            return default

    @classmethod
    def volume_profile_poc(cls, df: pd.DataFrame, bins: int = 20) -> Dict[str, Any]:
        """成交量分布 POC：资金最集中的价位"""
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
    def analyze(cls, df: pd.DataFrame, cfg: Config) -> Optional[StockTechData]:
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

        # 量比
        vol_ma5 = volume.rolling(5).mean()
        ma5_vol = cls.safe_float(vol_ma5, 0.0)
        latest_vol = cls.safe_float(volume, 0.0)
        if ma5_vol > 0:
            ratio = latest_vol / ma5_vol
            vol_status = "放量" if ratio > cfg.volume_surge_ratio else "缩量" if ratio < cfg.volume_contract_ratio else "持平"
        else:
            vol_status = "持平"

        # RSI
        try:
            rsi = float(ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1])
        except Exception:
            rsi = 50.0

        # MACD
        try:
            macd_obj = ta.trend.MACD(close)
            macd_line = cls.safe_float(macd_obj.macd())
            macd_signal = cls.safe_float(macd_obj.macd_signal())
            macd_diff = cls.safe_float(macd_obj.macd_diff())
            prev_macd_diff = cls.safe_float(macd_obj.macd_diff().shift(1), macd_diff)
        except Exception:
            macd_line = macd_signal = macd_diff = prev_macd_diff = 0.0

        # 均线
        ma5 = cls.safe_float(close.rolling(5).mean())
        ma20 = cls.safe_float(close.rolling(20).mean())
        ma50 = cls.safe_float(close.rolling(50).mean())
        ma200 = cls.safe_float(close.rolling(200).mean())

        # 布林带
        try:
            bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
            bb_high = cls.safe_float(bb.bollinger_hband())
            bb_mid = cls.safe_float(bb.bollinger_mavg())
            bb_low = cls.safe_float(bb.bollinger_lband())
        except Exception:
            bb_high = bb_mid = bb_low = 0.0

        # ATR
        try:
            atr = float(ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1])
        except Exception:
            atr = 0.0

        # PE
        pe = cls._fetch_pe(df)

        # POC
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
    def _fetch_pe(df: pd.DataFrame) -> Optional[float]:
        try:
            ticker = getattr(df, "name", None)
            if not ticker:
                return None
            info = yf.Ticker(ticker).info
            return info.get("trailingPE") or info.get("forwardPE")
        except Exception:
            return None


# ==================== 反弹/反转判断层 ====================
class ReversalAnalyzer:
    @staticmethod
    def analyze(data: StockTechData) -> Dict[str, str]:
        signals = []
        confidence = 0

        macd = data.get("MACD", 0)
        macd_sig = data.get("MACD信号", 0)
        macd_hist = data.get("MACD柱", 0)
        prev_hist = data.get("前日MACD柱", macd_hist)

        # MACD
        if macd > macd_sig and macd_hist > 0:
            signals.append("MACD金叉")
            confidence += 25
        elif macd_hist > prev_hist and macd_hist < 0:
            signals.append("MACD绿柱收敛")
            confidence += 15
        elif macd_hist > 0 and macd_hist > prev_hist:
            signals.append("MACD红柱放大")
            confidence += 15

        # RSI
        rsi = data.get("RSI_14", 50)
        if 30 < rsi < 45:
            signals.append("RSI从超卖区回升")
            confidence += 20
        elif rsi < 30:
            signals.append("RSI深度超卖")
            confidence += 10
        elif rsi > 55:
            signals.append("RSI进入强势区")
            confidence += 10

        # 均线
        close = data.get("收盘价", 0)
        ma5 = data.get("MA5", 0)
        ma20 = data.get("MA20", 0)
        if close > ma5 > ma20:
            signals.append("站上短期均线")
            confidence += 20
        elif close > ma5:
            signals.append("站上MA5")
            confidence += 10

        # 量价
        vol_status = data.get("量比状态", "持平")
        if vol_status == "放量":
            signals.append("放量确认")
            confidence += 15
        elif vol_status == "缩量" and close > ma5:
            signals.append("缩量企稳")
            confidence += 10

        # 布林带
        bb_low = data.get("布林下轨", 0)
        if bb_low > 0 and close <= bb_low * 1.01:
            signals.append("触及布林下轨")
            confidence += 10

        # 判断
        if confidence >= 65 and len(signals) >= 3:
            label = "反转"
            conf_level = "高"
        elif confidence >= 40 and len(signals) >= 2:
            label = "反弹"
            conf_level = "中" if confidence >= 55 else "低"
        else:
            label = "无"
            conf_level = "低"

        desc = f"{' | '.join(signals)}（置信度{confidence}分）" if signals else "暂无明确信号"
        return {
            "信号": label,
            "置信度": conf_level,
            "描述": desc,
        }


# ==================== 抄底评分层 ====================
class BottomFishingEngine:
    @staticmethod
    def score(data: StockTechData) -> Dict[str, Any]:
        score = 0
        reasons: List[str] = []
        current = data.get("收盘价", 0)
        poc = data.get("资金集中价位")

        # RSI
        rsi = data.get("RSI_14", 50)
        if rsi < 30:
            score += 30
            reasons.append("RSI超卖")
        elif rsi < 40:
            score += 15
            reasons.append("RSI偏低")

        # 布林带
        bb_low = data.get("布林下轨", 0)
        if bb_low > 0 and current <= bb_low * 1.02:
            score += 25
            reasons.append("触及/逼近布林下轨")

        # 缩量
        if data.get("量比状态") == "缩量":
            score += 15
            reasons.append("恐慌盘衰竭（缩量企稳）")

        # POC 距离
        if poc and poc > 0 and current > 0:
            dist = abs(current - poc) / poc * 100
            if dist < 3:
                score += 30
                reasons.append(f"接近资金集中区（{poc:.2f}，主力成本支撑）")

        return {"抄底评分": score, "抄底依据": reasons}


# ==================== 杠杆强平线层 ====================
class RiskEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @staticmethod
    def margin_call_price(entry: float, leverage: float, maintenance: float) -> Optional[float]:
        if entry <= 0 or leverage <= 1 or maintenance <= 0:
            return None
        return round(entry * (leverage - 1) / (leverage * (1 - maintenance)), 2)

    def analyze(self, data: StockTechData) -> Dict[str, Any]:
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

            result["details"][f"{lev}x"] = {
                "强平价": mc,
                "距强平ATR倍数": atr_mult,
                "风险等级": risk,
            }

        result["综合风险等级"] = max_risk
        result["描述"] = f"当前价距{self.cfg.leverage_levels[-1]}x强平约{result['details'].get(f'{self.cfg.leverage_levels[-1]}x', {}).get('距强平ATR倍数', 'N/A')}倍ATR"
        return result


# ==================== 宏观与 SOX ====================
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
        vol_pcr, oi_pcr = self._get_put_call_ratio()
        macro["Volume PCR"] = vol_pcr if vol_pcr is not None else "无数据"
        macro["OI PCR"] = oi_pcr if oi_pcr is not None else "无数据"
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
        except Exception as e:
            logger.error(f"FRED 批量失败: {e}")
        return result

    def _fetch_finra_margin(self) -> Dict[str, Any]:
        """FINRA 保证金债务无官方 API，支持读取用户手动下载的 CSV"""
        csv_path = self.cfg.output_dir / "finra_margin.csv"
        if not csv_path.exists():
            return {"FINRA保证金债务": "需手动下载: https://www.finra.org/rules-guidance/key-topics/margin-accounts"}
        try:
            df = pd.read_csv(csv_path)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").set_index("date")
            series = df["margin_debt"].astype(float)
            if len(series) < 13:
                return {"FINRA保证金债务": "数据不足12个月，无法计算YoY"}
            yoy = series.pct_change(periods=12) * 100
            latest_yoy = yoy.iloc[-1]
            prev_yoy = yoy.iloc[-2]
            rolled_over = latest_yoy < prev_yoy

            if latest_yoy > 60:
                zone = "极度危险（历史顶部区间）"
            elif latest_yoy > 40:
                zone = f"警戒区（历史前兆区间）{'，且已开始回落⚠️' if rolled_over else '，仍在加速'}"
            else:
                zone = "正常"

            return {
                "FINRA保证金债务YoY%": round(latest_yoy, 1),
                "FINRA杠杆区间": zone,
                "FINRA增速回落": rolled_over,
            }
        except Exception as e:
            logger.warning(f"读取 FINRA 数据失败: {e}")
            return {"FINRA保证金债务": "CSV解析失败"}

    def _get_put_call_ratio(self) -> Tuple[Optional[float], Optional[float]]:
        if not self.cfg.alpha_vantage_key:
            return None, None
        url = f"https://www.alphavantage.co/query?function=PUT_CALL_RATIO&apikey={self.cfg.alpha_vantage_key}"
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
            if "data" in data and data["data"]:
                latest = data["data"][-1]
                return (
                    float(latest.get("volume_put_call_ratio", 0)) or None,
                    float(latest.get("open_interest_put_call_ratio", 0)) or None,
                )
        except Exception as e:
            logger.warning(f"PCR 失败: {e}")
        return None, None

    def get_sox(self) -> SOXSignals:
        df = self.fetcher.fetch_yf("^SOX", period="3mo")
        if df.empty:
            return {
                "最新价": None, "回撤": None, "技术性熊市": False,
                "RSI": None, "信号列表": ["无法获取 SOX 数据"],
            }
        close = df["Close"].astype(float)
        latest = float(close.iloc[-1])
        peak = float(close.max())
        drawdown = (latest - peak) / peak * 100
        ma20 = safe_float(close.rolling(20).mean())
        ma50 = safe_float(close.rolling(50).mean())
        ma200 = safe_float(close.rolling(200).mean())

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        try:
            rsi_val = float(100 - (100 / (1 + rs.iloc[-1])))
        except Exception:
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
            "最新价": latest,
            "回撤": drawdown,
            "RSI": rsi_val,
            "MA20": ma20,
            "MA50": ma50,
            "MA200": ma200,
            "MACD": float(macd_line.iloc[-1]),
            "MACD信号": float(macd_signal.iloc[-1]),
            "支撑11200": latest > self.cfg.sox_support_level,
            "技术性熊市": drawdown < -20,
            "信号列表": signals,
        }
def get_sp500_signals():
    sp500 = safe_fetch("^GSPC", period="3mo")
    if sp500.empty:
        return {"error": "无法获取 标普500 数据"}
    close = sp500['Close'].astype(float)
    latest_close = float(close.iloc[-1])
    peak = float(close.max())
    drawdown = (latest_close - peak) / peak * 100
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1])
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    try:
        rs_last = float((gain / loss).iloc[-1])
    except:
        rs_last = 1.0
    rsi_val = float(100 - (100 / (1 + rs_last))) if rs_last != 0 else 50.0
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd_line = exp12 - exp26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_val = float(macd_line.iloc[-1])
    macd_sig_val = float(macd_signal.iloc[-1])
    # 关键位：4800 心理关口 + 200日均线
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

def safe_float(series: pd.Series, default: float = 0.0) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return default


# ==================== 情绪指数（修正版） ====================
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
            "极度贪婪" if composite > 75 else
            "贪婪" if composite > 55 else
            "中性" if composite > 45 else
            "恐惧" if composite > 25 else
            "极度恐惧"
        )
        return {"情绪指数": composite, "标签": label}


# ==================== 预警服务 ====================
class AlertService:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def send_telegram(self, message: str) -> None:
        if not self.cfg.telegram_bot_token or not self.cfg.telegram_chat_id:
            return
        try:
            url = f"https://api.telegram.org/bot{self.cfg.telegram_bot_token}/sendMessage"
            requests.post(
                url,
                json={"chat_id": self.cfg.telegram_chat_id, "text": message, "parse_mode": "HTML"},
                timeout=10,
            )
        except Exception as e:
            logger.warning(f"Telegram 失败: {e}")

    def check_and_send(self, macro: Dict, stock_dict: Dict[str, Any], sox: SOXSignals) -> List[str]:
        alerts: List[str] = []
        if sox.get("技术性熊市"):
            alerts.append(f"🐻 SOX 技术性熊市（{sox.get('回撤', 0):.1f}%）")
        if sox.get("最新价", 0) < self.cfg.sox_support_level:
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


# ==================== 新闻 ====================
class NewsFetcher:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def fetch_baidu(self, query: str) -> List[Dict]:
        if not self.cfg.serpapi_key:
            return []
        try:
            from serpapi.google_search import GoogleSearch
            results = GoogleSearch({
                "engine": "baidu_news", "q": query,
                "api_key": self.cfg.serpapi_key, "num": 3,
            }).get_dict()
            return results.get("news_results", [])
        except Exception as e:
            logger.warning(f"百度新闻失败 ({query}): {e}")
            return []


# ==================== 富途数据（可选本地补充） ====================
def load_futu_data() -> Dict[str, Dict]:
    path = Path("data/futu_data.json")
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("stocks", {})
    except Exception:
        return {}


# ==================== AI 报告 ====================
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
        except Exception as e:
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
            market = "港股" if sym.endswith(".HK") else "美股"
            lev = leverage_dict.get(sym, {})
            mc_str = "；".join([
                f"{k}强平${v['强平价']:.2f}({v['距强平ATR倍数']}xATR)"
                for k, v in lev.get("details", {}).items()
            ])
            bf = BottomFishingEngine.score(data)
            rev = ReversalAnalyzer.analyze(data)

            facts.append(f"""
{sym}（{market}）
收盘 {data.get('收盘价',0):.2f} 涨跌{data.get('涨跌幅',0):.2f}% 量比{data.get('量比状态','N/A')}
RSI {data.get('RSI_14',0):.1f} MACD {data.get('MACD',0):.3f} PE {data.get('PE_Ratio','N/A')}
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
    "sniper": {{
      "ideal_buy": "理想买入位",
      "second_buy": "二次加仓位",
      "stop_loss": "止损位",
      "target": "止盈目标"
    }},
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
        except Exception as e:
            logger.error(f"卡片解析失败: {e}")
            return None


# ==================== 主控编排 ====================
class ReportOrchestrator:
    def __init__(self):
        self.cfg = Config()
        self.fetcher = DataFetcher(self.cfg)
        self.macro_collector = MacroCollector(self.cfg)
        self.risk_engine = RiskEngine(self.cfg)
        self.alert_svc = AlertService(self.cfg)
        self.news_fetcher = NewsFetcher(self.cfg)
        self.ai = AIReportGenerator(self.cfg)
        self.futu_data = load_futu_data()

    def _fetch_stock_parallel(self) -> Tuple[Dict[str, StockTechData], Dict[str, Any]]:
        results: Dict[str, StockTechData] = {}
        leverage_results: Dict[str, Any] = {}

        def worker(sym: str) -> Tuple[str, Optional[StockTechData], Optional[Dict]]:
            try:
                df = self.fetcher.get_stock_df(sym)
                data = TechnicalAnalyzer.analyze(df, self.cfg)
                if not data:
                    return sym, None, None
                data["symbol"] = sym

                # 补充富途 PE（如果本地有数据且 yfinance 没拿到）
                futu_pe = self.futu_data.get(sym, {}).get("pe_ratio")
                if data.get("PE_Ratio") is None and futu_pe is not None:
                    data["PE_Ratio"] = float(futu_pe)

                # 叠加新功能
                bf = BottomFishingEngine.score(data)
                data["抄底评分"] = bf["抄底评分"]
                data["抄底依据"] = bf["抄底依据"]

                rev = ReversalAnalyzer.analyze(data)
                data["反弹反转信号"] = rev["信号"]
                data["反弹反转置信度"] = rev["置信度"]
                data["反弹反转描述"] = rev["描述"]

                # 杠杆风险
                lev = self.risk_engine.analyze(data)
                data["杠杆风险等级"] = lev["综合风险等级"]
                leverage_results[sym] = lev

                return sym, data, lev
            except Exception as e:
                logger.error(f"{sym} 失败: {e}")
                return sym, None, None

        hk = [s for s in self.cfg.stocks if s.endswith(".HK")]
        us = [s for s in self.cfg.stocks if not s.endswith(".HK")]

        with ThreadPoolExecutor(max_workers=4) as ex:
            for fut in as_completed({ex.submit(worker, s): s for s in us}):
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

    def _persist(self, macro: Dict, stock_dict: Dict, sox: SOXSignals, leverage_dict: Dict, cards: Optional[List], report_md: Optional[str]):
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
            payload = {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "stocks": leverage_dict,
            }
            (out / "leverage_risk.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        if report_md:
            (out / "report.md").write_text(report_md, encoding="utf-8")

        if cards:
            payload = {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "stocks": cards,
            }
            (out / "cards.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    
    def run(self):
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
                "信号列表": ["无法获取"]
            }
        sp500_row = {k: v for k, v in sp500.items() if k != "信号列表"}
        sp500_row["信号列表"] = "；".join(sp500.get("信号列表", []))
        pd.DataFrame([sp500_row]).to_csv(self.cfg.output_dir / "sp500.csv", index=False)
        logger.info("✅ 标普500 数据已保存")

        try:
            spy = self.fetcher.fetch_yf("SPY", period="2d")
            adv_dec = float((spy["Close"].iloc[-1] - spy["Close"].iloc[-2]) / spy["Close"].iloc[-2] * 100)
        except Exception:
            adv_dec = None

        sentiment = SentimentEngine.calculate(macro.get("VIX", 0), macro.get("Volume PCR"), sox.get("RSI", 50))
        logger.info(f"情绪: {sentiment['情绪指数']} ({sentiment['标签']})")

        news = {}
        if self.cfg.serpapi_key:
            for sym in self.cfg.stocks:
                news[sym] = self.news_fetcher.fetch_baidu(sym.replace(".HK", ""))

        alerts = self.alert_svc.check_and_send(macro, stock_dict, sox)
        if alerts:
            logger.info(f"触发 {len(alerts)} 条预警")

        report_md = None
        cards = None
        if self.ai.client:
            report_md = self.ai.generate_overview(macro, sox, sentiment, adv_dec, news)
            if report_md:
                logger.info("大盘报告完成")
            cards = self.ai.generate_cards(stock_dict, macro, sox, leverage_dict)
            if cards:
                logger.info(f"决策卡片完成，共 {len(cards)} 只")
        else:
            logger.info("跳过 AI 报告")

        self._persist(macro, stock_dict, sox, leverage_dict, cards, report_md)
        logger.info("✅ 全部完成")


if __name__ == "__main__":
    ReportOrchestrator().run()
