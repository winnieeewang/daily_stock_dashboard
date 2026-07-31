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

# ==================== 1. 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("quant_report")


# ==================== 2. 配置层 ====================
@dataclass(frozen=True)
class Config:
    """全量配置集中管理"""
    fred_api_key: str = field(default_factory=lambda: os.environ.get("FRED_API", ""))
    alpha_vantage_key: str = field(default_factory=lambda: os.environ.get("ALPHA_API", ""))
    telegram_bot_token: str = field(default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.environ.get("TELEGRAM_CHAT_ID", ""))
    deepseek_api_key: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY", ""))
    serpapi_key: str = field(default_factory=lambda: os.environ.get("SERPAPI", ""))

    leverage_levels: Tuple[float, ...] = (1.5, 2.0)
    maintenance_margin: float = 0.30
    lookback_days: int = 60

    # 路径配置
    output_dir: Path = field(default_factory=lambda: Path("data"))

    # 阈值配置（原魔法数字）
    vix_alert_threshold: float = 25.0
    sox_support_level: float = 11200.0
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    volume_surge_ratio: float = 1.2
    volume_contract_ratio: float = 0.8

    # 股票池
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
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ==================== 3. 数据结构定义 ====================
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
    MA5: float
    MA20: float
    MA50: float
    MA200: float
    布林上轨: float
    布林中轨: float
    布林下轨: float
    ATR: float
    PE_Ratio: Optional[float]


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


# ==================== 4. 工具装饰器 ====================
def retry_on_error(max_retries: int = 2, exceptions: Tuple = (Exception,)):
    """带日志的简单重试装饰器"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    logger.warning(f"{func.__name__} 第{attempt + 1}次失败: {e}")
            logger.error(f"{func.__name__} 最终失败: {last_exc}")
            return None
        return wrapper
    return decorator


# ==================== 5. 数据获取层 ====================
class DataFetcher:
    """统一封装 Yahoo Finance / AKShare 数据获取"""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @retry_on_error(max_retries=2, exceptions=(Exception,))
    def fetch_yf(self, ticker: str, period: str = "3mo") -> pd.DataFrame:
        """安全获取 Yahoo Finance 数据"""
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"{ticker} 返回空数据")
        # 处理可能的 MultiIndex 列
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    def fetch_hk(self, symbol: str) -> pd.DataFrame:
        """港股数据优先走 AKShare，失败回退 Yahoo"""
        try:
            import akshare as ak
            code = symbol.replace(".HK", "").zfill(5)
            df = ak.stock_hk_daily(symbol=code, adjust="qfq")
            if df.empty:
                raise ValueError("AKShare 返回空数据")

            # === 关键修复：兼容不同版本 akshare 的列名 ===
            # 有些版本返回中文列名，有些返回英文，做双重映射
            rename_map = {
                "日期": "Date",
                "date": "Date",
                "开盘": "Open",
                "open": "Open",
                "收盘": "Close",
                "close": "Close",
                "最高": "High",
                "high": "High",
                "最低": "Low",
                "low": "Low",
                "成交量": "Volume",
                "volume": "Volume",
            }
            # 只重命名实际存在的列
            existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
            df = df.rename(columns=existing_renames)

            # 如果已经有 Date 列，转为 datetime 并设为索引
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")
            elif not isinstance(df.index, pd.DatetimeIndex):
                # 尝试把第一列当日期
                df.index = pd.to_datetime(df.index)
            
            df = df.sort_index()

            required = ["Open", "High", "Low", "Close", "Volume"]
            # 如果还有缺失，尝试大小写不敏感匹配
            for req in required:
                if req not in df.columns:
                    # 尝试小写
                    if req.lower() in df.columns:
                        df[req] = df[req.lower()]
            
            # 只保留需要的列，并强制数值化
            df = df[[c for c in required if c in df.columns]]
            if len(df.columns) < len(required):
                missing = set(required) - set(df.columns)
                raise ValueError(f"AKShare 返回数据缺少列: {missing}")

            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            return df
        except Exception as e:
            logger.warning(f"AKShare 获取 {symbol} 失败({e})，回退 Yahoo Finance")
            return self.fetch_yf(symbol, period=f"{self.cfg.lookback_days}d")

    def get_stock_df(self, symbol: str) -> pd.DataFrame:
        """根据市场自动路由数据源"""
        if symbol.endswith(".HK"):
            return self.fetch_hk(symbol)
        return self.fetch_yf(symbol, period=f"{self.cfg.lookback_days}d")


# ==================== 6. 技术指标层 ====================
class TechnicalAnalyzer:
    """技术指标计算与清洗"""

    @staticmethod
    def safe_float(series: pd.Series, default: float = 0.0) -> float:
        try:
            val = series.iloc[-1]
            return float(val) if pd.notna(val) else default
        except Exception:
            return default

    @classmethod
    def analyze(cls, df: pd.DataFrame, cfg: Config) -> Optional[StockTechData]:
        if len(df) < 20:
            logger.warning("数据不足20条，跳过技术指标计算")
            return None

        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        volume = df["Volume"].astype(float)

        latest_close = cls.safe_float(close)
        prev_close = cls.safe_float(close.shift(1), latest_close)
        change = (latest_close - prev_close) / prev_close * 100 if prev_close else 0.0

        # 量比状态
        vol_ma5 = volume.rolling(5).mean()
        ma5_vol = cls.safe_float(vol_ma5, 0.0)
        if ma5_vol > 0:
            latest_vol = cls.safe_float(volume, 0.0)
            ratio = latest_vol / ma5_vol
            if ratio > cfg.volume_surge_ratio:
                vol_status = "放量"
            elif ratio < cfg.volume_contract_ratio:
                vol_status = "缩量"
            else:
                vol_status = "持平"
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
        except Exception:
            macd_line = macd_signal = macd_diff = 0.0

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
            atr = float(ta.volatility.AverageTrueRange(
                high, low, close, window=14
            ).average_true_range().iloc[-1])
        except Exception:
            atr = 0.0

        # PE（独立获取，失败不影响主流程）
        pe = cls._fetch_pe(df, close)

        return {
            "收盘价": latest_close,
            "涨跌幅": change,
            "成交量": int(cls.safe_float(volume, 0)),
            "量比状态": vol_status,
            "RSI_14": rsi,
            "MACD": macd_line,
            "MACD信号": macd_signal,
            "MACD柱": macd_diff,
            "MA5": ma5,
            "MA20": ma20,
            "MA50": ma50,
            "MA200": ma200,
            "布林上轨": bb_high,
            "布林中轨": bb_mid,
            "布林下轨": bb_low,
            "ATR": atr,
            "PE_Ratio": pe,
        }

    @staticmethod
    def _fetch_pe(df: pd.DataFrame, close: pd.Series) -> Optional[float]:
        """延迟获取 PE，避免频繁调用 info 接口"""
        try:
            # 尝试从缓存或本地推断，避免每次都调 API
            ticker = getattr(df, "name", None)  # 如果 df 带 name
            if not ticker:
                return None
            info = yf.Ticker(ticker).info
            return info.get("trailingPE") or info.get("forwardPE")
        except Exception:
            return None


# ==================== 7. 宏观与 SOX 层 ====================
class MacroCollector:
    """宏观数据、FRED、PCR 等聚合"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.fetcher = DataFetcher(cfg)

    def collect_all(self) -> Dict[str, Any]:
        macro: Dict[str, Any] = {}

        # 基础宏观指标
        for name, sym in self.cfg.macro_indices.items():
            val = self._get_macro_value(sym)
            macro[name] = val if val is not None else "无数据"

        # FRED 数据
        if self.cfg.fred_api_key:
            macro.update(self._fetch_fred_batch())
        else:
            macro["美国国债规模"] = "未配置FRED Key"
            macro["芝加哥联储杠杆指数"] = "未配置FRED Key"
            macro["2年期实际利率（近似）"] = "未配置FRED Key"

        # PCR
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
            # 美国国债规模
            debt = fred.get_series("GFDEBTN")
            result["美国国债规模"] = float(debt.iloc[-1]) if not debt.empty else None
            # 杠杆指数
            lev = fred.get_series("NFCILEVERAGE")
            result["芝加哥联储杠杆指数"] = float(lev.iloc[-1]) if not lev.empty else None
            # 实际利率
            dgs2 = fred.get_series("DGS2")
            t5yie = fred.get_series("T5YIE")
            if not dgs2.empty and not t5yie.empty:
                result["2年期实际利率（近似）"] = f"{float(dgs2.iloc[-1] - t5yie.iloc[-1]):.2f}%"
        except Exception as e:
            logger.error(f"FRED 批量获取失败: {e}")
        return result

    def _get_put_call_ratio(self) -> Tuple[Optional[float], Optional[float]]:
        if not self.cfg.alpha_vantage_key:
            return None, None
        url = (
            "https://www.alphavantage.co/query"
            f"?function=PUT_CALL_RATIO&apikey={self.cfg.alpha_vantage_key}"
        )
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
            logger.warning(f"PCR 获取失败: {e}")
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

        ma20 = cls_safe_float(close.rolling(20).mean())
        ma50 = cls_safe_float(close.rolling(50).mean())
        ma200 = cls_safe_float(close.rolling(200).mean())

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        try:
            rsi_val = float(100 - (100 / (1 + rs.iloc[-1])))
        except Exception:
            rsi_val = 50.0

        # MACD
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd_line = exp12 - exp26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_val = float(macd_line.iloc[-1])
        macd_sig_val = float(macd_signal.iloc[-1])

        signals: List[str] = []
        if latest < self.cfg.sox_support_level:
            signals.append(f"⚠️ SOX 跌破{self.cfg.sox_support_level:.0f}关键支撑")
        else:
            signals.append(f"✅ SOX 站上{self.cfg.sox_support_level:.0f}支撑")

        if ma20 < ma50:
            signals.append("🔻 20/50日均线死叉")
        if rsi_val < self.cfg.rsi_oversold:
            signals.append("🟢 RSI超卖，可能反弹")
        elif rsi_val > self.cfg.rsi_overbought:
            signals.append("🔴 RSI超买，警惕回调")
        if drawdown < -20:
            signals.append(f"🐻 技术性熊市（回撤{drawdown:.1f}%）")
        if macd_val < macd_sig_val:
            signals.append("🔻 MACD卖出信号")

        return {
            "最新价": latest,
            "回撤": drawdown,
            "RSI": rsi_val,
            "MA20": ma20,
            "MA50": ma50,
            "MA200": ma200,
            "MACD": macd_val,
            "MACD信号": macd_sig_val,
            "支撑11200": latest > self.cfg.sox_support_level,
            "技术性熊市": drawdown < -20,
            "信号列表": signals,
        }


def cls_safe_float(series: pd.Series, default: float = 0.0) -> float:
    """模块级快捷函数"""
    try:
        return float(series.iloc[-1])
    except Exception:
        return default


# ==================== 8. 风险与情绪层 ====================
class RiskEngine:
    """杠杆、强平、情绪指数"""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @staticmethod
    def margin_call_price(entry: float, leverage: float, maintenance: float) -> Optional[float]:
        if entry <= 0 or leverage <= 1 or maintenance <= 0:
            return None
        return round(entry * (leverage - 1) / (leverage * (1 - maintenance)), 2)

    def leverage_analysis(self, data: StockTechData) -> Dict[str, Any]:
        price = data.get("收盘价", 0)
        bb_low = data.get("布林下轨", 0)
        atr = data.get("ATR", 0)
        rsi = data.get("RSI_14", 50)

        margin_calls = {}
        for lev in self.cfg.leverage_levels:
            mc = self.margin_call_price(price, lev, self.cfg.maintenance_margin)
            if mc:
                margin_calls[f"{lev}x"] = mc

        distance_pct = ((price - bb_low) / bb_low * 100) if bb_low > 0 else 0
        vol_effect = (
            "高" if price > 0 and atr / price > 0.05 else
            "中" if price > 0 and atr / price > 0.025 else
            "低" if price > 0 else "未知"
        )

        # 风险评分
        score = 0
        score += 40 if distance_pct < 2 else 20 if distance_pct < 5 else 5
        score += 30 if vol_effect == "高" else 15 if vol_effect == "中" else 0
        if rsi < 30 and distance_pct < 0:
            score += 20

        level = "高" if score >= 60 else "中" if score >= 35 else "低"
        return {
            "风险等级": level,
            "距布林下轨": round(distance_pct, 2),
            "波动率影响": vol_effect,
            "强平价格": margin_calls,
            "描述": f"当前价格距布林下轨 {distance_pct:.1f}%，波动率{vol_effect}，综合杠杆风险{level}",
        }

    @staticmethod
    def sentiment(vix: Any, vol_pcr: Any, sox_rsi: Any) -> SentimentResult:
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


# ==================== 9. 新闻与预警层 ====================
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
            logger.warning(f"Telegram 发送失败: {e}")

    def check_and_send(self, macro: Dict, stock_dict: Dict[str, Any], sox: SOXSignals) -> List[str]:
        alerts: List[str] = []
        if sox.get("技术性熊市"):
            alerts.append(f"🐻 SOX 技术性熊市（回撤 {sox.get('回撤', 0):.1f}%）")
        if sox.get("最新价", 0) < self.cfg.sox_support_level:
            alerts.append(f"⚠️ SOX 跌破 {self.cfg.sox_support_level:.0f}")

        for sym, data in stock_dict.items():
            # === 关键修复：跳过 None / 失败数据 ===
            if not data:
                logger.debug(f"{sym} 无数据，跳过预警检查")
                continue

            rsi = data.get("RSI_14", 50)
            if rsi > self.cfg.rsi_overbought:
                alerts.append(f"🔴 {sym} RSI={rsi:.1f} 超买")
            elif rsi < self.cfg.rsi_oversold:
                alerts.append(f"🟢 {sym} RSI={rsi:.1f} 超卖")

        vix_val = macro.get("VIX", 0)
        try:
            if float(vix_val) > self.cfg.vix_alert_threshold:
                alerts.append(f"🌪️ VIX 超过 {self.cfg.vix_alert_threshold}，当前 {vix_val}")
        except (TypeError, ValueError):
            pass

        if alerts:
            msg = "📢 <b>市场预警</b>\n" + "\n".join(alerts)
            self.send_telegram(msg)
        return alerts


class NewsFetcher:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def fetch_baidu(self, query: str) -> List[Dict]:
        if not self.cfg.serpapi_key:
            return []
        params = {
            "engine": "baidu_news",
            "q": query,
            "api_key": self.cfg.serpapi_key,
            "num": 3,
        }
        try:
            # 新版 serpapi 的正确导入路径
            from serpapi.google_search import GoogleSearch
            results = GoogleSearch(params).get_dict()
            return results.get("news_results", [])
        except Exception as e:
            logger.warning(f"百度新闻获取失败 ({query}): {e}")
            return []

# ==================== 10. AI 报告层 ====================
class AIReportGenerator:
    """DeepSeek / OpenAI 接口封装"""

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
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"AI 调用失败: {e}")
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
            # 兜底：找第一个 [ 和最后一个 ]
            start, end = raw.find("["), raw.rfind("]")
            if start != -1 and end != -1 and end > start:
                return json.loads(raw[start:end + 1])
            raise

    def generate_market_overview(
        self,
        macro: Dict,
        sox: SOXSignals,
        sentiment: SentimentResult,
        adv_dec: Optional[float],
        news: Dict[str, List[Dict]],
    ) -> Optional[str]:
        prompt = f"""你是一位专业股票分析师，请根据以下数据生成一份简洁的大盘总览。
报告日期：{datetime.now().strftime('%Y-%m-%d %H:%M')}
---
### 宏观概览
{json.dumps(macro, indent=2, ensure_ascii=False)}
涨跌家数比（SPY涨跌幅近似）：{adv_dec if adv_dec is not None else '无数据'}%
---
### SOX 指数信号
- 最新价：{sox.get('最新价', 'N/A')}
- 回撤：{sox.get('回撤', 'N/A')}%
- 技术性熊市：{'是' if sox.get('技术性熊市') else '否'}
- RSI(14)：{sox.get('RSI', 'N/A')}
- MA20：{sox.get('MA20', 'N/A')}
关键信号：{chr(10).join(['- ' + s for s in sox.get('信号列表', [])])}
---
### 市场情绪
情绪指数：{sentiment['情绪指数']} （{sentiment['标签']}）
---
### 个股新闻摘要
{json.dumps(news, indent=2, ensure_ascii=False) if news else '无新闻数据'}
---
要求：1.宏观判断 2.SOX解读 3.情绪与资金 4.今日整体操作建议 5.风险提示。
总字数500字以内，专业简洁，使用emoji。个股具体买卖点位不需要展开。
"""
        return self._call(prompt, max_tokens=1200)

    def generate_decision_cards(
        self,
        stock_dict: Dict[str, StockTechData],
        macro: Dict,
        sox: SOXSignals,
    ) -> Optional[List[Dict]]:
        valid = {s: d for s, d in stock_dict.items() if d}
        if not valid:
            logger.warning("无有效个股数据，跳过决策卡片")
            return None

        facts = []
        for sym, data in valid.items():
            market = "港股" if sym.endswith(".HK") else "美股"
            mc = RiskEngine(self.cfg).leverage_analysis(data)["强平价格"]
            mc_str = "；".join([f"{k}强平${v:.2f}" for k, v in mc.items()])

            facts.append(f"""
{sym}（{market}）
收盘 {data.get('收盘价', 0):.2f}，涨跌幅 {data.get('涨跌幅', 0):.2f}%，量比{data.get('量比状态', 'N/A')}
RSI {data.get('RSI_14', 0):.2f}，MACD {data.get('MACD', 0):.3f}，PE {data.get('PE_Ratio', 'N/A')}
MA5 {data.get('MA5', 0):.2f} / MA20 {data.get('MA20', 0):.2f} / MA50 {data.get('MA50', 0):.2f}
布林带：上轨{data.get('布林上轨', 0):.2f} 中轨{data.get('布林中轨', 0):.2f} 下轨{data.get('布林下轨', 0):.2f}
ATR {data.get('ATR', 0):.2f}
斩杀线：{mc_str if mc_str else 'N/A'}
""")

        prompt = f"""你是一位专业的美股/港股分析师。请针对以下每只股票，生成结构化决策卡片。
市场背景：VIX={macro.get('VIX', 'N/A')}，标普500={macro.get('标普500', 'N/A')}，SOX回撤={sox.get('回撤', 'N/A')}%

个股数据：
{''.join(facts)}

请严格只输出一个 JSON 数组，不要任何前后缀说明文字、不要 markdown 代码块标记。数组每个元素对应一只股票，字段如下：
[
  {{
    "symbol": "股票代码（须与上面给出的代码完全一致，如 0700.HK）",
    "score": 0到100的整数评分,
    "operation": "买入 或 观望 或 卖出",
    "trend": "看多 或 看空 或 震荡",
    "core_view": "50字以内核心判断",
    "catalysts": ["利好催化1", "利好催化2"],
    "risks": ["风险点1", "风险点2"],
    "sniper": {{
      "ideal_buy": "理想买入价位描述",
      "second_buy": "二次加仓/回调买入位描述",
      "stop_loss": "止损位描述",
      "target": "止盈目标描述"
    }},
    "sectors": ["相关板块1", "相关板块2"]
  }}
]
"""
        raw = self._call(prompt, system="你是专业金融分析师，只输出严格合法的 JSON 数组，不输出任何其他文字。", max_tokens=3000)
        if not raw:
            return None
        try:
            return self.extract_json(raw)
        except Exception as e:
            logger.error(f"决策卡片 JSON 解析失败: {e}")
            return None


# ==================== 11. 主控层 ====================
class ReportOrchestrator:
    """编排整个报告生成流程"""

    def __init__(self):
        self.cfg = Config()
        self.fetcher = DataFetcher(self.cfg)
        self.macro_collector = MacroCollector(self.cfg)
        self.risk_engine = RiskEngine(self.cfg)
        self.alert_svc = AlertService(self.cfg)
        self.news_fetcher = NewsFetcher(self.cfg)
        self.ai = AIReportGenerator(self.cfg)

    def _fetch_stock_parallel(self) -> Dict[str, StockTechData]:
        """并发获取所有股票技术指标"""
        results: Dict[str, StockTechData] = {}

        def worker(sym: str) -> Tuple[str, Optional[StockTechData]]:
            try:
                df = self.fetcher.get_stock_df(sym)
                data = TechnicalAnalyzer.analyze(df, self.cfg)
                if data:
                    data["symbol"] = sym
                return sym, data
            except Exception as e:
                logger.error(f"获取 {sym} 失败: {e}")
                return sym, None

        # 港股数据源（akshare）有线程安全问题倾向，降低并发或单独处理
        hk_stocks = [s for s in self.cfg.stocks if s.endswith(".HK")]
        us_stocks = [s for s in self.cfg.stocks if not s.endswith(".HK")]

        # US stocks 并发
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_map = {executor.submit(worker, s): s for s in us_stocks}
            for future in as_completed(future_map):
                sym, data = future.result()
                results[sym] = data

        # HK stocks 串行（akshare 底层可能依赖全局状态）
        for sym in hk_stocks:
            sym, data = worker(sym)
            results[sym] = data

        return results

    def _persist(self, macro: Dict, stock_dict: Dict, sox: SOXSignals, cards: Optional[List], report_md: Optional[str]):
        """统一持久化"""
        out = self.cfg.output_dir

        # 宏观
        pd.DataFrame([macro]).to_csv(out / "macro.csv", index=False)

        # 个股
        records = []
        for sym, data in stock_dict.items():
            if not data:
                continue
            row = {"symbol": sym, **data}
            risk = self.risk_engine.leverage_analysis(data)
            row["杠杆风险"] = risk["风险等级"]
            for k, v in risk["强平价格"].items():
                row[f"强平价格_{k}"] = v
            records.append(row)
        if records:
            pd.DataFrame(records).to_csv(out / "stocks.csv", index=False)

        # SOX
        sox_row = {k: v for k, v in sox.items() if k != "信号列表"}
        sox_row["信号列表"] = "；".join(sox.get("信号列表", []))
        pd.DataFrame([sox_row]).to_csv(out / "sox.csv", index=False)

        # AI 报告
        if report_md:
            (out / "report.md").write_text(report_md, encoding="utf-8")

        # 决策卡片
        if cards:
            payload = {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "stocks": cards,
            }
            (out / "cards.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def run(self):
        logger.info("📊 报告生成流程启动")
        # 1. 宏观
        macro = self.macro_collector.collect_all()
        logger.info("宏观数据采集完成")

        # 2. 个股（并发）
        stock_dict = self._fetch_stock_parallel()
        valid_count = sum(1 for v in stock_dict.values() if v)
        logger.info(f"个股数据采集完成，有效 {valid_count}/{len(self.cfg.stocks)}")

        # 3. SOX
        sox = self.macro_collector.get_sox()
        logger.info("SOX 信号采集完成")

        # 4. 涨跌家数比
        try:
            spy_df = self.fetcher.fetch_yf("SPY", period="2d")
            adv_dec = float((spy_df["Close"].iloc[-1] - spy_df["Close"].iloc[-2]) / spy_df["Close"].iloc[-2] * 100)
        except Exception:
            adv_dec = None

        # 5. 情绪
        sentiment = self.risk_engine.sentiment(
            macro.get("VIX", 0),
            macro.get("Volume PCR"),
            sox.get("RSI", 50),
        )
        logger.info(f"市场情绪: {sentiment['情绪指数']} ({sentiment['标签']})")

        # 6. 新闻
        news = {}
        if self.cfg.serpapi_key:
            for sym in self.cfg.stocks:
                news[sym] = self.news_fetcher.fetch_baidu(sym.replace(".HK", ""))

        # 7. 预警
        alerts = self.alert_svc.check_and_send(macro, stock_dict, sox)
        if alerts:
            logger.info(f"触发 {len(alerts)} 条预警")

        # 8. AI 生成
        report_md = None
        cards = None
        if self.ai.client:
            report_md = self.ai.generate_market_overview(macro, sox, sentiment, adv_dec, news)
            if report_md:
                logger.info("大盘总览报告生成完成")
            cards = self.ai.generate_decision_cards(stock_dict, macro, sox)
            if cards:
                logger.info(f"决策卡片生成完成，共 {len(cards)} 只")
        else:
            logger.info("未配置 DeepSeek API Key，跳过 AI 报告")

        # 9. 持久化
        self._persist(macro, stock_dict, sox, cards, report_md)
        logger.info("✅ 全部数据已保存，流程结束")


if __name__ == "__main__":
    ReportOrchestrator().run()
