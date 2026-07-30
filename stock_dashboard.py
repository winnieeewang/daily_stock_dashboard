import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import ta
import json
import warnings
import os
import requests
from fredapi import Fred

warnings.filterwarnings('ignore')

# ---------- 配置 ----------
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

STOCKS = [
    "MU", "AAOI", "GOOGL", "MSFT", "AMZN", "MRVL", "LITE",
    "SNDK", "NVDA", "ORCL", "SPCX", "SKHY", "TSLA",
    "0700.HK", "0883.HK", "3750.HK"
]

MACRO_INDICES = {
    "VIX": "^VIX",
    "美元指数": "DX-Y.NYB",
    "标普500": "^GSPC",
    "纳斯达克100": "^NDX",
    "黄金": "GC=F",
    "WTI原油": "CL=F",
    "10年期美债收益率": "^TNX",
}

LOOKBACK_DAYS = 60

# ---------- 工具函数 ----------
def safe_fetch(ticker, period="2mo"):
    try:
        data = yf.download(ticker, period=period, progress=False)
        if not data.empty:
            return data
    except:
        pass
    return pd.DataFrame()

def get_macro_value(symbol):
    data = safe_fetch(symbol, period="5d")
    if data.empty:
        return None
    val = data['Close'].iloc[-1]
    return float(val) if isinstance(val, (np.floating, np.integer)) else val

def get_advance_decline():
    spy = safe_fetch("SPY", period="2d")
    if len(spy) < 2:
        return None
    change = (spy['Close'].iloc[-1] - spy['Close'].iloc[-2]) / spy['Close'].iloc[-2] * 100
    return float(change)

# ---------- FRED ----------
def get_fred_data(series_id):
    if not FRED_API_KEY:
        print(f"⚠️ FRED API Key 未配置，跳过 {series_id}")
        return None
    try:
        fred = Fred(api_key=FRED_API_KEY)
        series = fred.get_series(series_id)
        if not series.empty:
            val = float(series.iloc[-1])
            print(f"✅ FRED {series_id}: {val}")
            return val
        else:
            print(f"⚠️ FRED {series_id} 空数据")
            return None
    except Exception as e:
        print(f"❌ FRED {series_id} 失败: {e}")
        return None

def get_finra_margin_debt():
    try:
        if not FRED_API_KEY:
            return None
        fred = Fred(api_key=FRED_API_KEY)
        nfci = fred.get_series("NFCI")
        if not nfci.empty:
            val = float(nfci.iloc[-1])
            print(f"✅ FINRA 近似 (NFCI): {val}")
            return val
    except:
        pass
    return None

# ---------- Put-Call Ratio ----------
def get_put_call_ratio():
    if not ALPHA_VANTAGE_KEY:
        return None, None
    try:
        url = f"https://www.alphavantage.co/query?function=PUT_CALL_RATIO&apikey={ALPHA_VANTAGE_KEY}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if "data" in data and data["data"]:
            latest = data["data"][-1]
            vol_pcr = float(latest.get("volume_put_call_ratio", 0))
            oi_pcr = float(latest.get("open_interest_put_call_ratio", 0))
            return vol_pcr, oi_pcr
    except:
        pass
    return None, None

# ---------- 港股数据 ----------
def get_hk_data(symbol):
    try:
        import akshare as ak
        code = symbol.replace('.HK', '').zfill(5)
        df = ak.stock_hk_daily(symbol=code, adjust="qfq")
        if not df.empty:
            # 列名映射（同之前）
            rename_map = {'日期':'Date','开盘':'Open','收盘':'Close','最高':'High','最低':'Low','成交量':'Volume'}
            df.rename(columns=rename_map, inplace=True)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            elif not isinstance(df.index, pd.DatetimeIndex):
                return pd.DataFrame()
            required = ['Open','High','Low','Close','Volume']
            for col in required:
                if col not in df.columns:
                    return pd.DataFrame()
            df = df[required]
            for col in required:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(inplace=True)
            df.sort_index(inplace=True)
            return df
    except:
        pass
    return safe_fetch(symbol, period=f"{LOOKBACK_DAYS}d")

# ---------- 个股技术 ----------
def get_stock_technical(symbol):
    is_hk = symbol.endswith(".HK")
    if is_hk:
        df = get_hk_data(symbol)
    else:
        df = safe_fetch(symbol, period=f"{LOOKBACK_DAYS}d")
    if df.empty or len(df) < 20:
        return None
    try:
        close = df['Close'].astype(float)
        high = df['High'].astype(float)
        low = df['Low'].astype(float)
        volume = df['Volume'].astype(float)
    except:
        return None
    try:
        latest_close = float(close.iloc[-1])
        latest_volume = float(volume.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close)>1 else latest_close
    except:
        return None
    change = (latest_close - prev_close) / prev_close * 100 if prev_close != 0 else 0
    # 成交量状态
    vol_ma5 = volume.rolling(5).mean()
    try:
        ma5_vol = float(vol_ma5.iloc[-1])
    except:
        ma5_vol = 0.0
    if ma5_vol > 0:
        if latest_volume > ma5_vol * 1.2:
            vol_status = "放量"
        elif latest_volume < ma5_vol * 0.8:
            vol_status = "缩量"
        else:
            vol_status = "持平"
    else:
        vol_status = "持平"
    # 技术指标
    try:
        rsi = float(ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1])
    except:
        rsi = 50.0
    try:
        macd_obj = ta.trend.MACD(close)
        macd_line = float(macd_obj.macd().iloc[-1])
        macd_signal = float(macd_obj.macd_signal().iloc[-1])
        macd_diff = float(macd_obj.macd_diff().iloc[-1])
    except:
        macd_line = macd_signal = macd_diff = 0.0
    try:
        ma5 = float(close.rolling(5).mean().iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])
    except:
        ma5 = ma20 = ma50 = ma200 = 0.0
    try:
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        bb_high = float(bb.bollinger_hband().iloc[-1])
        bb_mid = float(bb.bollinger_mavg().iloc[-1])
        bb_low = float(bb.bollinger_lband().iloc[-1])
    except:
        bb_high = bb_mid = bb_low = 0.0
    try:
        atr = float(ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1])
    except:
        atr = 0.0
    # PE
    pe = None
    try:
        ticker_obj = yf.Ticker(symbol)
        info = ticker_obj.info
        if info and 'trailingPE' in info:
            pe = float(info['trailingPE'])
        elif info and 'forwardPE' in info:
            pe = float(info['forwardPE'])
    except:
        pass
    return {
        "收盘价": latest_close,
        "涨跌幅%": change,
        "成交量": int(latest_volume),
        "量比状态": vol_status,
        "RSI(14)": rsi,
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
        "PE Ratio": pe,
    }

# ---------- SOX ----------
def get_sox_signals():
    sox = safe_fetch("^SOX", period="3mo")
    if sox.empty:
        return {"error": "无法获取 SOX 数据"}
    close = sox['Close'].astype(float)
    latest_close = float(close.iloc[-1])
    peak = float(close.max())
    drawdown = (latest_close - peak) / peak * 100
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma21 = float(close.rolling(21).mean().iloc[-1])
    ma40 = float(close.rolling(40).mean().iloc[-1])
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
    support_11200 = latest_close > 11200
    bear_market = drawdown < -20
    signals = []
    if not support_11200:
        signals.append("⚠️ SOX 跌破11200关键支撑")
    else:
        signals.append("✅ SOX 站上11200支撑")
    if ma21 < ma50:
        signals.append("🔻 21日均线下穿50日均线（死叉）")
    if ma20 < ma40:
        signals.append("🔻 20日均线下穿40日均线")
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
        "支撑11200": support_11200,
        "技术性熊市": bear_market,
        "信号列表": signals,
    }

# ---------- 预警 ----------
def send_alert(message):
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
            requests.post(url, json=payload, timeout=10)
        except:
            pass

def check_alerts(macro, stock_dict, sox):
    alerts = []
    if sox.get('技术性熊市'):
        alerts.append(f"🐻 SOX 已进入技术性熊市（回撤 {sox.get('回撤%',0):.1f}%）")
    if sox.get('最新价', 0) < 11200:
        alerts.append("⚠️ SOX 跌破 11200 支撑位")
    for sym, data in stock_dict.items():
        if "error" not in data:
            rsi = data.get('RSI(14)', 50)
            if rsi > 70:
                alerts.append(f"🔴 {sym} RSI={rsi:.1f} 超买")
            elif rsi < 30:
                alerts.append(f"🟢 {sym} RSI={rsi:.1f} 超卖")
    if macro.get('VIX', 0) > 25:
        alerts.append(f"🌪️ VIX 超过 25，当前 {macro['VIX']}")
    if alerts:
        msg = "📢 <b>市场预警</b>\n" + "\n".join(alerts)
        send_alert(msg)
    return alerts

# ---------- 主函数 ----------
def generate_report():
    print("📊 数据采集开始...")
    # 宏观
    macro = {}
    for name, sym in MACRO_INDICES.items():
        val = get_macro_value(sym)
        macro[name] = val if val is not None else "无数据"
    # FRED
    if FRED_API_KEY:
        macro["美国国债规模"] = get_fred_data("GFDEBTN") or "无数据"
        macro["芝加哥联储杠杆指数"] = get_fred_data("NFCI") or "无数据"
        macro["2年期实际利率"] = get_fred_data("TIPS2Y") or "无数据"
    else:
        macro["美国国债规模"] = "未配置FRED Key"
        macro["芝加哥联储杠杆指数"] = "未配置FRED Key"
        macro["2年期实际利率"] = "未配置FRED Key"
    margin_debt = get_finra_margin_debt()
    macro["FINRA保证金债务"] = margin_debt if margin_debt else "无数据"
    vol_pcr, oi_pcr = get_put_call_ratio()
    macro["Volume PCR"] = vol_pcr if vol_pcr else "无数据"
    macro["OI PCR"] = oi_pcr if oi_pcr else "无数据"
    # 个股
    stock_dict = {}
    for sym in STOCKS:
        tech = get_stock_technical(sym)
        if tech:
            stock_dict[sym] = tech
        else:
            stock_dict[sym] = {"error": "数据获取失败"}
            print(f"⚠️ {sym} 数据获取失败，已跳过")
    # SOX
    sox = get_sox_signals()
    if "error" in sox:
        sox = {"最新价":"N/A","回撤%":"N/A","技术性熊市":False,"RSI":"N/A","MA20":"N/A","MA50":"N/A","MA200":"N/A","信号列表":["无法获取"]}
    adv_dec = get_advance_decline()
    adv_dec = float(adv_dec) if adv_dec is not None else "无数据"
    # 保存 CSV
    os.makedirs("data", exist_ok=True)
    macro_df = pd.DataFrame([macro])
    macro_df.to_csv("data/macro.csv", index=False)
    print("✅ 宏观数据已保存")
    stock_records = []
    for sym, data in stock_dict.items():
        if "error" not in data:
            row = {"symbol": sym}
            row.update(data)
            stock_records.append(row)
    if stock_records:
        pd.DataFrame(stock_records).to_csv("data/stocks.csv", index=False)
        print("✅ 个股数据已保存")
    sox_row = {k:v for k,v in sox.items() if k != "信号列表"}
    sox_row["信号列表"] = "；".join(sox.get("信号列表", []))
    pd.DataFrame([sox_row]).to_csv("data/sox.csv", index=False)
    print("✅ SOX 数据已保存")
    # 预警
    alerts = check_alerts(macro, stock_dict, sox)
    if alerts:
        print("📢 触发预警:", alerts)
    # AI 报告
    if DEEPSEEK_API_KEY:
        print("🧠 正在生成 AI 报告...")
        from openai import OpenAI
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
        prompt = f"""你是一位专业股票分析师，请根据以下数据生成一份详细的每日投资看板报告。
报告日期：{datetime.now().strftime('%Y-%m-%d %H:%M')}
---
### 一、宏观概览
{json.dumps(macro, indent=2, ensure_ascii=False)}
涨跌家数比（SPY涨跌幅近似）：{adv_dec}%
---
### 二、SOX 指数信号
- 最新价：{sox.get('最新价','N/A')}
- 回撤：{sox.get('回撤%','N/A')}%
- 技术性熊市：{'是' if sox.get('技术性熊市') else '否'}
- RSI(14)：{sox.get('RSI','N/A')}
- MA20：{sox.get('MA20','N/A')}
关键信号：{chr(10).join(['- '+s for s in sox.get('信号列表',[])])}
---
### 三、个股技术面
"""
        for sym, data in stock_dict.items():
            if "error" not in data:
                market = "港股" if sym.endswith(".HK") else "美股"
                prompt += f"""
**{sym} ({market})**
- 收盘: ${data.get('收盘价','N/A'):.2f}  涨跌幅: {data.get('涨跌幅%','N/A'):.2f}%
- 量比: {data.get('量比状态','N/A')}  PE: {data.get('PE Ratio','N/A')}
- RSI: {data.get('RSI(14)','N/A'):.2f}  MACD: {data.get('MACD','N/A'):.3f}
- MA20: {data.get('MA20','N/A'):.2f}  MA50: {data.get('MA50','N/A'):.2f}
"""
        prompt += """
---
### 四、任务要求
生成包含：1.宏观判断 2.SOX解读 3.个股技术分析 4.情绪与资金 5.操作建议 6.风险提示
总字数控制在1200字以内，专业、简洁，使用emoji。
"""
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": "专业金融分析师"},{"role":"user","content":prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            report = response.choices[0].message.content
            with open("data/report.md", "w", encoding="utf-8") as f:
                f.write(report)
            print("✅ AI报告已保存")
        except Exception as e:
            print(f"❌ AI报告生成失败: {e}")
    else:
        print("⏭️ 未配置DEEPSEEK_API_KEY，跳过AI报告")
    print("🎯 数据采集完成！")

if __name__ == "__main__":
    generate_report()
