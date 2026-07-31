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
def bottom_fishing_score(data, poc_price, current_price):
    """综合评分：越高越接近抄底区间，返回 0-100"""
    score = 0
    reasons = []
    if data["RSI(14)"] < 30:
        score += 30; reasons.append("RSI超卖")
    elif data["RSI(14)"] < 40:
        score += 15; reasons.append("RSI偏低")
    if data["收盘价"] <= data["布林下轨"] * 1.02:
        score += 25; reasons.append("触及/逼近布林下轨")
    if data["量比状态"] == "缩量":
        score += 15; reasons.append("恐慌盘衰竭（缩量企稳）")
    dist_to_poc = abs(current_price - poc_price) / poc_price * 100
    if dist_to_poc < 3:
        score += 30; reasons.append(f"接近资金集中区（{poc_price}，主力成本支撑）")
    return {"抄底评分": score, "依据": reasons}
def rebound_or_reversal(data):
    """判断价格上涨是短线反弹还是趋势反转"""
    bullish_alignment = data["MA5"] > data["MA20"] > data["MA50"]  # 多头排列
    above_ma50 = data["收盘价"] > data["MA50"]
    volume_confirm = data["量比状态"] == "放量"
    macd_golden = data["MACD"] > data["MACD信号"]

    signals_met = sum([bullish_alignment, above_ma50, volume_confirm, macd_golden])
    if signals_met >= 3:
        return "反转（趋势性）：均线多头排列+放量+MACD金叉，非单纯超跌反弹"
    elif data["涨跌幅%"] > 0 and data["RSI(14)"] < 50:
        return "反弹（阶段性）：仅超跌修复，尚未突破关键均线结构，警惕冲高回落"
    else:
        return "震荡：暂无明确方向信号"
def market_sentiment_index(vix, vol_pcr, sox_rsi):
    """0-100，越高越贪婪，越低越恐惧（仿 Fear & Greed 简化版）"""
    vix_score = max(0, min(100, 100 - (vix - 12) * 4))       # VIX 12→满分，VIX 37→0分
    pcr_score = max(0, min(100, (vol_pcr - 0.5) * 100)) if vol_pcr else 50  # PCR越高越恐慌，需反向：这里简化，PCR>1时偏恐慌
    pcr_score = max(0, min(100, 100 - (vol_pcr - 0.5) * 80)) if vol_pcr else 50
    rsi_score = sox_rsi if sox_rsi else 50
    composite = round(vix_score * 0.4 + pcr_score * 0.3 + rsi_score * 0.3, 1)
    label = "极度贪婪" if composite > 75 else "贪婪" if composite > 55 else "中性" if composite > 45 else "恐惧" if composite > 25 else "极度恐惧"
    return {"情绪指数": composite, "标签": label}
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

# ---------- 决策卡片 JSON 生成（仿 daily_stock_analysis 决策仪表盘风格） ----------
def parse_json_response(raw: str):
    """兜底解析 LLM 返回的 JSON，去掉可能的 markdown 围栏和多余文字"""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    # 尝试直接解析；失败则截取第一个 [ 到最后一个 ] 之间的内容再试一次
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise
LEVERAGE_LEVELS = (1.5, 2)  # 可按你实际常用杠杆调整

def generate_leverage_risk(stock_dict):
    risk_data = {}
    for sym, data in stock_dict.items():
        if "error" not in data:
            risk_data[sym] = stock_leverage_warning(data, leverage_levels=LEVERAGE_LEVELS)
    with open("data/leverage_risk.json", "w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.now().strftime('%Y-%m-%d %H:%M'), "stocks": risk_data},
                  f, ensure_ascii=False, indent=2)
    print("✅ 个股杠杆强平线已计算")
def generate_decision_cards(client, stock_dict, macro, sox):
    """针对每只（港股/美股）股票生成结构化决策卡片，保存到 data/cards.json"""
    valid_stocks = {s: d for s, d in stock_dict.items() if "error" not in d}
    if not valid_stocks:
        print("⚠️ 无有效个股数据，跳过决策卡片生成")
        return

    stock_facts = []
    for sym, data in valid_stocks.items():
        market = "港股" if sym.endswith(".HK") else "美股"
        stock_facts.append(f"""
{sym}（{market}）
收盘 {data.get('收盘价', 0):.2f}，涨跌幅 {data.get('涨跌幅%', 0):.2f}%，量比{data.get('量比状态', 'N/A')}
RSI {data.get('RSI(14)', 0):.2f}，MACD {data.get('MACD', 0):.3f}，PE {data.get('PE Ratio', 'N/A')}
MA5 {data.get('MA5', 0):.2f} / MA20 {data.get('MA20', 0):.2f} / MA50 {data.get('MA50', 0):.2f}
布林带：上轨{data.get('布林上轨', 0):.2f} 中轨{data.get('布林中轨', 0):.2f} 下轨{data.get('布林下轨', 0):.2f}
ATR {data.get('ATR', 0):.2f}
""")

    prompt = f"""你是一位专业的美股/港股分析师。请针对以下每只股票，生成结构化决策卡片。
市场背景：VIX={macro.get('VIX', 'N/A')}，标普500={macro.get('标普500', 'N/A')}，SOX回撤={sox.get('回撤%', 'N/A')}%

个股数据：
{"".join(stock_facts)}

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
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是专业金融分析师，只输出严格合法的 JSON 数组，不输出任何其他文字。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=3000
        )
        raw = response.choices[0].message.content
        cards = parse_json_response(raw)
        with open("data/cards.json", "w", encoding="utf-8") as f:
            json.dump(
                {"generated_at": datetime.now().strftime('%Y-%m-%d %H:%M'), "stocks": cards},
                f, ensure_ascii=False, indent=2
            )
        print(f"✅ 决策卡片已保存，共 {len(cards)} 只股票")
    except Exception as e:
        print(f"❌ 决策卡片生成失败: {e}")

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

    # AI 分析：整体大盘总览（保留原有 Markdown 报告） + 个股结构化决策卡片
    if DEEPSEEK_API_KEY:
        from openai import OpenAI
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")

        # 1) 整体大盘总览 Markdown（原有逻辑，作为顶部综述文字保留）
        print("🧠 正在生成大盘总览报告...")
        prompt = f"""你是一位专业股票分析师，请根据以下数据生成一份简洁的大盘总览。
报告日期：{datetime.now().strftime('%Y-%m-%d %H:%M')}
---
### 宏观概览
{json.dumps(macro, indent=2, ensure_ascii=False)}
涨跌家数比（SPY涨跌幅近似）：{adv_dec}%
---
### SOX 指数信号
- 最新价：{sox.get('最新价','N/A')}
- 回撤：{sox.get('回撤%','N/A')}%
- 技术性熊市：{'是' if sox.get('技术性熊市') else '否'}
- RSI(14)：{sox.get('RSI','N/A')}
- MA20：{sox.get('MA20','N/A')}
关键信号：{chr(10).join(['- '+s for s in sox.get('信号列表',[])])}
---
### 任务要求
生成包含：1.宏观判断 2.SOX解读 3.情绪与资金 4.今日整体操作建议 5.风险提示
总字数控制在500字以内，专业、简洁，使用emoji。个股层面的具体买卖点位不需要在这里展开（会在下方决策卡片中单独呈现）。
"""
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": "专业金融分析师"}, {"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            report = response.choices[0].message.content
            with open("data/report.md", "w", encoding="utf-8") as f:
                f.write(report)
            print("✅ 大盘总览报告已保存")
        except Exception as e:
            print(f"❌ 大盘总览报告生成失败: {e}")

        # 2) 个股结构化决策卡片（仿 daily_stock_analysis 决策仪表盘风格）
        print("🧠 正在生成个股决策卡片...")
        generate_decision_cards(client, stock_dict, macro, sox)
    else:
        print("⏭️ 未配置DEEPSEEK_API_KEY，跳过AI报告")

    print("🎯 数据采集完成！")

if __name__ == "__main__":
    generate_report()
