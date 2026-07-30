# ====================================================================
# 每日股票看板 - DeepSeek 版本（完整可用）
# 支持港股/美股 + SOX 信号 + 技术指标 + K线图
# ====================================================================



# 2. 导入库
import yfinance as yf
import pandas as pd
import numpy as np
from openai import OpenAI
import requests
from datetime import datetime, timedelta
import ta
import json
import warnings
import os
import matplotlib.pyplot as plt
import mplfinance as mpf

warnings.filterwarnings('ignore')

try:
    import akshare as ak
    print("✅ AKShare 导入成功")
except ImportError:
    ak = None
    print("⚠️ AKShare 未安装")

# ====================================================================
# 配置区（请务必修改）
# ====================================================================

# 【必填】你的 DeepSeek API Key
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    # 如果在本地 Colab 运行，可以临时写在这里（不要提交到 GitHub）
    DEEPSEEK_API_KEY = "sk-e9823add77db40bb85ef993f4d338a9b"  # ← 替换成你的

# 初始化 DeepSeek Client
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)
print("✅ DeepSeek Client 已初始化")

# 股票列表（港股必须加 .HK）
STOCKS = [
    "MU", "AAOI", "GOOGL", "MSFT", "AMZN", "MRVL", "LITE", "SNDK", "NVDA",
    "0700.HK",   # 腾讯
    "0883.HK",   # 中国海洋石油
]

MACRO_INDICES = {
    "VIX": "^VIX",
    "美元指数": "DX-Y.NYB",
    "人民币汇率": "CNY=X",
    "标普500": "^GSPC",
    "纳斯达克100": "^NDX",
    "日经225": "^N225",
    "恒生指数": "^HSI",
    "黄金": "GC=F",
    "WTI原油": "CL=F",
    "10年期美债收益率": "^TNX",
}

LOOKBACK_DAYS = 60

# ====================================================================
# 数据获取函数
# ====================================================================

def safe_fetch(ticker, period="2mo"):
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            return pd.DataFrame()
        return data
    except Exception as e:
        print(f"⚠️ yfinance 获取 {ticker} 失败: {e}")
        return pd.DataFrame()

def get_macro_value(symbol):
    data = safe_fetch(symbol, period="5d")
    if data.empty:
        return None
    val = data['Close'].iloc[-1]
    if isinstance(val, pd.Series):
        val = val.iloc[0]
    return float(val) if isinstance(val, (np.floating, np.integer)) else val

def get_advance_decline():
    spy = safe_fetch("SPY", period="2d")
    if len(spy) < 2:
        return None
    change = (spy['Close'].iloc[-1] - spy['Close'].iloc[-2]) / spy['Close'].iloc[-2] * 100
    return float(change)

def fetch_fred(series_id, api_key):
    if not api_key:
        return None
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'sort_order': 'desc',
        'limit': 1
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if 'observations' in data and data['observations']:
            val = data['observations'][0]['value']
            if val and val != '.':
                return float(val)
    except:
        pass
    return None

# ====================================================================
# 港股数据获取（AKShare + yfinance 回退）
# ====================================================================

def get_hk_data(symbol):
    if ak is not None:
        code = symbol.replace('.HK', '').zfill(5)
        try:
            df = ak.stock_hk_daily(symbol=code, adjust="qfq")
            if not df.empty:
                rename_map = {}
                if '日期' in df.columns:    rename_map['日期'] = 'Date'
                if '开盘' in df.columns:    rename_map['开盘'] = 'Open'
                if '收盘' in df.columns:    rename_map['收盘'] = 'Close'
                if '最高' in df.columns:    rename_map['最高'] = 'High'
                if '最低' in df.columns:    rename_map['最低'] = 'Low'
                if '成交量' in df.columns:  rename_map['成交量'] = 'Volume'
                if 'date' in df.columns:    rename_map['date'] = 'Date'
                if 'open' in df.columns:    rename_map['open'] = 'Open'
                if 'close' in df.columns:   rename_map['close'] = 'Close'
                if 'high' in df.columns:    rename_map['high'] = 'High'
                if 'low' in df.columns:     rename_map['low'] = 'Low'
                if 'volume' in df.columns:  rename_map['volume'] = 'Volume'
                df.rename(columns=rename_map, inplace=True)
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                elif not isinstance(df.index, pd.DatetimeIndex):
                    return pd.DataFrame()
                required = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required:
                    if col not in df.columns:
                        return pd.DataFrame()
                df = df[required]
                for col in required:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df.dropna(inplace=True)
                df.sort_index(inplace=True)
                return df
        except Exception as e:
            print(f"⚠️ AKShare 获取 {symbol} 失败: {e}")
    return safe_fetch(symbol, period=f"{LOOKBACK_DAYS}d")

# ====================================================================
# 个股技术指标计算
# ====================================================================

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
    except Exception as e:
        print(f"⚠️ {symbol} 数据转换失败: {e}")
        return None
    
    try:
        latest_close = float(close.iloc[-1])
        latest_volume = float(volume.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) > 1 else latest_close
    except Exception as e:
        print(f"⚠️ {symbol} 提取最新值失败: {e}")
        return None
    
    change = (latest_close - prev_close) / prev_close * 100 if prev_close != 0 else 0
    
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
    }

# ====================================================================
# SOX 指数信号
# ====================================================================

def get_sox_signals():
    sox = safe_fetch("^SOX", period="3mo")
    if sox.empty:
        return {"error": "无法获取 SOX 数据"}
    
    close = sox['Close'].astype(float)
    latest_close = float(close.iloc[-1])
    peak = float(close.max())
    drawdown = (latest_close - peak) / peak * 100
    
    try:
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma21 = float(close.rolling(21).mean().iloc[-1])
        ma40 = float(close.rolling(40).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])
    except:
        ma20 = ma21 = ma40 = ma50 = ma200 = 0.0
    
    try:
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi_val = float(100 - (100 / (1 + rs.iloc[-1]))) if rs.iloc[-1] != 0 else 50.0
    except:
        rsi_val = 50.0
    
    try:
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd_line = exp12 - exp26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_val = float(macd_line.iloc[-1])
        macd_sig_val = float(macd_signal.iloc[-1])
    except:
        macd_val = macd_sig_val = 0.0
    
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

# ====================================================================
# 新闻（模拟）
# ====================================================================

def fetch_news(symbols, limit=3):
    news_list = []
    for sym in symbols[:limit]:
        news_list.append(f"{sym}: 暂无最新详细新闻，请查看专业财经平台。")
    return news_list

# ====================================================================
# 生成报告 + 图表（DeepSeek）
# ====================================================================

        adv_dec = get_advance_decline()
        adv_dec = float(adv_dec) if adv_dec is not None else "无数据"
        news = fetch_news(STOCKS, 3)
        
        # ---- 保存数据到 CSV ----
        import os
        import pandas as pd

        os.makedirs("data", exist_ok=True)

        macro_df = pd.DataFrame([macro])
        macro_df.to_csv("data/macro.csv", index=False)
        print("✅ 宏观数据已保存到 data/macro.csv")

        stock_records = []
        for sym, data in stock_dict.items():
            if "error" not in data:
                row = {"symbol": sym}
                row.update(data)
                stock_records.append(row)
        if stock_records:
            stock_df = pd.DataFrame(stock_records)
            stock_df.to_csv("data/stocks.csv", index=False)
            print("✅ 个股数据已保存到 data/stocks.csv")
        else:
            print("⚠️ 无有效个股数据可保存")

        sox_df = pd.DataFrame([sox])
        sox_df.to_csv("data/sox.csv", index=False)
        print("✅ SOX 信号已保存到 data/sox.csv")
        
        # ---- 构建 Prompt ----
        prompt = f"""你是一位专业股票分析师，请根据以下数据生成一份详细的每日投资看板报告。

**报告日期**：{datetime.now().strftime('%Y-%m-%d %H:%M')}"""
---
### 一、宏观概览
{json.dumps(macro, indent=2, ensure_ascii=False)}

涨跌家数比（SPY涨跌幅近似）：{adv_dec}% （正=上涨家数多）
---
### 二、SOX 指数信号
- 最新价：{sox.get('最新价', 'N/A')}
- 从高点回撤：{sox.get('回撤%', 'N/A')}%
- 是否技术性熊市：{'是' if sox.get('技术性熊市') else '否'}
- RSI(14)：{sox.get('RSI', 'N/A')}
- MA20：{sox.get('MA20', 'N/A')}
- MA50：{sox.get('MA50', 'N/A')}
- MA200：{sox.get('MA200', 'N/A')}
关键信号：
{chr(10).join(['- ' + s for s in sox.get('信号列表', [])])}
---
### 三、个股技术面数据
"""
        for sym, data in stock_dict.items():
            if "error" not in data:
                market = "港股" if sym.endswith(".HK") else "美股"
                prompt += f"""
**{sym} ({market})**
- 收盘价: ${data.get('收盘价', 'N/A'):.2f}
- 涨跌幅: {data.get('涨跌幅%', 'N/A'):.2f}%
- 成交量状态: {data.get('量比状态', 'N/A')}
- RSI(14): {data.get('RSI(14)', 'N/A'):.2f}
- MACD: {data.get('MACD', 'N/A'):.3f}
- 均线 MA5/MA20/MA50/MA200: {data.get('MA5', 'N/A'):.2f} / {data.get('MA20', 'N/A'):.2f} / {data.get('MA50', 'N/A'):.2f} / {data.get('MA200', 'N/A'):.2f}
- 布林带 (上/中/下): {data.get('布林上轨', 'N/A'):.2f} / {data.get('布林中轨', 'N/A'):.2f} / {data.get('布林下轨', 'N/A'):.2f}
- ATR(波动率): {data.get('ATR', 'N/A'):.2f}
"""
            else:
                prompt += f"**{sym}**: {data.get('error', '数据获取失败')}\n"
        
        prompt += """
---
### 四、新闻摘要
"""
        for n in news:
            prompt += f"- {n}\n"
        
        prompt += """
---
### 五、任务要求
请生成包含以下内容的报告（总字数控制在1200字以内）：
1. **宏观判断**：结合各项指标，分析当前经济环境及对股市的影响。
2. **SOX指数解读**：解读半导体板块趋势，指出关键支撑/阻力，并对持仓中的半导体个股给出指引。
3. **个股技术分析**：基于RSI、MACD、均线、布林带等，给出每只股票的短期趋势判断。
4. **情绪与资金**：根据涨跌家数比和成交量状态，评估市场情绪。
5. **操作建议**：当日及次日的具体策略（买入/持有/减仓/回避），尤其关注SOX的影响。
6. **风险提示**：指出当前最需要警惕的风险因素。
请使用简洁、专业的语言，并合理使用emoji增加可读性。
"""
        
        print("🧠 正在调用 DeepSeek 生成分析报告...")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位专业金融分析师。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        report = response.choices[0].message.content
        
        # ---- 输出文本报告 ----
        print("\n" + "="*100)
        print(f"📈 增强版每日看板报告 - {datetime.now().strftime('%Y-%m-%d')}")
        print("="*100)
        print(report)
        print("\n" + "="*100)
        print("✅ 报告生成完成")
        
        # ---- 生成个股 K 线图 ----
        print("\n📊 正在生成个股 K 线图...")
        for sym, data in stock_dict.items():
            if "error" not in data:
                df = safe_fetch(sym, period="3mo")
                if not df.empty and len(df) > 20:
                    try:
                        df_plot = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                        df_plot.index = pd.to_datetime(df_plot.index)
                        df_plot['MA20'] = df_plot['Close'].rolling(20).mean()
                        df_plot['MA50'] = df_plot['Close'].rolling(50).mean()
                        
                        apds = [
                            mpf.make_addplot(df_plot['MA20'], color='orange', width=1),
                            mpf.make_addplot(df_plot['MA50'], color='green', width=1),
                        ]
                        mpf.plot(
                            df_plot,
                            type='candle',
                            style='charles',
                            title=f"{sym} 近3个月走势 (MA20橙, MA50绿)",
                            volume=True,
                            addplot=apds,
                            figsize=(12, 6),
                            savefig=f'{sym}_chart.png'
                        )
                        print(f"✅ {sym} 图表已保存为 {sym}_chart.png")
                    except Exception as e:
                        print(f"⚠️ {sym} 绘图失败: {e}")
        print("✅ 所有图表生成完成")
    
    except Exception as e:
        print(f"❌ 报告生成过程中发生错误：{e}")
        import traceback
        traceback.print_exc()

# ====================================================================
# 主程序
# ====================================================================

if __name__ == "__main__":
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "你的_DeepSeek_API_Key":
        print("❌ 错误：请先在代码中设置 DEEPSEEK_API_KEY！")
    else:
        generate_report()
