import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import json
import yfinance as yf

st.set_page_config(layout="wide", page_title="每日股票看板", initial_sidebar_state="expanded")

# ---------- 自定义 CSS ----------
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #141b2d 50%, #1a2335 100%); }
    .metric-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 16px 20px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        height: 100%;
    }
    .metric-card:hover { border-color: rgba(0,200,255,0.3); transform: translateY(-2px); }
    .metric-value { font-size: 24px; font-weight: 700; color: #ffffff; line-height: 1.2; }
    .metric-label { font-size: 12px; color: rgba(255,255,255,0.5); letter-spacing: 0.5px; text-transform: uppercase; }
    .section-title { font-size: 20px; font-weight: 600; color: #ffffff; border-left: 3px solid #00d4ff; padding-left: 14px; margin: 24px 0 16px 0; }
    .stock-card { background: rgba(255,255,255,0.03); border-radius: 16px; padding: 16px 20px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 12px; }
    .footer { text-align: center; color: rgba(255,255,255,0.15); font-size: 11px; padding: 30px 0 10px 0; border-top: 1px solid rgba(255,255,255,0.04); margin-top: 30px; }
    .report-box { background: rgba(255,255,255,0.03); border-radius: 16px; padding: 20px 24px; border: 1px solid rgba(255,255,255,0.06); margin-top: 10px; color: rgba(255,255,255,0.85); font-size: 14px; line-height: 1.6; }
    .sox-signal { background: rgba(255,255,255,0.03); border-radius: 12px; padding: 10px 16px; margin-top: 8px; border-left: 2px solid rgba(0,212,255,0.3); color: rgba(255,255,255,0.6); font-size: 13px; }
    .card-tag { padding: 2px 10px; border-radius: 12px; font-size: 13px; }
    .sector-tag { background: rgba(0,212,255,0.1); color: #00d4ff; padding: 2px 10px; border-radius: 10px; font-size: 12px; margin-right: 6px; display:inline-block; margin-bottom:4px; }
</style>
""", unsafe_allow_html=True)

# ---------- 侧边栏 ----------
st.sidebar.title("⚙️ 控制面板")
if st.sidebar.button("🔄 强制刷新数据"):
    st.cache_data.clear()
    st.rerun()

view = st.sidebar.radio("📌 导航", ["📊 大盘概览", "📈 个股详情", "🎯 AI决策卡片", "📝 每日报告"])

period_map = {"1个月": "1mo", "3个月": "3mo", "6个月": "6mo", "1年": "1y"}
selected_period = st.sidebar.selectbox("📅 K线图周期", list(period_map.keys()), index=1)
sort_option = st.sidebar.selectbox("📊 个股排序", ["默认顺序", "涨幅高→低", "涨幅低→高", "PE高→低", "PE低→高"])

# ---------- 标题 ----------
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
    <div>
        <span style="font-size:28px;font-weight:700;color:#ffffff;">📈 每日股票看板</span>
        <span style="font-size:13px;color:rgba(255,255,255,0.25);margin-left:12px;">QUANT DASHBOARD</span>
    </div>
    <div style="font-size:12px;color:rgba(255,255,255,0.2);background:rgba(255,255,255,0.04);padding:4px 14px;border-radius:20px;border:1px solid rgba(255,255,255,0.05);">
        📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>
</div>
<hr style="border:0;height:1px;background:linear-gradient(90deg,transparent,rgba(0,212,255,0.15),transparent);margin:8px 0 16px 0;">
""", unsafe_allow_html=True)

# ---------- 加载数据 ----------
@st.cache_data(ttl=3600)
def load_data():
    macro_df = pd.read_csv("data/macro.csv") if os.path.exists("data/macro.csv") else None
    stocks_df = pd.read_csv("data/stocks.csv") if os.path.exists("data/stocks.csv") else None
    sox_df = pd.read_csv("data/sox.csv") if os.path.exists("data/sox.csv") else None
    report = ""
    if os.path.exists("data/report.md"):
        with open("data/report.md", "r", encoding="utf-8") as f:
            report = f.read()
    return macro_df, stocks_df, sox_df, report

@st.cache_data(ttl=3600)
def load_cards():
    if os.path.exists("data/cards.json"):
        try:
            with open("data/cards.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

macro_df, stocks_df, sox_df, report = load_data()
cards_data = load_cards()

OP_COLOR = {"买入": "#00e676", "观望": "#ffd54f", "卖出": "#ff5252"}

def score_gauge(score, operation):
    color = OP_COLOR.get(operation, "#00d4ff")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'font': {'color': '#ffffff', 'size': 32}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'rgba(255,255,255,0.2)', 'tickfont': {'color': 'rgba(255,255,255,0.3)'}},
            'bar': {'color': color},
            'bgcolor': 'rgba(255,255,255,0.03)',
            'borderwidth': 0,
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(height=160, margin=dict(l=10, r=10, t=10, b=10),
                       paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    return fig

# ---------- 视图：大盘概览 ----------
if view == "📊 大盘概览":
    # 宏观数据
    st.markdown('<div class="section-title">🌍 宏观数据</div>', unsafe_allow_html=True)
    if macro_df is not None and not macro_df.empty:
        macro_cols = ["VIX", "美元指数", "标普500", "纳斯达克100", "黄金", "WTI原油", "10年期美债收益率",
                      "美国国债规模", "芝加哥联储杠杆指数", "2年期实际利率（近似）", "FINRA保证金债务", "Volume PCR", "OI PCR"]
        available = [c for c in macro_cols if c in macro_df.columns]
        cols_per_row = 6
        for i in range(0, len(available), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, col_name in enumerate(available[i:i+cols_per_row]):
                val = macro_df.iloc[0][col_name]
                if isinstance(val, (int, float)):
                    if col_name in ["VIX", "美元指数", "标普500", "纳斯达克100", "黄金", "WTI原油", "10年期美债收益率"]:
                        display = f"{val:.2f}"
                    elif col_name == "美国国债规模":
                        display = f"{val/1e12:.2f}T" if val > 1e12 else f"{val:,.0f}"
                    else:
                        display = f"{val:.2f}" if abs(val) < 1000 else f"{val:,.0f}"
                else:
                    display = str(val)
                with row_cols[j]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{col_name}</div>
                        <div class="metric-value">{display}</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ 暂无宏观数据")

    # SOX
    st.markdown('<div class="section-title">📉 SOX 半导体指数</div>', unsafe_allow_html=True)
    if sox_df is not None and not sox_df.empty:
        sox_row = sox_df.iloc[0]
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">📊 最新价</div><div class="metric-value">{sox_row.get('最新价', 'N/A'):.2f}</div></div>""", unsafe_allow_html=True)
        with col2:
            drawdown = sox_row.get('回撤%', 0)
            color = "#ff5252" if drawdown < -20 else "#ffd54f"
            st.markdown(f"""<div class="metric-card"><div class="metric-label">📉 回撤</div><div class="metric-value" style="color:{color};">{drawdown:.2f}%</div></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">📈 RSI(14)</div><div class="metric-value">{sox_row.get('RSI', 'N/A'):.2f}</div></div>""", unsafe_allow_html=True)
        with col4:
            status = "🐻 熊市" if sox_row.get('技术性熊市') else "🐂 非熊市"
            color = "#ff5252" if sox_row.get('技术性熊市') else "#00e676"
            st.markdown(f"""<div class="metric-card"><div class="metric-label">📌 状态</div><div class="metric-value" style="font-size:18px;color:{color};">{status}</div></div>""", unsafe_allow_html=True)
        with col5:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">📌 MA20</div><div class="metric-value" style="font-size:18px;">{sox_row.get('MA20', 'N/A'):.2f}</div></div>""", unsafe_allow_html=True)
        if '信号列表' in sox_row and sox_row['信号列表']:
            signals = sox_row['信号列表']
            if isinstance(signals, str):
                signals = signals.split('；')
            elif not isinstance(signals, list):
                signals = [str(signals)]
            signal_html = "；".join(signals)
            st.markdown(f"""<div class="sox-signal">⚡ 关键信号：{signal_html}</div>""", unsafe_allow_html=True)
    else:
        st.warning("⚠️ 暂无 SOX 数据")

    # 宏观历史趋势
    st.markdown('<div class="section-title">📈 宏观历史趋势（近30天）</div>', unsafe_allow_html=True)
    macro_hist_symbols = {"VIX": "^VIX", "10年期美债收益率": "^TNX", "标普500": "^GSPC"}
    for name, sym in macro_hist_symbols.items():
        try:
            hist = yf.download(sym, period="1mo", progress=False)
            if not hist.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name=name, line=dict(color='#00d4ff')))
                fig.update_layout(height=150, margin=dict(l=10, r=10, t=20, b=10),
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)', color='rgba(255,255,255,0.2)')
                fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)', color='rgba(255,255,255,0.2)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption(f"⚠️ {name} 数据加载失败")
        except Exception as e:
            st.caption(f"⚠️ {name} 错误: {str(e)[:50]}")

# ---------- 视图：个股详情 ----------
elif view == "📈 个股详情":
    st.markdown('<div class="section-title">📊 个股数据</div>', unsafe_allow_html=True)
    if stocks_df is not None and not stocks_df.empty:
        if sort_option == "涨幅高→低":
            stocks_df = stocks_df.sort_values("涨跌幅%", ascending=False)
        elif sort_option == "涨幅低→高":
            stocks_df = stocks_df.sort_values("涨跌幅%", ascending=True)
        elif sort_option == "PE高→低":
            stocks_df = stocks_df.sort_values("PE Ratio", ascending=False, na_position='last')
        elif sort_option == "PE低→高":
            stocks_df = stocks_df.sort_values("PE Ratio", ascending=True, na_position='last')
        else:
            stock_order = ["MU","AAOI","GOOGL","MSFT","AMZN","MRVL","LITE","SNDK","NVDA","ORCL","SPCX","SKHY","TSLA","0700.HK","0883.HK","3750.HK"]
            stocks_df['order'] = stocks_df['symbol'].map({s:i for i,s in enumerate(stock_order)})
            stocks_df = stocks_df.sort_values('order')
        for _, row in stocks_df.iterrows():
            sym = row['symbol']
            if row.get('error'):
                st.warning(f"无法获取 {sym} 数据")
                continue
            close = row.get('收盘价', 0)
            change = row.get('涨跌幅%', 0)
            vol_status = row.get('量比状态', 'N/A')
            pe = row.get('PE Ratio', 'N/A')
            rsi = row.get('RSI(14)', 50)
            ma20 = row.get('MA20', 0)
            change_color = "#00e676" if change > 0 else "#ff5252" if change < 0 else "#ffd54f"
            change_symbol = "▲" if change > 0 else "▼" if change < 0 else "●"
            display_name = sym.replace(".HK", "")
            st.markdown(f"""
            <div class="stock-card">
                <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;">
                    <div>
                        <span style="font-weight:600;font-size:18px;color:#ffffff;">{display_name}</span>
                        <span style="margin-left:12px;font-size:14px;color:{change_color};">{change_symbol} {change:.2f}%</span>
                        <span style="margin-left:12px;font-size:14px;color:rgba(255,255,255,0.6);">💰 {close:.2f}</span>
                    </div>
                    <div style="display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:rgba(255,255,255,0.5);">
                        <span>📈 PE: {pe if pe != 'N/A' else 'N/A'}</span>
                        <span>📊 量比: {vol_status}</span>
                        <span>📉 RSI: {rsi:.1f}</span>
                        <span>📌 MA20: {ma20:.2f}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ---------- K线图（增强调试） ----------
            st.caption(f"⏳ 正在加载 {sym} 数据（周期: {selected_period}）...")
            try:
                period = period_map[selected_period]
                hist = yf.download(sym, period=period, progress=False)
                # 显示调试信息
                with st.expander(f"🔍 {sym} 调试信息"):
                    st.write(f"数据类型: {type(hist)}")
                    st.write(f"数据是否为空: {hist.empty if hist is not None else 'hist is None'}")
                    if hist is not None and not hist.empty:
                        st.write(f"数据条数: {len(hist)}")
                        st.write("数据列:", hist.columns.tolist())
                        st.write("数据样本:", hist.head(3))
                    else:
                        st.warning("无数据")
                
                if hist is not None and not hist.empty and 'Open' in hist.columns and len(hist) > 5:
                    # 绘制K线图
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                        row_heights=[0.7, 0.3])
                    fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                                                 low=hist['Low'], close=hist['Close'], name=sym,
                                                 increasing_line_color='#00e676', decreasing_line_color='#ff5252'), row=1, col=1)
                    ma20_hist = hist['Close'].rolling(20).mean()
                    ma50_hist = hist['Close'].rolling(50).mean()
                    fig.add_trace(go.Scatter(x=hist.index, y=ma20_hist, line=dict(color='#ffd54f', width=1.5), name="MA20"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=hist.index, y=ma50_hist, line=dict(color='#00d4ff', width=1.5), name="MA50"), row=1, col=1)
                    fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name="成交量", marker_color='rgba(0,212,255,0.2)'), row=2, col=1)
                    fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10),
                                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                      xaxis_rangeslider_visible=False, showlegend=False,
                                      font=dict(color='rgba(255,255,255,0.3)'))
                    fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)', color='rgba(255,255,255,0.2)')
                    fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)', color='rgba(255,255,255,0.2)')
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                else:
                    st.warning(f"⚠️ {sym} 数据不足（{len(hist) if hist is not None else 0}条记录）或列名不匹配，无法绘制K线图")
            except Exception as e:
                st.error(f"❌ {sym} K线图加载失败: {str(e)}")
    else:
        st.warning("⚠️ 暂无个股数据")

# ---------- 视图：AI决策卡片 ----------
elif view == "🎯 AI决策卡片":
    st.markdown('<div class="section-title">🎯 AI 决策卡片</div>', unsafe_allow_html=True)
    if cards_data and cards_data.get("stocks"):
        st.caption(f"生成时间：{cards_data.get('generated_at', 'N/A')}")
        for card in cards_data["stocks"]:
            sym = card.get("symbol", "N/A")
            op = card.get("operation", "观望")
            op_color = OP_COLOR.get(op, "#00d4ff")
            col_left, col_right = st.columns([3, 1])
            with col_left:
                st.markdown(f"""
                <div class="stock-card">
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                        <span style="font-weight:700;font-size:20px;color:#fff;">{sym}</span>
                        <span class="card-tag" style="background:{op_color}22;color:{op_color};">{op}</span>
                        <span style="color:rgba(255,255,255,0.4);font-size:13px;">{card.get('trend','')}</span>
                    </div>
                    <div style="color:rgba(255,255,255,0.75);font-size:14px;">{card.get('core_view','')}</div>
                </div>
                """, unsafe_allow_html=True)
                # 狙击点位
                sniper = card.get("sniper", {}) or {}
                s1, s2, s3, s4 = st.columns(4)
                sniper_labels = [
                    ("🎯 理想买入", sniper.get("ideal_buy", "-"), "#00d4ff"),
                    ("📈 二次买入", sniper.get("second_buy", "-"), "#00e676"),
                    ("🛑 止损位", sniper.get("stop_loss", "-"), "#ff5252"),
                    ("🏁 止盈目标", sniper.get("target", "-"), "#ffd54f"),
                ]
                for c, (label, val, color) in zip([s1, s2, s3, s4], sniper_labels):
                    with c:
                        st.markdown(f"""<div class="metric-card" style="border-left:2px solid {color};">
                            <div class="metric-label">{label}</div>
                            <div style="font-size:13px;color:rgba(255,255,255,0.8);margin-top:4px;">{val}</div>
                        </div>""", unsafe_allow_html=True)
                # 催化 & 风险
                cc1, cc2 = st.columns(2)
                with cc1:
                    catalysts = card.get("catalysts", [])
                    if catalysts:
                        st.markdown(f'<div class="sox-signal" style="border-left-color:#00e676;">✨ ' + '；'.join(catalysts) + '</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="sox-signal" style="border-left-color:#00e676;">✨ 暂无</div>', unsafe_allow_html=True)
                with cc2:
                    risks = card.get("risks", [])
                    if risks:
                        st.markdown(f'<div class="sox-signal" style="border-left-color:#ff5252;">🚨 ' + '；'.join(risks) + '</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="sox-signal" style="border-left-color:#ff5252;">🚨 暂无</div>', unsafe_allow_html=True)
                # 板块标签
                sectors = card.get("sectors", [])
                if sectors:
                    tags = "".join([f'<span class="sector-tag">{s}</span>' for s in sectors])
                    st.markdown(f'<div style="margin:8px 0 20px 0;">{tags}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="margin-bottom:20px;"></div>', unsafe_allow_html=True)
                
                # ---------- 增加“查看分析依据” ----------
                with st.expander("📊 查看分析依据（原始技术指标）"):
                    stock_row = stocks_df[stocks_df['symbol'] == sym]
                    if not stock_row.empty:
                        row = stock_row.iloc[0]
                        fields = ['收盘价', '涨跌幅%', '成交量', '量比状态', 'RSI(14)', 'MACD', 'MACD信号', 
                                  'MA5', 'MA20', 'MA50', 'MA200', '布林上轨', '布林中轨', '布林下轨', 'ATR', 'PE Ratio']
                        st.markdown("**技术指标**：")
                        cols = st.columns(4)
                        for idx, field in enumerate(fields):
                            val = row.get(field, 'N/A')
                            if isinstance(val, float):
                                val = f"{val:.2f}"
                            with cols[idx % 4]:
                                st.metric(field, val)
                        # 强平价格
                        st.markdown("**强平价格（斩仓线）**：")
                        lev_cols = st.columns(3)
                        for lev, col in zip(['2x', '3x', '5x'], lev_cols):
                            col_name = f'强平价格_{lev}'
                            if col_name in row:
                                col.metric(lev, row[col_name])
                    else:
                        st.warning("暂无原始数据")
            with col_right:
                st.plotly_chart(score_gauge(card.get("score", 50), op), use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("📌 决策卡片尚未生成，请等待每日数据更新（需配置 DEEPSEEK_API_KEY）")

# ---------- 视图：每日报告 ----------
elif view == "📝 每日报告":
    st.markdown("---")
    st.markdown('<div class="section-title">📝 每日大盘总览</div>', unsafe_allow_html=True)
    if report:
        st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)
    else:
        if os.path.exists("data/report.md"):
            st.error("❌ 报告文件存在但内容为空，请检查 AI 生成逻辑")
        else:
            st.info("📌 今日报告尚未生成，请等待 GitHub Actions 完成数据采集并生成 report.md")

# ---------- 底部 ----------
st.markdown(f"""
<div class="footer">
    ⚡ 数据每日自动更新 · 系统运行正常 · {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>
""", unsafe_allow_html=True)
