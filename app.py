import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Quant Dashboard", page_icon="📊", layout="wide")

DATA_DIR = Path("data")


@st.cache_data(ttl=300)
def load_data():
    stocks = pd.read_csv(DATA_DIR / "stocks.csv") if (DATA_DIR / "stocks.csv").exists() else pd.DataFrame()
    macro = pd.read_csv(DATA_DIR / "macro.csv") if (DATA_DIR / "macro.csv").exists() else pd.DataFrame()
    sox = pd.read_csv(DATA_DIR / "sox.csv") if (DATA_DIR / "sox.csv").exists() else pd.DataFrame()

    cards = {}
    if (DATA_DIR / "cards.json").exists():
        with open(DATA_DIR / "cards.json", encoding="utf-8") as f:
            cards = json.load(f)

    leverage = {}
    if (DATA_DIR / "leverage_risk.json").exists():
        with open(DATA_DIR / "leverage_risk.json", encoding="utf-8") as f:
            leverage = json.load(f)

    report = ""
    if (DATA_DIR / "report.md").exists():
        report = (DATA_DIR / "report.md").read_text(encoding="utf-8")

    return stocks, macro, sox, cards, leverage, report


def sentiment_badge(label: str):
    color = {
        "极度贪婪": "#ff4d4f",
        "贪婪": "#ff7875",
        "中性": "#faad14",
        "恐惧": "#52c41a",
        "极度恐惧": "#237804",
    }.get(label, "#999")
    return f'<span style="background:{color};color:white;padding:4px 12px;border-radius:12px;font-weight:bold;">{label}</span>'


def reversal_badge(signal: str, conf: str):
    if signal == "反转":
        bg, emoji = "#722ed1", "🚀"
    elif signal == "反弹":
        bg, emoji = "#1890ff", "📈"
    else:
        bg, emoji = "#8c8c8c", "➖"
    return f'<span style="background:{bg};color:white;padding:3px 10px;border-radius:10px;font-size:12px;">{emoji} {signal}（{conf}）</span>'


def bottom_fishing_badge(score: int):
    if score >= 70:
        bg, text = "#cf1322", f"🔥 强抄底信号 {score}分"
    elif score >= 50:
        bg, text = "#fa541c", f"⚡ 抄底机会 {score}分"
    elif score >= 30:
        bg, text = "#fa8c16", f"👀 观察区 {score}分"
    else:
        bg, text = "#bfbfbf", f"— {score}分"
    return f'<span style="background:{bg};color:white;padding:3px 10px;border-radius:10px;font-size:12px;">{text}</span>'


# ==================== 页面渲染 ====================
st.title("📊 每日量化看盘 Dashboard")
st.caption(f"数据更新于：{datetime.now().strftime('%Y-%m-%d %H:%M')}")

stocks_df, macro_df, sox_df, cards_data, leverage_data, report_md = load_data()

# ---------- 顶部：宏观概览 ----------
top_cols = st.columns([2, 1, 1])

with top_cols[0]:
    st.subheader("🧠 AI 大盘总览")
    if report_md:
        st.markdown(report_md)
    else:
        st.info("暂无 AI 报告，请先运行 stock_dashboard.py")

with top_cols[1]:
    st.subheader("📈 宏观指标")
    if not macro_df.empty:
        m = macro_df.iloc[0].to_dict()
        for k, v in list(m.items())[:10]:
            st.metric(label=k, value=f"{v:.2f}" if isinstance(v, (int, float)) else str(v))

with top_cols[2]:
    st.subheader("🎚️ 市场情绪")
    if not macro_df.empty:
        vix = macro_df.iloc[0].get("VIX", "N/A")
        pcr = macro_df.iloc[0].get("Volume PCR", "N/A")
        st.metric("VIX", vix)
        st.metric("Put/Call Ratio", pcr)
        # 情绪标签
        if not stocks_df.empty and "抄底评分" in stocks_df.columns:
            avg_score = stocks_df["抄底评分"].mean()
            st.metric("平均抄底评分", f"{avg_score:.0f}")

# ---------- SOX 区域 ----------
if not sox_df.empty:
    st.divider()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("SOX 最新价", f"{sox_df.iloc[0].get('最新价', 'N/A')}")
    c2.metric("回撤", f"{sox_df.iloc[0].get('回撤', 'N/A')}%")
    c3.metric("RSI", f"{sox_df.iloc[0].get('RSI', 'N/A')}")
    c4.metric("MA20", f"{sox_df.iloc[0].get('MA20', 'N/A')}")
    c5.metric("MA50", f"{sox_df.iloc[0].get('MA50', 'N/A')}")
    signals = str(sox_df.iloc[0].get("信号列表", "")).split("；")
    st.caption(" | ".join([f"`{s}`" for s in signals if s]))

# ---------- 个股卡片 ----------
st.divider()
st.subheader("📋 个股决策卡片")

if stocks_df.empty:
    st.warning("暂无个股数据，请先运行 stock_dashboard.py")
    st.stop()

# 筛选器
col_filter1, col_filter2, col_filter3 = st.columns(3)
with col_filter1:
    market_filter = st.selectbox("市场", ["全部", "美股", "港股"])
with col_filter2:
    op_filter = st.selectbox("操作", ["全部", "买入", "观望", "卖出"])
with col_filter3:
    rev_filter = st.selectbox("反弹/反转", ["全部", "反转", "反弹", "无"])

cards_list = cards_data.get("stocks", []) if isinstance(cards_data, dict) else []
cards_map = {c["symbol"]: c for c in cards_list} if isinstance(cards_list, list) else {}

display_df = stocks_df.copy()

# 合并 AI 卡片数据
if "操作" not in display_df.columns and cards_map:
    display_df["操作"] = display_df["symbol"].map(lambda x: cards_map.get(x, {}).get("operation", "—"))
    display_df["趋势"] = display_df["symbol"].map(lambda x: cards_map.get(x, {}).get("trend", "—"))
    display_df["核心观点"] = display_df["symbol"].map(lambda x: cards_map.get(x, {}).get("core_view", "—"))

# 过滤
if market_filter == "美股":
    display_df = display_df[~display_df["symbol"].str.endswith(".HK")]
elif market_filter == "港股":
    display_df = display_df[display_df["symbol"].str.endswith(".HK")]

if op_filter != "全部" and "操作" in display_df.columns:
    display_df = display_df[display_df["操作"] == op_filter]

if rev_filter != "全部":
    display_df = display_df[display_df["反弹反转信号"] == rev_filter]

# 排序：抄底评分高的在前
if "抄底评分" in display_df.columns:
    display_df = display_df.sort_values("抄底评分", ascending=False)

st.write(f"显示 {len(display_df)} / {len(stocks_df)} 只股票")

# 网格布局：每行 3 张卡片
for i in range(0, len(display_df), 3):
    row = st.columns(3)
    for j in range(3):
        if i + j >= len(display_df):
            break
        r = display_df.iloc[i + j]
        sym = r["symbol"]
        card = cards_map.get(sym, {})

        with row[j]:
            trend_color = {
                "看多": "border-left: 4px solid #52c41a;",
                "看空": "border-left: 4px solid #ff4d4f;",
                "震荡": "border-left: 4px solid #faad14;",
            }.get(r.get("趋势", "—"), "border-left: 4px solid #d9d9d9;")

            st.markdown(f"""
            <div style="{trend_color} background:#fafafa; padding:16px; border-radius:8px; margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="margin:0;">{sym}</h3>
                    <span style="font-size:12px; color:#666;">{r.get('趋势','—')}</span>
                </div>
                <div style="font-size:24px; font-weight:bold; margin:8px 0;">
                    {r.get('收盘价',0):.2f} 
                    <span style="font-size:14px; color:{'#52c41a' if r.get('涨跌幅',0)>=0 else '#ff4d4f'};">
                        {'+' if r.get('涨跌幅',0)>=0 else ''}{r.get('涨跌幅',0):.2f}%
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("RSI", f"{r.get('RSI_14',0):.1f}")
            m2.metric("MACD", f"{r.get('MACD',0):.3f}")
            m3.metric("PE", r.get("PE_Ratio", "N/A"))

            # 新功能标签：抄底 + 反弹/反转
            st.markdown(
                f"<div style='margin:6px 0;'>{bottom_fishing_badge(r.get('抄底评分', 0))}  {reversal_badge(r.get('反弹反转信号','无'), r.get('反弹反转置信度','低'))}</div>",
                unsafe_allow_html=True,
            )

            # 抄底依据
            reasons = r.get("抄底依据", [])
            if reasons and isinstance(reasons, list):
                st.caption("🎯 " + " / ".join(reasons))

            # 反弹反转描述
            desc = r.get("反弹反转描述", "")
            if desc and desc != "暂无明确信号":
                st.caption(f"📐 {desc}")

            # POC
            poc = r.get("资金集中价位")
            if poc and not pd.isna(poc):
                st.caption(f"💰 POC 资金集中区: **{poc:.2f}** (占比 {r.get('集中度',0):.1f}%)")

            # 杠杆预警
            lev = leverage_data.get("stocks", {}).get(sym, {})
            if lev and "details" in lev:
                st.write("**⚠️ 杠杆预警**")
                lev_cols = st.columns(len(lev["details"]))
                for idx, (lev_key, lev_info) in enumerate(lev["details"].items()):
                    atr_mult = lev_info.get("距强平ATR倍数", 999)
                    if isinstance(atr_mult, (int, float)):
                        if atr_mult < 3:
                            color = "#ff5252"
                        elif atr_mult < 6:
                            color = "#ffd54f"
                        else:
                            color = "#52c41a"
                    else:
                        color = "#bfbfbf"
                    with lev_cols[idx]:
                        st.markdown(
                            f"<div style='border-left:3px solid {color}; padding-left:8px; font-size:12px;'>"
                            f"<b>{lev_key}</b><br>强平 ${lev_info.get('强平价','N/A')}<br>"
                            f"<span style='color:{color};'>{atr_mult}倍ATR</span></div>",
                            unsafe_allow_html=True,
                        )

            # 核心观点（AI）
            core = r.get("核心观点") or card.get("core_view", "")
            if core:
                st.info(core)

            # 展开详情
            with st.expander("🔍 详情"):
                st.write(f"**操作建议**: {r.get('操作', card.get('operation', '—'))}")
                st.write(f"**MA5/20/50**: {r.get('MA5',0):.2f} / {r.get('MA20',0):.2f} / {r.get('MA50',0):.2f}")
                st.write(f"**布林**: {r.get('布林上轨',0):.2f} / {r.get('布林中轨',0):.2f} / {r.get('布林下轨',0):.2f}")
                st.write(f"**ATR**: {r.get('ATR',0):.2f}")
                st.write(f"**杠杆风险**: {r.get('杠杆风险等级', 'N/A')}")

                sniper = card.get("sniper", {})
                if sniper:
                    st.write("---")
                    st.write(f"🎯 **理想买入**: {sniper.get('ideal_buy', '—')}")
                    st.write(f"🎯 **二次加仓**: {sniper.get('second_buy', '—')}")
                    st.write(f"🛑 **止损**: {sniper.get('stop_loss', '—')}")
                    st.write(f"🏁 **目标**: {sniper.get('target', '—')}")

                sectors = card.get("sectors", [])
                if sectors:
                    st.caption("🏷 " + " · ".join(sectors))

                catas = card.get("catalysts", [])
                risks = card.get("risks", [])
                if catas:
                    st.success("**利好**: " + " / ".join(catas))
                if risks:
                    st.error("**风险**: " + " / ".join(risks))

st.divider()
st.caption("数据仅供研究参考，不构成投资建议。")
