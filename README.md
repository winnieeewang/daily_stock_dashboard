# 📈 Investment Copilot

> 投资分析工作台 v2.1 — 面向专业投资者的多模块、自动化、AI 增强的 Streamlit 看板。
> Author: Winnie Wang

---

## 🆕 v2.1 主要升级（v2.0 → v2.1）

| 模块 | v2.0 | v2.1 (current) |
|------|------|----------------|
| **Dashboard 指标卡** | 5 张 | **9 张**：新增 2-Year Scorecard / U.S. National Debt / FINRA Margin Debt / NFCI Leverage |
| **FedWatch** | numpy bug | ✅ 修复 `_normalize_close_series` |
| **AI 报告自动更新** | 手动 | ✅ 3 个 GitHub Actions（morning/evening/daily）按 cron 自动跑 |
| **个股新闻** | SerpApi 单一源 | ✅ **6 源聚合**：SerpApi + Finnhub + NewsAPI + Yahoo RSS + Stocktwits + 东方财富 |
| **Vol/OI PCR** | 缺失 | ✅ yfinance `option_chain` 抓 Vol PCR / OI PCR / 隐含波动率 |
| **下周走势预测** | 无 | ✅ DeepSeek 综合 政策 + 消息 + 基本面 + 技术面 + 期权 |

---

## 🆕 v2.0 主要升级

| 模块 | v1.0 (initial) | v2.0 |
|------|----------------|------|
| **首页** | 单列流式，看板/分析混在一起 | 🏠 Dashboard + 5 个独立页面 |
| **市场全景** | 5 个简单卡片 | F&G 5 因子 / 情绪 / VIX / SOX / 10Y / 今日重点 / Top 机会 / Top 风险 |
| **热力图** | 无 | 美股 (60+ 标的) + 港股 (30+ 标的) 全市场 treemap |
| **跨资产对比** | 单图归一化，NVDA 拆股异常 | Price / Return % / Log Return 三档，auto_adjust 修复 |
| **AI 报告** | 日报 + 周报 | ➕ Morning Brief（盘前）+ Evening Recap（盘后） |
| **个股分析** | 单列 | 三栏工作台：左(自选/历史/搜索) / 中(分析/AI/K线/资金/策略) / 右(评分/风险/新闻/市场) |
| **FedWatch** | 无 | SOFR/FF 期货反推下次会议降息概率 |
| **经济日历** | 无 | SerpApi + 静态兜底 |

---

## 📁 项目结构

```
.
├── app.py                  # Streamlit 主入口（Dashboard + 5 个页面）
├── stock_dashboard.py      # 数据采集 + AI 报告生成（每日跑）
├── utils.py                # 工具模块：新闻/F&G/FedWatch/日历/热力图/PCR/预测
├── requirements.txt        # Python 依赖
├── data/                   # 自动生成的数据文件
│   ├── macro.csv           # 宏观指数
│   ├── stocks.csv          # 个股行情 + 技术指标
│   ├── sox.csv             # SOX 半导体信号
│   ├── sp500.csv           # 标普 500 信号
│   ├── cards.json          # AI 决策卡片
│   ├── leverage_risk.json  # 杠杆强平监控
│   ├── news.json           # 新闻（多源聚合：SerpApi+Finnhub+NewsAPI+Yahoo+Stocktwits+东财）
│   ├── report.md           # AI 大盘总览
│   ├── weekly_report.md    # 每周总结（周五生成）
│   ├── morning_brief.md    # ☀️ 盘前早报
│   ├── evening_recap.md    # 🌙 盘后总结
│   ├── extra_indicators.json   # 🆕 2Y / 美债 / Margin Debt / NFCI
│   ├── options_pcr.json        # 🆕 Vol/OI PCR（每只自选股）
│   └── predictions.json        # 🆕 下周走势预测（每只自选股）
├── .github/workflows/      # 🆕 3 个自动任务
│   ├── morning.yml         # 盘前 12:30 UTC = 美东 08:30
│   ├── evening.yml         # 盘后 21:30 UTC = 美东 17:30
│   └── daily.yml           # 盘后 20:30 UTC 全量更新
└── README.md
```

---

## 🚀 本地运行

```bash
# 1. 克隆
git clone https://github.com/<your-name>/investment-copilot.git
cd investment-copilot

# 2. 安装依赖
pip install -r requirements.txt

# 3. （可选）配置 Secret
mkdir -p .streamlit
cat > .streamlit/secrets.toml <<EOF
# 必填项：无（所有 API 都有免费 fallback）
# 推荐配置（让 Dashboard 的 9 张卡全部点亮）
FRED_API = "your-fred-key"          # 4 张新指标卡
DEEPSEEK_API_KEY = "your-deepseek"  # AI 报告 / 下周预测
SERPAPI = "your-serpapi"            # 主力新闻（100/月免费）
FINNHUB_API = "your-finnhub"        # 美股个股新闻（60/min 免费，可选）
NEWSAPI_KEY = "your-newsapi"        # 宏观新闻（100/天免费，可选）
EOF

# 4. 采集数据（首次或每日更新）
python stock_dashboard.py              # 全量
python stock_dashboard.py --morning    # 只跑盘前（Morning Brief + 新指标）
python stock_dashboard.py --evening    # 只跑盘后（Recap + 预测 + PCR）
python stock_dashboard.py --extras     # 只刷新 4 张新指标 + Vol/OI PCR
python stock_dashboard.py --predictions # 只生成下周走势预测
python stock_dashboard.py --news       # 只刷新多源新闻

# 5. 启动看板
streamlit run app.py
```

打开 http://localhost:8501

---

## ⚙️ GitHub Actions 自动跑（v2.1：3 个工作流）

| 工作流 | 文件 | Cron (UTC) | 美东时间 | 任务 |
|--------|------|-----------|---------|------|
| **Morning** | `.github/workflows/morning.yml` | `30 12 * * 1-5` | ≈ 08:30 盘前 | Morning Brief + 新指标 + PCR |
| **Evening** | `.github/workflows/evening.yml` | `30 21 * * 1-5` | ≈ 17:30 盘后 | Evening Recap + 预测 + PCR + 新指标 |
| **Daily** | `.github/workflows/daily.yml` | `30 20 * * 1-5` | ≈ 16:30 盘后 | 全量（含 brief+recap+周报+新指标+预测） |

**注意：cron 用 UTC，** 美股在夏令时（3-10月）UTC-4，冬令时 UTC-5。Cron 建议根据 DST 调整。

### GitHub Secrets 清单

| Secret 名 | 优先级 | 用途 | 获取方式 |
|-----------|--------|------|----------|
| **`FRED_API`** | ⭐⭐⭐ 强烈推荐 | 4 张新指标卡 | https://fred.stlouisfed.org/docs/api/api_key.html |
| **`DEEPSEEK_API_KEY`** | ⭐⭐⭐ 强烈推荐 | AI 报告 / 预测 | https://platform.deepseek.com/ |
| **`SERPAPI`** | ⭐⭐ 推荐 | 主力新闻（100/月免费） | https://serpapi.com/ |
| `FINNHUB_API` | ⭐ 可选 | 美股个股新闻（60/min） | https://finnhub.io/register |
| `NEWSAPI_KEY` | ⭐ 可选 | 宏观新闻（100/天） | https://newsapi.org/register |
| `TELEGRAM_BOT_TOKEN` | 可选 | Telegram 推送预警 | @BotFather |
| `TELEGRAM_CHAT_ID` | 可选 | 推送目标 | @userinfobot |

> 💡 **最低成本方案**：只配 `FRED_API` + `DEEPSEEK_API_KEY` + `SERPAPI` 就能让 9 张 Dashboard 卡 + AI 报告 + 多源新闻全部工作。免费源（Yahoo RSS、Stocktwits、东财）始终可用。

---

## 🔧 v2.1 修复点详细说明

### 1. 修复 FedWatch numpy bug
**症状：** `FedWatch 暂不可用: 'numpy.float64' object has no attribute 'iloc'`

**根因：** `yf.download("SR3=F")` 在某些情况下返回的 `df["Close"]` 是单行单列 DataFrame，`.squeeze()` 把它变成了 scalar (numpy.float64)，后面再调 `.iloc` 就崩了。

**修复（utils.py）：** 抽离 `_normalize_close_series()` 统一处理 MultiIndex / 单行 / Series / DataFrame 所有边界情况：
```python
def _normalize_close_series(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]  # 多列取第一列
    return close.squeeze() if isinstance(close, pd.Series) else None
```

### 2. Dashboard 加 4 张新指标卡
| 卡 | 数据源 | FRED Code | 含义 |
|----|-------|-----------|------|
| **2-Year Scorecard** | FRED + yfinance | `DGS2` | 2s10s 利差，倒挂 = 衰退预警 |
| **U.S. National Debt** | FRED | `GFDEBTN` | 美国国债总额（单位：T） |
| **FINRA Margin Debt** | FRED | `MDEBT` | 散户融资余额，同比 +30% = 杠杆高 |
| **Chicago Fed NFCI Leverage** | FRED | `NFCILEVERAGE` | 杠杆可获得性，>0.5 = 紧缩 |

> 💡 这 4 张卡的共同点：**FRED 全免费**。去 https://fred.stlouisfed.org/docs/api/api_key.html 申请一个 key 就能用。

### 3. AI 报告自动更新（GitHub Actions）
v2.0 只能手动跑 `python stock_dashboard.py`。v2.1 提供 3 个工作流：
- **morning.yml**：每个交易日 12:30 UTC（美东 08:30）跑 `--morning`
- **evening.yml**：每个交易日 21:30 UTC（美东 17:30）跑 `--evening`
- **daily.yml**：每个交易日 20:30 UTC（美东 16:30）跑全量

工作流跑完会自动 `git commit` + `git push` 把 `data/` 提交回仓库，Streamlit Cloud 部署会立即看到更新。

### 4. Vol/OI PCR（用 yfinance 免费期权链）
**v2.0 缺失**——Alpha Vantage 25 次/天的免费额度根本不够。

**v2.1 修复：** 用 `yf.Ticker(sym).option_chain(expiry)` 拉真实期权链：
```python
calls = chain.calls   # ['volume', 'openInterest', 'impliedVolatility', ...]
puts = chain.puts
vol_pcr = puts['volume'].sum() / calls['volume'].sum()
oi_pcr  = puts['openInterest'].sum() / calls['openInterest'].sum()
```
判断标准：Vol PCR > 1.2 = 看空/对冲情绪重；< 0.8 = 市场偏多。

### 5. 下周走势预测（综合四维）
**v2.1 新增**：每个交易日盘后给每只自选股生成 300-500 字的下周走势预测。

DeepSeek 输入：技术面 (RSI/MACD/MA/ATR) + 消息面（最近 5 条新闻）+ 政策面（未来 2 周经济日程）+ 基本面 (PE/财报) + 期权 (Vol/OI PCR)。

输出结构：
```
=== 1. 综合判断 ===     一句话定位（看多/中性/看空）
=== 2. 关键驱动 ===     3-5 个核心因素
=== 3. 关键价位 ===     阻力位 / 支撑位 / 预计波动区间
=== 4. 风险因素 ===     2-3 个可能颠覆判断的变量
=== 5. 操作建议 ===     加仓/减仓/观望/对冲（不给具体点位）
```

### 6. 多源新闻聚合（6 个数据源，3 个完全免费）
| 数据源 | 免费额度 | 注册难度 | 用途 |
|--------|---------|---------|------|
| SerpApi | 100/月 | 简单 | 主力宏观/政策/个股 |
| Finnhub | 60 req/min | 简单 | 美股个股新闻 |
| NewsAPI | 100/day | 简单 | 宏观新闻 |
| **Yahoo Finance RSS** | ∞ | 不需要 | 美股/港股个股（**最稳定**） |
| **Stocktwits API** | ∞ | 不需要 | 散户情绪 |
| **东方财富网 API** | ∞ | 不需要 | 中文宏观/A股 |

聚合策略：每个源失败不影响其他，会去重并标注 `sources_used`。

### 7. NVDA 拆股 + 10Y 量纲（v2.0 已修）
```python
# 关键：auto_adjust=True
df = yf.download(sym, period=period, progress=False, auto_adjust=True)
```
三档视图：Price（对数价格 + 10Y 副轴）/ Return %（累计回报）/ Log Return。

---

## 🧠 AI 报告（DeepSeek）

| 报告 | 时机 | 字数 | 风格 |
|------|------|------|------|
| `report.md` | 每日收盘 | 500 | 简明总览 |
| `cards.json` | 每日 | JSON | 结构化决策 |
| `weekly_report.md` | 周五 | 800-1200 | 复盘 + 展望 |
| `morning_brief.md` | 每日盘前 | 500-800 | 卖方 Morning Note |
| `evening_recap.md` | 每日盘后 | 800-1200 | 卖方 EOD Note |
| `predictions.json` | 每日盘后 | 300-500 × N 只 | 下周走势预测（每只自选） |

---

## 📊 数据来源

| 数据 | 来源 | 频率 | 费用 |
|------|------|------|------|
| 行情（股票/指数/期货） | yfinance | 实时 | 免费 |
| 宏观（VIX/10Y/油/金/美元） | yfinance | 实时 | 免费 |
| **FRED（2Y/美债/Margin/NFCI）** | fredapi | 每日 | 免费（需 key） |
| 4 张新指标卡 | FRED | 每日 | 免费 |
| Vol/OI PCR | yfinance option_chain | 实时 | 免费 |
| 新闻（多源） | 6 源聚合 | 实时 | Yahoo/Stocktwits/东财 免费 |
| AI | DeepSeek | 每次调用 | ¥0.001/千字 |
| FedWatch | CME SOFR/FF 期货 | 实时 | 免费 |

---

## 🛠️ 常见问题

### Q: 配置了 SERPAPI 还是抓不到新闻？
A: 检查三件事：
1. Secret 名是否大写 `SERPAPI`
2. workflow 的 env 块是否传了 `SERPAPI: ${{ secrets.SERPAPI }}`
3. SerpApi 账户免费额度 100/月，可能用完了 → 用免费 fallback (Yahoo/Stocktwits/东财)

### Q: FedWatch 显示"暂不可用"？
A: v2.1 已修复 numpy bug。如果仍报错，可能是 `SR3=F` / `ZQ=F` 在你所在地区 yfinance 拿不到。改用 `^IRX`（13W 国债）作为隐含利率代理即可。

### Q: 下游 Streamlit Cloud 看到的数据不更新？
A: 1) 检查 GitHub Actions 是否成功跑完（Actions tab）；2) 确认 `data/` 目录被 commit + push；3) Streamlit Cloud 默认会重启服务读新数据。

---

## 🛡️ 免责声明

本工具仅供研究和教育用途，不构成任何投资建议。所有 AI 输出需人工复核，市场数据存在延迟或错误可能。
