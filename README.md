# 📈 Investment Copilot

> 投资分析工作台 v2.0 — 面向专业投资者的多模块、自动化、AI 增强的 Streamlit 看板。
> Author: Winnie Wang

---

## 🆕 v2.0 主要升级

| 模块 | v1.0 (initial) | v2.0 (current) |
|------|----------------|----------------|
| **首页** | 单列流式，看板/分析混在一起 | 🏠 Dashboard + 5 个独立页面 |
| **市场全景** | 5 个简单卡片 | F&G 5 因子 / 情绪 / VIX / SOX / 10Y / 今日重点 / Top 机会 / Top 风险 |
| **热力图** | 无 | 美股 (60+ 标的) + 港股 (30+ 标的) 全市场 treemap |
| **新闻** | 只抓百度个股新闻 | 宏观 / 政策 / 个股 三档抓取 + 去重；中美双引擎 |
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
├── utils.py                # 工具模块：新闻/F&G/FedWatch/日历/热力图
├── requirements.txt        # Python 依赖
├── data/                   # 自动生成的数据文件
│   ├── macro.csv           # 宏观指数
│   ├── stocks.csv          # 个股行情 + 技术指标
│   ├── sox.csv             # SOX 半导体信号
│   ├── sp500.csv           # 标普 500 信号
│   ├── cards.json          # AI 决策卡片
│   ├── leverage_risk.json  # 杠杆强平监控
│   ├── news.json           # 新闻（宏观/政策/个股）
│   ├── report.md           # AI 大盘总览
│   ├── weekly_report.md    # 每周总结（周五生成）
│   ├── morning_brief.md    # ☀️ 盘前早报（升级新增）
│   └── evening_recap.md    # 🌙 盘后总结（升级新增）
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
SERPAPI = "your-serpapi-key"
DEEPSEEK_API_KEY = "your-deepseek-key"
FRED_API = "your-fred-key"
ALPHA_API = "your-alphavantage-key"   # 可选，期权 PCR
TELEGRAM_BOT_TOKEN = ""                # 可选，Telegram 推送
TELEGRAM_CHAT_ID = ""
EOF

# 4. 采集数据（首次或每日更新）
python stock_dashboard.py
# 跳过 AI 报告：python stock_dashboard.py --no-brief --no-recap

# 5. 启动看板
streamlit run app.py
```

打开 http://localhost:8501

---

## ⚙️ GitHub Actions 自动跑

`.github/workflows/daily.yml`：

```yaml
name: 每日量化看盘
on:
  schedule:
    - cron: '30 13 * * *'   # UTC 13:30 = 美东盘前（夏令时 09:30 / 冬令时 08:30）
  workflow_dispatch:

jobs:
  run-dashboard:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v6
        with: { python-version: '3.10' }
      - run: pip install -r requirements.txt
      - run: python stock_dashboard.py
        env:
          FRED_API: ${{ secrets.FRED_API }}
          ALPHA_API: ${{ secrets.ALPHA_API }}
          DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
          SERPAPI: ${{ secrets.SERPAPI }}
      - run: |
          git config user.name "github-actions"
          git config user.email "actions@github.com"
          git add -f data/
          git diff --staged --quiet || git commit -m "每日数据更新 $(date +'%Y-%m-%d')"
          git push
```

### GitHub Secrets 清单

| Secret 名 | 必填 | 获取方式 |
|-----------|------|---------|
| `FRED_API` | 否 | https://fred.stlouisfed.org/docs/api/api_key.html |
| `ALPHA_API` | 否 | https://www.alphavantage.co/support/#api-key （期权 PCR） |
| `DEEPSEEK_API_KEY` | 否（但强烈建议） | https://platform.deepseek.com/ |
| `SERPAPI` | 否 | https://serpapi.com/ （新闻/FedWatch/Calendar） |
| `TELEGRAM_BOT_TOKEN` | 否 | @BotFather |
| `TELEGRAM_CHAT_ID` | 否 | @userinfobot |

---

## 🔧 修复点详细说明

### 1. NVDA 拆股异常（历史跨资产对比）

**原因：** `yf.download()` 默认不传 `auto_adjust=True` 时，NVDA 在 2024-06-10 的 10-1 拆股没复权，历史绝对价 $1.20 跟当前 $950 归一化后爆炸。

**修复（utils.py）：**
```python
df = yf.download(sym, period=period, progress=False, auto_adjust=True)  # 关键
```

并新增三档视图：
- **Price**：对数价格轴 + 10Y 收益率副轴
- **Return %**：累计回报（起点 0%）
- **Log Return**：log(price/price[0])

### 2. SerpApi 抓不到宏观/政策新闻

**原因：** 旧版只调用 `fetch_baidu(symbol)`，对 `0700` 这种纯数字搜百度搜不到有用结果，更别说 Fed/CPI。

**修复（utils.py + stock_dashboard.py）：**
- 新增 `fetch_macro_news()` 走 Google News，搜索 10 个宏观关键词
- 新增 `fetch_policy_news()` 走 Google News，搜索 SEC/财政部/中国央行等
- 新增 `fetch_all_news()` 一次性抓全
- 去重逻辑：按 link/title MD5 去重

### 3. 三栏信息层级

**修复（app.py）：** 完整的左/中/右三栏工作台，左侧 1/12、中间 2.2/12、右侧 1/12，详情见 `page_stock_deepdive()`。

---

## 🧠 AI 报告（DeepSeek）

| 报告 | 时机 | 字数 | 风格 |
|------|------|------|------|
| `report.md` | 每日收盘 | 500 | 简明总览 |
| `cards.json` | 每日 | JSON | 结构化决策 |
| `weekly_report.md` | 周五 | 800-1200 | 复盘 + 展望 |
| `morning_brief.md` | 每日盘前 | 500-800 | 卖方策略师 Morning Note |
| `evening_recap.md` | 每日盘后 | 800-1200 | 卖方策略师 EOD Note |

**Morning Brief 示例结构：**
```
=== 1. 隔夜发生了什么 ===
=== 2. 今天的核心主题 ===
=== 3. 关键观察 & 风险 ===
=== 4. 交易思路 ===
```

---

## 📊 数据来源

| 数据 | 来源 | 频率 |
|------|------|------|
| 行情（股票/指数/期货） | yfinance | 实时 |
| 宏观（VIX/10Y/油/金/美元） | yfinance | 实时 |
| FRED（国债/杠杆指数） | fredapi | 每日 |
| 新闻 | SerpApi（Google + 百度） | 实时 |
| 期权 PCR | Alpha Vantage | 每日 25 次 |
| AI | DeepSeek | 每次调用 |
| FedWatch | CME SOFR/FF 期货 | 实时 |

---

## 🛡️ 免责声明

本工具仅供研究和教育用途，不构成任何投资建议。所有 AI 输出需人工复核，市场数据存在延迟或错误可能。
