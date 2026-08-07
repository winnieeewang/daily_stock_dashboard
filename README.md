# 📈 Winnie's Daily Stock Dashboard

> **Investment Copilot v3.0** — 面向专业投资者的 AI 增强型多市场股票分析看板。
> 美股 / A股 / 港股 三市场覆盖，FRED 宏观 + 多源新闻 + 底部信号灯 + AI 炒手对战 + 双 LLM 研判。
> Author: 🐻 Winnie Wang

---

## 🆕 v3.0 主要升级（v2.5 → v3.0）

### 一、UI 分层重构：结论层 + 证据层 + 钻取层

| 层级 | 可见性 | 内容 |
|------|--------|------|
| **第0层 · 结论区** | 默认唯一可见，无需滚动 | 组合策略 Portfolio Context（多头/空头/震荡判定）+ 今日重点 + Top 机会/风险 + 实时行情速览条 |
| **第1层 · 证据层** | 4 主题 Tab | 📊 宏观（情绪/VIX/全球指数/利率/风险雷达/底部信号灯）/ 🧩 结构（US/CN/HK分段+热力图+维科夫扫描）/ 📅 事件（经济日历+FedWatch）/ 🧠 研判（Morning Brief/Evening Recap/智能荐股） |
| **第2层 · 钻取层** | 个股深度页 | 概览卡 + AI 摘要 + 3 Subtab（K线技术面/策略与风险/AI研判） |

### 二、底部信号灯（Bottom Signal）— 0-4 分底部确信度

全新模块 `bottom_signal.py`，分层计算：

```
底部确信度(0-4) = 宏观环境分(0-2, 全市场统一) + 个股结构分(0-2, 逐股不同)
宏观环境分 = 监管恐慌命中(0/1) + 杠杆去化命中(0/1)
个股结构分 = 拥挤出清命中(0/1) + 估值资金综合命中(0/1)
```

**交通灯语义**：🔴 0-1分（观望）/ 🟡 2分（部分确认）/ 🟢 3-4分（多维度共振）

**已实现的维度**：
- ✅ 热力图离散度（过去60天分布比较）
- ✅ 同板块 PE 相对排名（后25%命中）
- ✅ PE 历史分位框架（`data/pe_history.json` 逐日累积，满6个月启用）
- ✅ 政策新闻恐慌关键词计数（紧急/平准基金/国家队/限制卖空/熔断/央行声明）
- ✅ PEG<1 判断（成长股性价比）
- ✅ 维科夫 SC（抛售高潮）复用检测
- ✅ 避险资产联动（GLD/TLT/DXY 30日滚动相关系数）
- ✅ OI PCR 历史序列化（`data/options_pcr_history.json`）
- ✅ FedWatch 非议息窗口检测（紧急降息概率）

### 三、🤖 AI 炒手对战（AI Trader Battle）— KIMI vs DeepSeek

虚拟 $100万 独立账户，双模型每日盘后独立决策、长期跟踪对比。

**核心文件**：`ai_traders.py`

| 功能 | 说明 |
|------|------|
| **候选池** | S&P500 + 纳指100 成分股维科夫/多因子扫描 Top15 |
| **简报生成** | 账户状态 + 宏观快照 + 候选池（含底部信号灯分数） |
| **双模型决策** | KIMI / DeepSeek 各自独立调用 `U._call_llm()` 输出 JSON 交易指令 |
| **护栏检查** | 代码真实存在(yfinance校验)、只做多、单标的≤总资产25%、现金/持仓上限自动裁剪 |
| **模拟成交** | BUY/SELL 更新 portfolio.json，每日 mark-to-market 算 NAV |
| **状态持久化** | `data/ai_traders/{kimi,deepseek}/portfolio.json + nav_history.csv + trades.jsonl` |
| **回测验证** | `backtest_threshold.py` — 统计评分≥60 vs <60 的20日收益率、胜率、最大回撤 |

**前端展示**：Dashboard 新增第 5 个 Tab「🤖 AI炒手」
- 净值曲线对比（归一化到100）
- 当前持仓表 + 现金/净值概览
- 交易日志时间线（最近20笔，可展开看 reasoning）
- 候选池 vs 模型自选胜率对比表

### 四、实时行情深度集成

| 市场 | 实时数据源优先级 |
|------|----------------|
| A股 | 同花顺 Financial-API（云端Key）→ 东方财富 push2 → 腾讯 gtimg → akshare |
| 港股 | 富途 OpenD（本地）→ 东方财富 → 腾讯 gtimg → 新浪 |
| 美股 | yfinance（日K）+ 实时报价（多源降级） |

**新增实时维度**：
- 当日分时折线（`fetch_intraday_trend`，盘中刷新）
- 五档盘口（`fetch_order_book`，A股/港股）
- 逐笔成交明细（`fetch_tick_detail`，20笔带方向 BUY/SELL/NEUTRAL）

### 五、数据层 10 模块清单

| # | 模块 | 职责 | 关键文件 |
|---|------|------|----------|
| ① | 宏观数据层 | 利率/杠杆/波动率/期权情绪 | `utils.py`: FRED系列/MarginDebt/FedWatch |
| ② | 新闻舆情层 | 宏观/政策/个股/公告/龙虎榜 | `utils.py`: 6源聚合/政策新闻/公告/龙虎榜 |
| ③ | 行情与实时层 | 实时报价/盘口/分时/逐笔 | `utils.py`: 东财/腾讯/新浪多源降级 |
| ④ | A股数据层 | 指数/热力图/K线/资金流 | `utils.py`: akshare/东财/腾讯 |
| ⑤ | 港股数据层 | 港股日线/实时报价 | `utils.py`: akshare/富途/东财 |
| ⑥ | 美股/ETF数据层 | 历史K线/新闻/指数 | `utils.py`: yfinance/Finnhub/Stocktwits |
| ⑦ | 技术指标与评分层 | 组合/风险/选股/观察位 | `screener.py` + `risk.py` + `bottom_signal.py` |
| ⑧ | AI研判层 | DeepSeek/OpenRouter LLM生成 | `utils.py`: `_call_llm()` |
| ⑨ | 数据管线持久化层 | 每日CI生成静态数据 | `stock_dashboard.py` + GitHub Actions |
| ⑩ | 外部增强层(可选) | 本地/Key增强，缺失不影响主流程 | `utils_ths.py`(同花顺) / 富途OpenD |

---

## 📁 项目结构

```
.
├── app.py                      # Streamlit 主入口（Dashboard + 个股深度 + 跨资产 + 新闻 + 诊断 + 使用说明）
├── stock_dashboard.py          # 每日跑批：数据采集 + AI报告 + 底部信号 + PCR序列化
├── ai_traders.py               # 🤖 AI炒手：KIMI vs DeepSeek 虚拟交易对战
├── backtest_threshold.py       # 回测验证：智能荐股60分阈值胜率统计
├── bottom_signal.py            # 🚦 底部信号灯：0-4分底部确信度计算
├── screener.py                 # 选股三层架构：维科夫事件 + 多因子评分
├── risk.py                     # R倍数止盈 + 杠杆强平监控
├── utils.py                    # 数据获取：新闻/F&G/FedWatch/日历/热力图/PCR/实时行情/LLM
├── utils_ths.py                # 同花顺 Financial-API 适配层（可选增强）
├── secrets_loader.py           # Secret 双路读取（os.environ + st.secrets）
├── requirements.txt            # Python 依赖
├── STREAMLIT_DEPLOY.md         # 部署文档（含同花顺云端接入步骤）
├── data/                       # 自动生成的数据文件
│   ├── stocks.csv              # 个股行情 + 技术指标
│   ├── macro.csv               # 宏观指数
│   ├── sox.csv / sp500.csv     # 行业/指数信号
│   ├── cards.json              # AI决策卡片
│   ├── bottom_scores.json      # 底部信号灯批量计算结果
│   ├── pe_history.json         # PE历史分位（逐日累积）
│   ├── dispersion_history.json # 热力图离散度历史
│   ├── options_pcr.json        # 最新PCR快照
│   ├── options_pcr_history.json# PCR时间序列（180天）
│   ├── safe_haven_history.json # 避险资产价格历史（90天）
│   ├── news.json               # 多源新闻聚合
│   ├── leverage_risk.json      # 杠杆强平监控
│   ├── predictions.json        # 下周走势预测
│   ├── morning_brief.md        # 盘前早报
│   ├── evening_recap.md        # 盘后总结
│   └── ai_traders/             # AI炒手状态目录
│       ├── kimi/               # portfolio.json / nav_history.csv / trades.jsonl
│       └── deepseek/           # portfolio.json / nav_history.csv / trades.jsonl
└── .github/workflows/
    ├── daily.yml               # 每日盘后全量更新 + AI炒手决策
    └── (legacy morning/evening)
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
# 推荐配置（让 Dashboard 全部功能满血运行）
FRED_API = "your-fred-key"
DEEPSEEK_API_KEY = "your-deepseek-key"
OPENROUTER_API_KEY = "your-openrouter-key"
SERPAPI = "your-serpapi"
HITHINK_FINANCE_API_KEY = "your-hithink-key"   # A股实时增强（同花顺云端）
EOF

# 4. 采集数据（首次或每日更新）
python stock_dashboard.py              # 全量
python ai_traders.py --dry-run         # AI炒手试运行（不实际成交）

# 5. 启动看板
streamlit run app.py
```

打开 http://localhost:8501

### 本地启用富途 OpenD（A股/港股 L2 实时）

1. 安装并登录 [OpenD](https://openapi.futunn.com/)（需开通 OpenAPI 权限）
2. `pip install "futu-api>=8.0.0"`
3. `.streamlit/secrets.toml` 添加：
   ```toml
   USE_FUTU = "true"
   FUTU_OPEND_HOST = "127.0.0.1"
   FUTU_OPEND_PORT = "11111"
   ```
4. 确保 OpenD 运行后启动 `streamlit run app.py`

---

## ⚙️ GitHub Actions 自动跑

| 工作流 | 文件 | Cron (UTC) | 任务 |
|--------|------|-----------|------|
| **Daily** | `.github/workflows/daily.yml` | `30 20 * * 1-5` | 全量数据采集 + AI报告 + AI炒手每日决策 |

**GitHub Secrets 清单**：

| Secret 名 | 优先级 | 用途 |
|-----------|--------|------|
| `FRED_API` | ⭐⭐⭐ | 4张宏观指标卡 |
| `DEEPSEEK_API_KEY` | ⭐⭐⭐ | AI报告/预测/炒手决策（首选LLM） |
| `OPENROUTER_API_KEY` | ⭐⭐ | AI兜底（DeepSeek失败时切Claude 3.5 Sonnet） |
| `SERPAPI` | ⭐⭐ | 主力新闻（100/月免费） |
| `HITHINK_FINANCE_API_KEY` | ⭐⭐ | A股实时行情增强（同花顺云端） |
| `TELEGRAM_BOT_TOKEN` | 可选 | Telegram推送 |
| `TELEGRAM_CHAT_ID` | 可选 | 推送目标 |

> 💡 **最低成本方案**：只配 `FRED_API` + `DEEPSEEK_API_KEY` 即可让 Dashboard 核心功能全部工作。

---

## ☁️ 云部署

### Streamlit Cloud（推荐）

1. 推送到 GitHub 仓库（已含 `.github/workflows/daily.yml`）
2. https://share.streamlit.io → New app → 选仓库/分支/`app.py`
3. **App settings → Secrets** 粘贴 TOML（与本地 `.streamlit/secrets.toml` 同格式）
4. 数据由 GitHub Actions 每天自动刷新并 `git push` 回 `data/`，Streamlit Cloud 自动同步

> ⚠️ `*.streamlit.app` 在中国大陆不稳定。国内用户建议用 CloudStudio / 腾讯云等国内节点反代。

---

## 🛠️ 常见问题

### Q: Dashboard 结论区显示空白？
A: 检查 `data/stocks.csv` 是否存在且包含数据。首次使用或数据过期时运行 `python stock_dashboard.py` 重新生成。

### Q: 配置了同花顺 Key 但 A股仍不显示实时？
A: Streamlit Cloud 的 Secrets 面板需配 `HITHINK_FINANCE_API_KEY`（不是 GitHub Repository Secrets）。本地则写在 `.streamlit/secrets.toml`。

### Q: AI炒手没有数据？
A: 首次需手动运行 `python ai_traders.py` 或等 GitHub Actions 自动跑。`--dry-run` 可先测试不实际成交。

### Q: 港股代码 Yahoo 返回 404？
A: 部分港股代码在 yfinance 上格式不兼容（如 07709.HK / 01879.HK）。已接入东财/腾讯/新浪多源降级，数据仍会显示。

---

## 🛡️ 免责声明

本工具仅供研究和教育用途，不构成任何投资建议。所有 AI 输出需人工复核，市场数据存在延迟或错误可能。
