# 📈 Investment Copilot

> **Winnie's Daily Stock Dashboard** v2.5 — 面向专业投资者的多模块、自动化、AI 增强的 Streamlit 看板。
> 美股 / A股 / 港股 三市场覆盖，FRED 宏观 + 多源新闻 + DeepSeek/OpenRouter 双 LLM + 宏观风险雷达 + 智能荐股 + ZhuLinsen 策略拼接。
> Author: 🐻 Winnie Wang

---

## 🆕 v2.4 主要升级（v2.3 → v2.4）

| 模块 | v2.3 | v2.4 (current) |
|------|------|----------------|
| **组合策略卡** | 无 | ✅ Dashboard 顶部新增「🎯 组合策略 Portfolio Context」：根据自选股今日涨跌判断多头占优 📈 / 空头占优 📉 / 震荡 ⚖️ / 空仓 💤；显示持仓数、平均涨跌、ETF 持仓、多空空仓占比与堆叠条 |
| **全球主指数** | 仅在 3 个 tab 里 | ✅ Dashboard 新增一行 5 卡：🇨🇳 上证 (000001.SS) / 🇭🇰 恒指 (^HSI) / 🇺🇸 标普 / 纳指 / 道指，30 日走势自动打 🐂/➡️/🐻 微表情 |
| **VXN 恐慌指数** | 仅 VIX | ✅ 情绪行 5→6 卡，新增 VXN (^VXN) 卡 + 联动 emoji（😱/😟/😐/😀）；自建情绪指数、F&G 标题也按数值带微表情 |
| **宏观风险雷达** | 无 | ✅ 新增「🚨 宏观风险雷达」6 组（Regime / Rates / Risk / Ratios / Cross / A-Share），每组绿/黄/红灯 + 关键指标 + emoji + 综合评分 |
| **VIX vs VXN 走势图** | 无 | ✅ 跨资产对比页新增「😱 恐慌指数对比」双 Y 轴图，与上面跨资产图共用同一时间维度；并提供 30 警戒带 + 最新/利差/占比指标 |
| **微表情体系** | 散落 emoji | ✅ 统一 `U.emoji_for_sentiment / emoji_for_panic / emoji_for_market_regime / emoji_for_dominance`，所有数值卡都按阈值带表情 |
| **ZhuLinsen YAML 拼接到 Morning Brief** | 无 | ✅ 新增 `U.build_morning_brief_prompt()`：自动下载 ZhuLinsen/daily_stock_analysis → 解析 `strategies/*.yaml` → 把 `instructions` 拼到 prompt 末尾（详见下方"如何拼接"） |

## 🆕 v2.5 主要升级（v2.4 → v2.5）

| 模块 | 说明 |
|------|------|
| **自选股扩容** | watchlist 由 28 → 41 只。新增港股 **00293 国泰航空 / 03690 美团-W / 01138 中远海能 / 03968 招商银行**，美股 **EUV(Corgi Lithography) / RKLB / GEV / FUTU / UNH / NVO / NFLX / JNJ / INTU** |
| **公司名映射** | 新增 `U.STOCK_NAMES` 中文名表 + `get_stock_name()`；所有股票展示均「代码 · 公司名」（Top5 卡 / 自选列表 / 今日重点 / 荐股榜） |
| **📊 智能荐股引擎** | 新增 `U.recommend_stocks()` + Dashboard「📊 智能荐股」板块。综合 **技术面**(MA20/60 趋势) + **估值**(PE) + **动量**(当日) + **杠杆止损位**(≈2.2×ATR 或 7% 下限)，输出三类：**日内做T候选**(高波动 ATR≥2.5%) / **中期持股推荐**(综合评分高+看多) / **综合买入信号**(评分≥60)，每只带 🐂看多/🐻看空/➡️震荡 与参考止损位 |
| **小熊维尼图标** | `Author: Winnie Wang` 旁加 🐻 小熊 SVG 图标（页脚 + README + PWA 小程序） |
| **手机网页版 PWA 同步** | `pwa/` 同步上线：41 只快照 + 同名映射 + 等价荐股算法 + 公司名 + 🐻 图标，部署在 CloudStudio（见下方链接） |

> ⚠️ 荐股为**量化初筛**，仅供参考，不构成投资建议。当前评分权重为 技术面 50% + 估值 25% + 动量 25%；**消息面**暂未纳入量化评分（已有多源新闻模块，后续可接入情绪打分）。

### 🎯 怎么用 ZhuLinsen YAML 拼到 Morning Brief

`utils.py` 已内置三个配套函数：

```python
# 1) 一次性下载/解析 ZhuLinsen 仓库的策略 YAML
from utils import build_morning_brief_prompt, MORNING_BRIEF_PROMPT

context = {  # 喂给原 MORNING_BRIEF_PROMPT 的字段
    "date": "2026-07-31",
    "market_status": "Open",
    "fear_greed_score": 62,
    "fear_greed_label": "贪婪",
    # ... 其他 MORNING_BRIEF_PROMPT 字段
}

# 2) 一步到位：基础 prompt + 策略 instructions 拼接
final_prompt = build_morning_brief_prompt(
    context,
    include_zhu_linsen=True,
    only_categories=["trend", "reversal"],   # 可选：按分类过滤
    max_chars=3000,                          # 防止 prompt 过长
)
# 3) 喂给 LLM（与现有 render_morning_brief 同样的 _call_llm 调用）
from openai import OpenAI
resp = OpenAI(api_key=...).chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是专业卖方策略师…"},
        {"role": "user", "content": final_prompt},
    ],
)
```

底层会自动从 GitHub zip 下载 `https://github.com/ZhuLinsen/daily_stock_analysis/archive/refs/heads/main.zip` 到 `vendor/daily_stock_analysis/`，解析 `strategies/*.yaml`（每个 YAML 都有 `name / display_name / category / instructions` 字段），把 `instructions` 拼成：
```
【已挂载的策略框架（盘前/盘中请参照下列框架评估）】
--- 均线多头排列 (trend) ---
...instructions 文本...
--- 缠论中枢 (framework) ---
...instructions 文本...
```
塞到 `MORNING_BRIEF_PROMPT` 末尾。需要 `PyYAML`（已加进 `requirements.txt`）。

> ⚠️ **首次运行会从 GitHub 下载 zip**（约几 MB），下完缓存在 `vendor/daily_stock_analysis/`；下次直接复用。可改为 `git submodule add https://github.com/ZhuLinsen/daily_stock_analysis vendor/daily_stock_analysis` 走 git 方式。

---

## 🆕 v2.3 主要升级（v2.2 → v2.3）

| 模块 | v2.2 | v2.3 (current) |
|------|------|----------------|
| **Dashboard 分段** | 美股/港股/A股 混在一屏 | ✅ **🌍 全球市场总览按市场分段**：🇺🇸 美股 / 🇨🇳 A股 / 🇭🇰 港股 三个独立 Tab |
| **FRED/SerpApi 连接** | GitHub Secrets 配了却 "数据缺失" | ✅ **根因修复**：新增 `_get_secret()` 双路读 key（os.environ + st.secrets），Streamlit Cloud 从此能读到 FRED_API / SERPAPI |
| **A股数据** | akshare 兜底 | ✅ akshare 内置采集全景/K线/涨跌榜，`utils.akshare_available()` 在诊断页实时显示 |
| **4 张新指标卡** | 依赖 `extra_indicators.json` 静态文件 | ✅ **实时回退**：json 缺失时自动现场 `fetch_*()` 拉 FRED，不再卡 "需 FRED_API" |
| **LLM 兜底** | 仅 DeepSeek | ✅ 新增 **OPENROUTER_API_KEY**，DeepSeek 失败时自动切 OpenRouter（默认 Claude 3.5 Sonnet） |
| **诊断页** | 无 | ✅ 新增 `page_diagnostics()`，列出 FRED_API/DEEPSEEK/SERPAPI/FINNHUB/NEWSAPI/OPENROUTER + akshare 状态 |

### 🔧 v2.3 关键修复：Streamlit Cloud 读不到 Secret（必看）
**症状：** 在 GitHub → Settings → Secrets 里配了 `FRED_API`、`SERPAPI`，部署后 Dashboard 仍显示「数据缺失（需 FRED_API）」。

**根因：** Streamlit Cloud **只把 Repository Secrets 注入 `st.secrets`**，不会注入 `os.environ`。旧代码 `fred_key = os.environ.get("FRED_API")` 在云端永远拿到空串。

**修复（`utils.py` / `app.py`）：** 所有 key 统一走 `_get_secret()`，先读 `os.environ`（本地 / GitHub Actions），读不到再读 `st.secrets`（Streamlit Cloud）：
```python
def _get_secret(name: str, default: str = "") -> str:
    v = os.environ.get(name, "")
    if not v and _HAS_STREAMLIT:
        try:
            v = st.secrets.get(name, "")
        except Exception:
            v = ""
    return v
```
> 本地跑 `python stock_dashboard.py` 用 os.environ；Streamlit Cloud 跑 app.py 用 st.secrets。两端都通。

### 💡 可选集成（未内置，按需二开）
- **嵌入 `github.com/ZhuLinsen/daily_stock_analysis`**：可作为 A股/港股因子分析的参考实现，克隆后把它的 `stock_analysis` 模块 import 进 `utils.py` 即可扩展。当前仓库未直接依赖它，留作后续增强。

---

## 🆕 v2.2 主要升级（v2.1 → v2.2）

| 模块 | v2.1 | v2.2 (current) |
|------|------|----------------|
| **App 名称** | Investment Copilot | **Winnie's Daily Stock Dashboard**（标题栏 + 侧栏 + 顶栏统一） |
| **热力图** | 美股 + 港股 2 列 | ➕ **A股全市场热力图**（申万行业，3 列并排） |
| **新闻源** | SerpApi/Finnhub/NewsAPI/Yahoo/Stocktwits/东财 | ➕ **雪球 + 同花顺快讯 + 美股/港股无API RSS**（CNBC/MarketWatch/Seeking Alpha/Investing/AAStocks/HKET） |
| **自选股** | 美股 13 + 港股 3 | ➕ **A股 10 只**（强一股份/三环集团/电连技术/000426/002624/601872/601975/002258/001331/600150）+ 港股 07709/00981 |
| **A股采集** | 仅美股/港股 | ✅ akshare 回退抓取 A股（.SS/.SZ），yfinance 兜底 |
| **个股分层** | 单一 selectbox（全部混排） | ✅ **两层级：第一层市场（港股/A股/美股）→ 第二层所选市场内选股** |

### 无 API 的新闻/消息源推荐（美股 / 港股 / A股）
- **美股**：Yahoo Finance RSS（已有）、CNBC、MarketWatch、Seeking Alpha、Investing.com
- **港股**：东方财富港股（已有）、AAStocks、HKET（香港经济日报）
- **A股**：东方财富（已有）、**同花顺快讯**、**雪球**
- 以上全部免费、无需 API key；雪球接口偶需 cookie，失败时自动跳过，不影响其他源。

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
DEEPSEEK_API_KEY = "your-deepseek"  # AI 报告 / 下周预测（首选 LLM）
OPENROUTER_API_KEY = "your-openrouter"  # AI 报告 / 下周预测（DeepSeek 失败时兜底，默认 Claude 3.5 Sonnet）
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
| **`DEEPSEEK_API_KEY`** | ⭐⭐⭐ 强烈推荐 | AI 报告 / 预测（首选 LLM） | https://platform.deepseek.com/ |
| **`OPENROUTER_API_KEY`** | ⭐⭐ 推荐 | AI 报告 / 预测兜底（DeepSeek 失败自动切 Claude 3.5 Sonnet） | https://openrouter.ai/keys |
| **`SERPAPI`** | ⭐⭐ 推荐 | 主力新闻（100/月免费） | https://serpapi.com/ |
| `FINNHUB_API` | ⭐ 可选 | 美股个股新闻（60/min） | https://finnhub.io/register |
| `NEWSAPI_KEY` | ⭐ 可选 | 宏观新闻（100/天） | https://newsapi.org/register |
| `TELEGRAM_BOT_TOKEN` | 可选 | Telegram 推送预警 | @BotFather |
| `TELEGRAM_CHAT_ID` | 可选 | 推送目标 | @userinfobot |

> 💡 **最低成本方案**：只配 `FRED_API` + `DEEPSEEK_API_KEY` + `SERPAPI` 就能让 9 张 Dashboard 卡 + AI 报告 + 多源新闻全部工作。免费源（Yahoo RSS、Stocktwits、东财）始终可用。加 `OPENROUTER_API_KEY` 可让 AI 报告在 DeepSeek 偶发故障时不中断。

---

## ☁️ 云部署（让 Dashboard 24h 在线）

本项目为 **Streamlit 应用 + GitHub Actions 定时采集** 架构，部署后无需本地电脑常开，数据由 GitHub Actions 每天自动刷新并推回仓库，看板自动同步。

### 方案 A：Streamlit Cloud（海外，最简单）

1. 把本项目推到 GitHub 仓库（已含 `.github/workflows/` 三个定时任务）。
2. 打开 https://share.streamlit.io → **New app** → 选仓库、分支、`app.py` 作为入口。
3. **配置 Secrets**：在 Streamlit Cloud 的 `App settings → Secrets` 里粘贴 TOML（与本地 `.streamlit/secrets.toml` 同格式）：
   ```toml
   FRED_API = "xxxx"
   DEEPSEEK_API_KEY = "xxxx"
   OPENROUTER_API_KEY = "xxxx"
   SERPAPI = "xxxx"
   FINNHUB_API = "xxxx"
   NEWSAPI_KEY = "xxxx"
   ```
   > ⚠️ **关键**：这些 key 必须填在 Streamlit Cloud 的 **Secrets** 里，不是在 GitHub Repository Secrets。代码用 `_get_secret()` 走 `st.secrets` 读取（详见 v2.3 修复说明）。GitHub Repository Secrets 只给 GitHub Actions 用。
4. **改 App 名称（可选）**：Streamlit Cloud 默认的 URL 形如 `https://dailystockdashboard-ruxrka5bifsifhs4iy67s9.streamlit.app`，可在 `Settings → App name` 改成 `winnies-dailystockdashboard`，URL 即变为 `https://winnies-dailystockdashboard.streamlit.app`（更短、好记、无随机后缀）。**这一步只能在 Streamlit Cloud 后台手动改，代码改不了。**
5. **数据自动更新**：GitHub Actions 每天按 cron 跑采集 → `git commit & push` 回 `data/` → Streamlit Cloud 检测到新 commit 自动重启并读取最新数据。**无需手动重新部署。**

> 📌 部署后如果 Dashboard 仍「数据缺失」，90% 是因为 Secrets 没填在 Streamlit Cloud 的 Secrets 面板（见上第 3 步），而不是 GitHub Secrets。诊断页（侧栏「🔧 诊断」）会列出每个 key 是否成功读取。

### 方案 B：CloudStudio / 国内云（国内可访问，推荐给国内用户）

`*.streamlit.app` 在**中国大陆经常被墙/超时**，手机和办公网都打不开。若要国内稳定访问，把静态看板反向代理到国内节点即可：

- **腾讯云 CloudStudio**：把本仓库部署为静态/Python 应用，分配 `*.cloudstudio.app` 域名，**国内直连可用**。本机可用 `cloudstudio-deploy` 技能一键上传 `app.py` + `data/` 并起服务。
- **阿里云 / 腾讯云轻量应用服务器**：用 Nginx 反代 Streamlit（`streamlit run app.py --server.headless true`），再绑国内已备案域名。成本约 ¥50–100/月。
- **前提**：A股/港股数据（东方财富、雪球、同花顺）本身在国内访问更快；海外数据源（yfinance/FRED）在国内部署时可能需代理。

---

## 📱 国内访问 & 小程序方案

### 为什么手机打不开？
`*.streamlit.app` 在大陆网络不稳定；且 Streamlit 是 **Web 应用**，并非原生 App，不能直接上架 App Store / 应用宝。要在手机上"像 App 一样用"，有三条路：

| 方案 | 难度 | 成本 | 国内可用 | 说明 |
|------|------|------|---------|------|
| **① PWA 添加到主屏** | ⭐ 最低 | 免费 | ⚠️ 取决于部署域名 | 给 app.py 加 manifest + service worker，iOS Safari「添加到主屏幕」/ Android Chrome「安装应用」即可当 App 用。**前提是看板部署在国内可访问的域名（见方案 B）。** |
| **② 微信公众号 H5** | ⭐⭐ 低 | 免费/¥300 认证 | ✅ | 订阅号/服务号自定义菜单跳转看板 H5 链接，粉丝在微信内直接打开。需公众号 + 已备案域名。 |
| **③ 微信小程序 WebView** | ⭐⭐⭐ 中 | 企业主体 + ¥300 认证 | ✅ | 用 `<web-view>` 组件嵌入看板 H5。**硬性要求：小程序须为企业主体、绑定已 ICP 备案域名、在后台配置「业务域名」。** 个人主体无法用 web-view。 |

### 推荐落地路径（分步）
1. **先保证国内能访问**：用方案 B（CloudStudio 或国内云）把看板部署到国内域名。这是后面所有手机方案的前提。
2. **最快体验「App 化」**：在部署好的国内域名上加 PWA（manifest + SW），手机「添加到主屏幕」，零额外资质。
3. **要进微信生态**：注册公众号（方案 ②），把看板 H5 作为菜单/图文跳转；若要更重的小程序外壳，走方案 ③（需企业主体 + 备案）。

> 💡 **注意**：Streamlit 的 `st.secrets`、交互式组件在 WebView 内都能正常工作；但微信对 `web-view` 有域名白名单与备案校验，部署前先把域名备案好，否则小程序审核会被拒。

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

## 🧠 AI 报告（DeepSeek / OpenRouter 双 LLM）

> v2.3 起 LLM 调用走 `utils._call_llm()`：**优先 DeepSeek**，失败/超时自动切 **OpenRouter（默认 `anthropic/claude-3.5-sonnet`）**。配置 `OPENROUTER_API_KEY` 即可获得兜底，AI 报告与下周预测都不会因单一服务故障而中断。

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
| AI | DeepSeek（首选）/ OpenRouter（兜底 Claude 3.5 Sonnet） | 每次调用 | DeepSeek ¥0.001/千字；OpenRouter 按模型计费 |
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

### Q: GitHub Repository Secrets 里明明配了 FRED_API / SERPAPI，看板却显示「数据缺失（需 FRED_API）」？
A: **两个 Secrets 不是一回事。** 代码在 Streamlit Cloud 运行时只认 **Streamlit Cloud 后台的 Secrets 面板**（`App settings → Secrets`），不认 GitHub Repository Secrets。GitHub Repository Secrets 只给 GitHub Actions 用。解决办法：把同一组 key 也粘到 Streamlit Cloud 的 Secrets 面板（TOML 格式，见上文「方案 A 第 3 步」）。侧栏「🔧 诊断」页会实时显示每个 key 是否成功读取，可用来排错。

### Q: 手机/国内打不开 `*.streamlit.app`？
A: 该域名在大陆网络不稳定。两种解法：(1) 把看板部署到国内可访问节点（CloudStudio / 国内云，见「☁️ 云部署 方案 B」）；(2) 在已备案国内域名上加 PWA 或包成公众号 H5 / 微信小程序 WebView（见「📱 国内访问 & 小程序方案」）。

---

## 🛡️ 免责声明

本工具仅供研究和教育用途，不构成任何投资建议。所有 AI 输出需人工复核，市场数据存在延迟或错误可能。
