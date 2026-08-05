# Streamlit Cloud 部署 & 排错指南

> 本指南适用于：当你看到 **"Error running app"** / **Build failed** / 页面长时间白屏等部署类问题。

---

## 🔑 Part 1 — API Key 配置（必须，绝不入 git）

### Streamlit Cloud Secrets（强烈推荐）

`Settings → Secrets` 在网站右上角，添加：

```toml
# AI 研判
DEEPSEEK_API_KEY       = "sk-xxxxxxxxxxxx"
OPENROUTER_API_KEY     = "sk-or-xxxxxxxxxxxx"   # 可选，DeepSeek 兜底

# 同花顺 Financial-API（本次新增）
# 注册 https://fuyao.aicubes.cn/admin/ → 获取 key
HITHINK_FINANCE_API_KEY = "your-ths-key-here"

# 美股 / 宏观新闻
FRED_API               = "your-fred-key"
SERPAPI                = "your-serpapi-key"
FINNHUB_API            = "your-finnhub-key"
NEWSAPI_KEY            = "your-newsapi-key"

# 富途（仅本地用，云端勿加）
# USE_FUTU            = "true"
# FUTU_OPEND_HOST     = "127.0.0.1"
# FUTU_OPEND_PORT     = "11111"
```

### 本地开发 — `.streamlit/secrets.toml`

把上面 `toml` 全文保存为 `.streamlit/secrets.toml`（**此文件已加入 .gitignore，绝不入仓**）。

---

## 🔧 Part 2 — 5 步诊断 Streamlit "Error running app"

### Step 1 · 复制云端完整 traceback

1. 打开 https://share.streamlit.io/
2. 进入 dailystockdashboard 项目 → **「Manage app」**
3. **「Logs」** 标签
4. 找到最后一段红字（最近的 Traceback）
5. **整段复制** 贴给我——包含：
   ```
   File "/app/xxx.py", line N, in <module>
       ...
   TypeError: ...
   ```

### Step 2 · 最常见的 4 个云端错误及修复

| # | 现象 | 真因 | 修复 |
|---|---|---|---|
| ❶ | Build failed: `futu-api install failed` 或 Build 超时 | `futu-api` 是 native 包云端装不上 | **已修复**——本仓库 requirements.txt 已移除 futu-api |
| ❷ | ModuleNotFoundError: `No module named 'xxx'` | requirements.txt 没装该依赖 | 在 requirements.txt 加 `xxx>=版本` |
| ❸ | `KeyError: 'FRED_API'` | Streamlit Secrets 没配这个 key | 去 Streamlit Cloud → Secrets 加上 |
| ❹ | `numpy.dtype size changed` 类型错误 | numpy/pandas 版本冲突 | 固定 `numpy<2,>=1.24` 或重新部署冷启动 |

### Step 3 · 检查依赖文件

确认 `requirements.txt` 是这个版本（不是更老的）：

```
yfinance>=0.2.40
pandas>=2.0.0
numpy>=1.24.0
ta>=0.11.0
requests>=2.31.0
fredapi>=0.5.1
openai>=1.0.0
streamlit>=1.30.0
plotly>=5.18.0
akshare>=1.13.0
serpapi>=0.1.0
feedparser>=6.0.10
PyYAML>=6.0
# 注释掉 futu-api（云端不需要）；本地 OpenD 用户手动 pip install futu-api
```

### Step 4 · 主动防御（已添加）

最近的两次大变更已增加了：
- **`_render_tomorrow_watch`** 整体包 try/except → 单只股票崩不会影响整页
- **`compute_tomorrow_watch`** 所有数值计算 + NaN/None 边界已保护
- **`utils_ths.py`** 无 key 时返回空（不抛错）
- **page_diagnostics** 新增 `HITHINK_FINANCE_API_KEY` 检测项

### Step 5 · 重启与冷启动

如果还报错，按顺序试：
1. Streamlit 控制台 → **「Reboot」**
2. 等 30 秒，刷新页面
3. 还不行 → 改 `requirements.txt` 触发重新部署
4. 仍然不行 → 把 Stack 贴给助手

---

## 🚨 Part 3 — 紧急联系 / 升级

拿到完整 stack trace 后贴给我，我会：
- 判断是 import error / runtime error / data error
- 直接改代码并 commit
- 必要时回退到上一个稳定 commit（`ee71289` 是上一稳定版）

---

## ⚠️ Part 4 — 安全红线

**绝对不要**：
- ❌ 把 API Key 写到任何 `.py`、`.toml`（除 `.streamlit/secrets.toml`）、`.md`、`.yaml`、`.json` 文件并 commit 到 git
- ❌ 把含 key 的内容贴到 chat / log
- ❌ 把 `.streamlit/secrets.toml` 加入任何 push 命令

**可以**：
- ✅ 用 Streamlit Cloud → Settings → Secrets 管理所有 key
- ✅ 在本地用 `.streamlit/secrets.toml`（已在 .gitignore）
- ✅ 在 .py 里用 `os.getenv("KEY_NAME")` 读（Key 只存在云端 Secrets）

---

最后更新：2026-08-05 · 新增「同花顺 Financial-API」集成支持
