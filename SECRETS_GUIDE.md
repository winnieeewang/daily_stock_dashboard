# 公开（public）仓库的 API Key 安全实践

> 适用：本项目同时部署在 **GitHub Actions（CI）** 与 **Streamlit Cloud（public 网页）** 两类公开环境。
> 核心目标：**任何 API key 都不出现在代码、git 历史、公开的配置文件中。**

---

## 0. 威胁模型：为什么 public 仓库必须严防

| 风险点 | 后果 |
|---|---|
| key 硬编码进 `.py` 并 push | 任何人都可 `git clone` 看到，且 `git log -p` 能翻到历史 |
| 含 key 的 `.env` / `secrets.toml` 误提交 | 同上，且 CI 日志可能回显 |
| frontend 代码里暴露 key | 浏览器 F12 → Network / Sources 直接拿走 |
| 把 key 写进公开 issue / PR / chat | 搜索引擎会索引 |

**结论**：key 只能存在于「平台托管的加密 Secrets」中，运行时注入，用完即弃，进程内存中使用，永不落盘 / 不打印明文。

---

## 1. 两种部署场景的配置方式（明确区分）

### 场景 A · GitHub Actions（CI，跑 `stock_dashboard.py`）

- **secret 在哪**：GitHub repo → **Settings → Secrets → Actions**（加密托管）
- **如何到代码**：workflow 用 `${{ secrets.NAME }}` 注入为**环境变量**，仅本次 job 运行期内存在
- **本地 / 普通 clone 能看到吗**：**不能**。本地没有这些变量；fork/外部 PR 默认也拿不到 repo secret（防供应链攻击）
- **日志安全**：`${{ secrets.* }}` 在 Actions 日志中永远显示为 `***`

```yaml
# .github/workflows/daily.yml 节选
- name: 跑全量数据采集
  env:
    FRED_API:           ${{ secrets.FRED_API }}
    DEEPSEEK_API_KEY:   ${{ secrets.DEEPSEEK_API_KEY }}
    SERPAPI:            ${{ secrets.SERPAPI }}
    TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
    TELEGRAM_CHAT_ID:   ${{ secrets.TELEGRAM_CHAT_ID }}
  run: python stock_dashboard.py
```
> 注：本项目 `daily.yml` 已注入上述变量。`ALPHA_API` 同理——在 repo Secrets 添加后补一行 `ALPHA_API: ${{ secrets.ALPHA_API }}` 即可。

### 场景 B · Streamlit Cloud（public 部署，跑 `app.py`）

- **secret 在哪**：Streamlit Cloud → 你的 app → **Settings → Secrets**（加密托管，显示时自动遮蔽）
- **如何到代码**：运行时由 Streamlit 注入为 `st.secrets`，不在代码 / 仓库中
- **本地开发**：用 `.streamlit/secrets.toml`（已被 `.gitignore` 排除），结构与云端一致

```toml
# .streamlit/secrets.toml（仅本地，绝不提交）
DEEPSEEK_API_KEY = "sk-xxx"
FRED_API          = "xxx"
SERPAPI           = "xxx"
```

### 统一读取入口

两个场景的 key 名字完全一样，代码**一处读取**：

```python
import utils as U
key = U._get_secret("DEEPSEEK_API_KEY")   # 先查 os.environ（CI/本地 .env），再查 st.secrets（Streamlit）
```

`secrets_loader.py` 的 `detect_runtime()` 会告诉你当前处于哪个场景，并打印对应说明。

---

## 2. 三大保护机制（落地清单）

### 机制 1 — GitHub Actions Secrets 注入环境变量 ✅
- [x] repo Settings → Secrets → Actions 添加每个 key（名字 = 程序里的 env 名）
- [x] workflow 用 `${{ secrets.NAME }}` 注入，不写死在 `run:` 脚本里
- [x] fork/外部 PR 默认禁用 repo secret（GitHub 默认行为）

### 机制 2 — Streamlit Cloud Secrets 管理 ✅
- [x] app Settings → Secrets 添加 key（名字 = `st.secrets` 的键）
- [x] 本地用 `.streamlit/secrets.toml`，**绝不**把 `secrets.toml` 提交（已在 `.gitignore`）
- [x] 前端代码（`.py` 经 streamlit 渲染的 HTML）里**只显示脱敏值**，不发真实 key 到浏览器

### 机制 3 — 避免把含 key 的配置文件提交到 public 仓库 ✅
- [x] **只提交模板**：`.env.example` / `.streamlit/secrets.toml.example`（占位符，无真实值）
- [x] **真实文件被忽略**：`.gitignore` 已含 `secrets.toml`、`.env`
- [x] **自检脚本**：`secrets_loader.py` 的 `scan_repo_for_leaks()` 会扫描已跟踪文件，发现真实 key 即报警
- [x] **提交前检查**：`git status` 确认不会把 `.env` / `secrets.toml` 加进来

```bash
# 验证哪些敏感文件被 git 忽略（应列出 .env / secrets.toml）
git check-ignore -v .env .streamlit/secrets.toml
```

---

## 3. `secrets_loader.py` 使用说明

统一加载器，演示「双场景 + 安全读取 + 自检」。

```bash
# 1) 本地（无 key）→ 显示脱敏状态 + 自检 + 提示缺失必需项
python secrets_loader.py

# 2) 用 .env 本地开发
python -m dotenv secrets_loader.py        # 需 pip install python-dotenv

# 3) CI 中运行（已被 daily.yml / secrets-demo.yml 调用思路一致）
python secrets_loader.py
```

**输出含义**：
- 环境识别：`github_actions` / `streamlit_cloud` / `local` + 该场景的配置说明
- 每个 key 脱敏显示（`sk-t***************cdef`），**绝不**打印明文
- 区分「必需缺失 / 可选缺失」：缺 `DEEPSEEK_API_KEY` 直接退出码 1，其余降级
- 安全自检：`scan_repo_for_leaks()` 扫描已跟踪文件，命中真实 key 即报警

---

## 4. 防护 Checklist（每次发版前过一遍）

- [ ] 没有把真实 key 写进任何 `.py` / `.yml` / `.toml` / `.md`
- [ ] `.env` 与 `secrets.toml` 在 `.gitignore`（已确认）
- [ ] 只提交了 `.env.example` / `secrets.toml.example`（占位符）
- [ ] GitHub Actions Secrets 已添加且 workflow 用 `${{ secrets.NAME }}` 注入
- [ ] Streamlit Cloud Secrets 已添加
- [ ] 前端只渲染脱敏值，不发真实 key 到浏览器
- [ ] 跑过 `python secrets_loader.py` 自检无泄露告警

---

## 5. 万一泄露了怎么办（应急）

1. **立即在提供方后台 Revoke / 重置 key**（最快止血）
2. **在 GitHub / Streamlit 删除并重新生成对应 secret**
3. **清理 git 历史**（key 曾提交过时）：
   ```bash
   pip install git-filter-repo
   git filter-repo --replace-text <(echo 'sk-xxx==>REDACTED')
   git push --force --all
   ```
4. 轮换后更新两个平台的 secret 值，重新部署

---

最后更新：2026-08-05 · 配合 v3.2.1 部署修复
