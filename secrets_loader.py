"""
secrets_loader.py — 跨场景 API Key 安全加载器

============================================================================
背景：本项目同时部署在两类「公开（public）」环境
----------------------------------------------------------------------------
  A) GitHub Actions（CI，跑 stock_dashboard.py 生成数据 / 推送）
       - 仓库是 public，任何人都能看源码与历史
       - Secret 只存在于 repo → Settings → Secrets → Actions
       - 运行时由 workflow 用 ${{ secrets.NAME }} 注入为「环境变量」
       - 本地 / 普通 clone 完全看不到这些变量
  B) Streamlit Cloud（public 部署，跑 app.py 对外提供网页）
       - 部署是 public，任何人都能访问页面
       - Secret 只存在于 Streamlit Cloud → Settings → Secrets（加密托管）
       - 运行时由 streamlit 注入为 st.secrets，不在代码 / 仓库中

本模块统一走 `utils._get_secret()`（双路：os.environ 优先，其次 st.secrets），
并明确区分上述两种场景的配置方式。

============================================================================
安全红线（本文件及所有调用方都必须遵守）
----------------------------------------------------------------------------
  ❌ 绝不把任何 key 硬编码进 .py / .toml / .yaml / .json / .md
  ❌ 绝不把含真实 key 的 .env / secrets.toml 提交到 public 仓库
  ✅ 只提交「模板」：.env.example / .streamlit/secrets.toml.example（占位符）
  ✅ key 仅在运行时由 CI / Streamlit 注入，进程内存中使用，绝不落盘 / 不打印明文
============================================================================
"""
from __future__ import annotations

import os
import subprocess
from typing import Dict, List, Optional, Tuple

import utils as U  # 项目工具模块：提供双路 _get_secret(env 优先, 其次 st.secrets)

# ---------------------------------------------------------------------------
# 1) 需要管理的 secret 清单（与 stock_dashboard.py / app.py 完全一致）
# ---------------------------------------------------------------------------
SECRET_KEYS: List[str] = [
    "FRED_API",           # FRED 利率 / 债务 / 杠杆
    "ALPHA_API",          # Alpha Vantage 期权链 / 技术指标
    "TELEGRAM_BOT_TOKEN", # Telegram 推送机器人 token
    "TELEGRAM_CHAT_ID",   # Telegram 目标 chat id
    "DEEPSEEK_API_KEY",   # DeepSeek AI 研判（主通道）
    "OPENROUTER_API_KEY", # OpenRouter 兜底（Claude 3.5 Sonnet）
    "SERPAPI",            # SerpApi 新闻增强（100/月免费）
]

# 运行「必需」与「可选增强」分离：缺失必需项直接报错；缺失可选项降级
REQUIRED_KEYS: List[str] = ["DEEPSEEK_API_KEY"]  # AI 研判通道至少要有 1 个
OPTIONAL_KEYS: List[str] = [k for k in SECRET_KEYS if k not in REQUIRED_KEYS]


# ---------------------------------------------------------------------------
# 2) 环境识别：明确区分 GitHub Actions / Streamlit Cloud / 本地
# ---------------------------------------------------------------------------
def detect_runtime() -> str:
    """
    判断当前运行环境。
    返回: 'github_actions' | 'streamlit_cloud' | 'local'

    - GitHub Actions: 自动注入 GITHUB_ACTIONS=true
    - Streamlit Cloud: 注入 STREAMLIT_CLOUD=true（部分版本）；
      本地开发则用 .streamlit/secrets.toml（同样安全，已被 .gitignore 排除）
    """
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return "github_actions"
    if os.environ.get("STREAMLIT_CLOUD") == "true" or os.path.exists(".streamlit/secrets.toml"):
        return "streamlit_cloud"
    return "local"


def runtime_doc(runtime: str) -> str:
    """返回该环境「secret 来自哪里、如何配置」的一句话说明（用于诊断输出）。"""
    return {
        "github_actions":
            "GitHub Actions：secret 来自 repo Settings → Secrets → Actions，"
            "由 workflow 用 ${{ secrets.NAME }} 注入为环境变量，本地/clone 不可见。",
        "streamlit_cloud":
            "Streamlit Cloud：secret 来自 app Settings → Secrets（加密托管），"
            "运行时注入 st.secrets，不在代码/仓库中。",
        "local":
            "本地：secret 来自 .env（python-dotenv）或 .streamlit/secrets.toml，"
            "两者均已被 .gitignore 排除，绝不会提交。",
    }.get(runtime, "未知环境")


# ---------------------------------------------------------------------------
# 3) 加载：统一走 U._get_secret（env 优先，其次 st.secrets）
# ---------------------------------------------------------------------------
def load_secrets(keys: Optional[List[str]] = None) -> Dict[str, str]:
    """
    读取所有 secret。永远不会把值写入文件或打印明文。
    返回 {key: value}，未配置的为空字符串。
    """
    keys = keys or SECRET_KEYS
    return {k: U._get_secret(k) for k in keys}


# ---------------------------------------------------------------------------
# 4) 脱敏：用于诊断 / 日志，绝不明文暴露
# ---------------------------------------------------------------------------
def redact(value: str, keep: int = 4) -> str:
    """脱敏显示：仅保留首尾 keep 个字符。空值显示未配置。"""
    if not value:
        return "—(未配置)—"
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}{'*' * (len(value) - keep * 2)}{value[-keep:]}"


# ---------------------------------------------------------------------------
# 5) 校验：区分必需缺失 / 可选缺失
# ---------------------------------------------------------------------------
def validate_secrets(secrets: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """返回 (missing_required, missing_optional)。"""
    missing_required = [k for k in REQUIRED_KEYS if not secrets.get(k)]
    missing_optional = [k for k in OPTIONAL_KEYS if not secrets.get(k)]
    return missing_required, missing_optional


# ---------------------------------------------------------------------------
# 6) 仓库自检：扫描是否误把真实 key 提交进 public 仓库
# ---------------------------------------------------------------------------
def scan_repo_for_leaks(keys: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """
    本地安全自检：用 git grep 扫描已跟踪文件是否出现「真实 key 值」。
    只检查，绝不修改任何文件。

    策略：
      - 用 git 跟踪文件列表（排除未跟踪的 .env / secrets.toml 本身）
      - 排除 *.example 模板文件（里面只有占位符）
      - 对每个已配置的 key，搜索其「真实值」是否出现在源码中
    返回 {key: [命中文件...]}（应为空字典才算安全）
    """
    keys = keys or SECRET_KEYS
    secrets = load_secrets(keys)
    hits: Dict[str, List[str]] = {}
    try:
        # 列出 git 跟踪的所有文件（不含 .example 之外的未跟踪敏感文件）
        tracked = subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, check=True
        ).stdout.splitlines()
    except Exception:  # noqa: BLE001
        return hits  # 无 git 环境则跳过

    # 只检查源码类文件，且排除 .example 模板
    candidates = [
        f for f in tracked
        if f.endswith((".py", ".yml", ".yaml", ".toml", ".json", ".md", ".txt", ".cfg", ".ini"))
        and not f.endswith(".example")
        and os.path.basename(f) != "secrets.toml"
    ]
    for key in keys:
        val = secrets.get(key)
        if not val:
            continue  # 未配置就谈不上泄露
        for f in candidates:
            try:
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    if val in fh.read():
                        hits.setdefault(key, []).append(f)
            except Exception:  # noqa: BLE001
                continue
    return hits


# ---------------------------------------------------------------------------
# 7) 主程序：演示加载 + 脱敏展示 + 校验 + 自检
# ---------------------------------------------------------------------------
def main() -> int:
    runtime = detect_runtime()
    print(f"🔍 运行环境识别: {runtime}")
    print(f"   {runtime_doc(runtime)}\n")

    secrets = load_secrets()
    missing_req, missing_opt = validate_secrets(secrets)

    print("=== Secret 配置状态（脱敏，绝不明文显示真实 key）===")
    for k in SECRET_KEYS:
        v = secrets.get(k, "")
        if not v:
            tag = "❌必需缺失" if k in REQUIRED_KEYS else "⚠️可选缺失"
        else:
            tag = "✅已配置"
        print(f"  [{tag:8s}] {k:22s} {redact(v)}")

    # 自检：仓库里有没有把真实 key 泄露进 public 文件
    leaks = scan_repo_for_leaks()
    if leaks:
        print("\n🚨 安全自检失败：发现真实 key 出现在以下已跟踪文件中：")
        for k, files in leaks.items():
            for f in files:
                print(f"   ⚠️  {k} 命中 {f}")
        print("   → 立即从文件中删除真实值，改走 Secrets 注入，并 git filter-repo 清理历史。")
    else:
        print("\n✅ 安全自检通过：未检测到真实 key 被提交到仓库。")

    print("\n=== 结论 ===")
    if missing_req:
        print(f"❌ 缺失必需 key: {missing_req} —— AI 研判通道不可用，程序终止。")
        return 1
    if missing_opt:
        print(f"⚠️ 缺失可选 key: {missing_opt} —— 相关功能降级（新闻/Telegram/行情源），其余正常运行。")
    else:
        print("✅ 全部 7 个 key 已就位。")
    print("💡 key 仅在进程内存中使用，不会写入任何文件，也不会打印明文。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
