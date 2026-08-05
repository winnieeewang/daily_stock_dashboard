# -*- coding: utf-8 -*-
"""生成开发指令 HTML（用标准库，零依赖）。"""
import html
import re

SRC = "outputs/开发指令-量化看板v3.0.md"
DST = "outputs/开发指令-量化看板v3.0.html"

with open(SRC, encoding="utf-8") as f:
    md = f.read()

lines = md.split("\n")
out = []
in_code = False
in_table = False
table_rows = []


def close_table():
    global table_rows
    if not table_rows:
        return
    head = table_rows[0]
    body = table_rows[1:]
    h = "".join("<th>{}</th>".format(html.escape(c.strip())) for c in head.split("|")[1:-1])
    b = "".join(
        "<tr>{}</tr>".format(
            "".join("<td>{}</td>".format(html.escape(c.strip())) for c in r.split("|")[1:-1])
        )
        for r in body
    )
    out.append("<table><thead><tr>{}</tr></thead><tbody>{}</tbody></table>".format(h, b))
    table_rows = []


for ln in lines:
    if ln.startswith("```"):
        close_table()
        if in_code:
            out.append("</pre>")
            in_code = False
        else:
            out.append("<pre>")
            in_code = True
        continue
    if in_code:
        out.append(html.escape(ln))
        continue
    if ln.strip().startswith("|") and ln.strip().endswith("|"):
        if not in_table:
            in_table = True
        table_rows.append(ln)
        continue
    if in_table:
        if ln.strip() and not ln.strip().startswith("|"):
            close_table()
            in_table = False
        else:
            continue
    if not ln.strip():
        out.append("")
        continue
    if ln.startswith("# "):
        out.append("<h1>{}</h1>".format(html.escape(ln[2:])))
    elif ln.startswith("## "):
        out.append("<h2>{}</h2>".format(html.escape(ln[3:])))
    elif ln.startswith("### "):
        out.append("<h3>{}</h3>".format(html.escape(ln[4:])))
    elif ln.startswith("#### "):
        out.append("<h4>{}</h4>".format(html.escape(ln[5:])))
    elif re.match(r"^\s*[-*] ", ln):
        out.append("<li>{}</li>".format(html.escape(ln[2:])))
    elif ln.startswith("> "):
        out.append("<blockquote>{}</blockquote>".format(html.escape(ln[2:])))
    elif re.match(r"^\s*\d+\. ", ln):
        num = ln.split(".")[0].strip()
        rest = ln.split(".", 1)[1].strip()
        out.append("<li><b>{}</b> {}</li>".format(html.escape(num), html.escape(rest)))
    else:
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", html.escape(ln))
        t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
        out.append("<p>{}</p>".format(t))
close_table()

css = """
<style>
  body { font-family: -apple-system,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif; max-width: 980px; margin: 0 auto; padding: 34px 26px; color: #1f2937; line-height: 1.75; background: #fff; }
  h1 { color: #1d4ed8; border-bottom: 3px solid #1d4ed8; padding-bottom: 12px; font-size: 26px; }
  h2 { color: #1e40af; margin-top: 38px; border-left: 4px solid #3b82f6; padding-left: 12px; font-size: 20px; }
  h3 { color: #111827; margin-top: 24px; font-size: 16px; }
  h4 { color: #374151; margin-top: 18px; font-size: 14px; }
  table { border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 13px; }
  th, td { border: 1px solid #e5e7eb; padding: 8px 11px; text-align: left; }
  th { background: #eff6ff; font-weight: 700; }
  tr:nth-child(even) td { background: #fafbfc; }
  code { background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-size: 12.5px; font-family: Consolas, monospace; }
  pre { background: #0f172a; color: #e2e8f0; padding: 16px; border-radius: 8px; overflow-x: auto; font-size: 13px; }
  pre code { background: none; color: inherit; padding: 0; }
  blockquote { border-left: 4px solid #f59e0b; background: #fffbeb; padding: 10px 16px; margin: 14px 0; border-radius: 0 6px 6px 0; }
  li { margin: 3px 0; }
  p { margin: 8px 0; }
</style>
"""
final = (
    '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1">'
    "<title>量化看板 v3.0 开发指令</title>{}</head><body>{}"
    '<p style="margin-top:44px;color:#9ca3af;font-size:12px;border-top:1px solid #e5e7eb;'
    'padding-top:12px;text-align:center;">Investment Copilot · 开发指令文档 · 2026-08-05</p>'
    "</body></html>"
).format(css, "".join(out))

with open(DST, "w", encoding="utf-8") as f:
    f.write(final)
print("HTML OK:", len(final), "bytes ->", DST)
