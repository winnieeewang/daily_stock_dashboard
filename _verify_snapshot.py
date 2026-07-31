import json

s = json.load(open("pwa/snapshot.json", encoding="utf-8"))
wl = s["watchlist"]
ga = s.get("generated_at")
print(f"快照生成时间(统一时间维度): {ga}")
print(f"股票数: {len(wl)}  市场情绪: {s.get('sentiment')}")
print("=" * 78)

fails, warns = [], []

def check(cond, msg, sym, kind="WARN"):
    if not cond:
        (fails if kind == "FAIL" else warns).append(f"[{sym}] {msg}")

for e in wl:
    sym = e["symbol"]; nm = e["name"]
    last = e.get("last")
    # 1) 价格有效
    check(last and last > 0, f"last 无效: {last}", sym, "FAIL")
    # 2) 时间一致性: last(quote) ≈ hist_last(history 同序列末值)
    hl = e.get("hist_last")
    if hl and last:
        dev = abs(last - hl) / last
        check(dev < 0.005, f"last({last}) 与 history 末值({hl}) 偏差 {dev*100:.2f}% >0.5%", sym, "FAIL")
    # 3) ret20 与 MA20 自洽
    ma20 = e.get("hist_ma20"); ret20 = e.get("hist_ret20")
    if ma20 and last and ret20 is not None:
        exp = (last / ma20 - 1) * 100
        check(abs(exp - ret20) < 1.2, f"ret20({ret20}) 与 MA20 推导({exp:.1f}) 不符", sym, "WARN")
    # 4) ATR 合理
    atr = e.get("hist_atrPct")
    if atr is not None:
        check(0.2 < atr < 15, f"ATR% 异常: {atr}", sym, "WARN")
    # 5) 成交量状态合理
    vr = e.get("hist_vol_ratio"); vs = e.get("hist_vol_state")
    if vr is not None:
        check(0.1 < vr < 10, f"vol_ratio 异常: {vr}", sym, "WARN")
        check(vs in ("放量", "缩量", "平量"), f"vol_state 异常: {vs}", sym, "FAIL")
    # 6) 价格阶梯: last >= ideal >= add >= stop
    ib, ab, sl = e.get("ideal_buy"), e.get("add_buy"), e.get("stop_loss")
    if all(x is not None for x in (last, ib, ab, sl)):
        check(last >= ib - 1e-6 and ib >= ab - 1e-6 and ab >= sl - 1e-6,
              f"价格阶梯乱序 last={last} ideal={ib} add={ab} stop={sl}", sym, "FAIL")
    # 7) 止损低于现价
    if sl and last:
        check(sl < last, f"止损({sl}) 不低于现价({last})", sym, "FAIL")
    # 8) 综合分自洽
    comp, tech, val, mom = e.get("comp"), e.get("tech"), e.get("val"), e.get("mom")
    if None not in (comp, tech, val, mom):
        expc = 0.5 * tech + 0.25 * val + 0.25 * mom
        check(abs(expc - comp) < 2.5, f"comp({comp}) 与 0.5*tech+0.25*val+0.25*mom({expc:.0f}) 不符", sym, "WARN")
    # 9) chgPct 合理
    chg = e.get("chgPct") or 0
    check(-30 <= chg <= 30, f"chgPct 超范围: {chg}", sym, "WARN")
    # 10) 多空因子存在
    if last and ma20:
        check(bool(e.get("bullish") or e.get("bearish")), f"无多空因子", sym, "WARN")

print(f"FAIL: {len(fails)}   WARN: {len(warns)}")
for f in fails:
    print("  ❌", f)
for w in warns:
    print("  ⚠️ ", w)
if not fails and not warns:
    print("  ✅ 全部通过：价格有效、时间维度一致、指标自洽、价格阶梯与评分逻辑正确。")

# 概览表
print("=" * 78)
print(f"{'代码':10s}{'名称':14s}{'现价':>10s}{'涨跌%':>8s}{'量':>5s}{'综合':>5s}{'偏向':>5s}{'止损':>10s}")
for e in sorted(wl, key=lambda x: -(x.get('comp') or 0)):
    print(f"{e['symbol']:10s}{e['name'][:12]:14s}{str(e.get('last')):>10s}"
          f"{str(e.get('chgPct')):>8s}{str(e.get('hist_vol_state') or '-'):>5s}"
          f"{str(e.get('comp')):>5s}{str(e.get('bias')):>6s}{str(e.get('stop_loss')):>10s}")
