import urllib.request, urllib.parse, json, ssl, datetime, math, time

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def _get(url, headers=None, timeout=20):
    req = urllib.request.Request(url, headers=headers or {"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return json.loads(r.read().decode("utf-8","ignore"))

def yahoo_chart(symbol, range_="1d", interval="1d"):
    for host in ("query1","query2"):
        try:
            url=f"https://{host}.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}?range={range_}&interval={interval}"
            d=_get(url)
            res=d.get("chart",{}).get("result")
            if res: return res[0]
        except Exception:
            continue
    return None

def quote(symbol):
    for attempt in range(3):
        r=yahoo_chart(symbol,"1d","1d")
        if r:
            m=r["meta"]
            last=m.get("regularMarketPrice"); prev=m.get("chartPreviousClose") or m.get("previousClose")
            closes=r.get("indicators",{}).get("quote",[{}])[0].get("close",[])
            clean=[c for c in closes if c is not None]
            if (last is None) and clean: last=clean[-1]
            if (prev is None) and len(clean)>=2: prev=clean[-2]
            chg=m.get("regularMarketChangePercent")
            if chg is None and last and prev: chg=(last-prev)/prev*100.0
            return {"last":last,"prevClose":prev,"chgPct":round(chg,2) if chg is not None else None}
        time.sleep(0.5)
    return None

def gtimg_hk(symbol):
    """腾讯财经兜底：港股 Yahoo 区域限制时取实时价+涨跌幅。symbol 形如 00293.HK"""
    code=symbol.split(".")[0]
    try:
        url=f"https://qt.gtimg.cn/q=hk{code}"
        req=urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0","Referer":"https://gu.qq.com/"})
        raw=urllib.request.urlopen(req,timeout=15,context=ctx).read()
        txt=raw.decode("gbk","ignore")
        for line in txt.split(";"):
            if "hk"+code in line and "~" in line:
                parts=line.split("~")
                price=float(parts[3]); raw=float(parts[32])
                # 不加硬封顶：2倍杠杆ETF单日真实波动可达 ±60%~；仅剔除明显解析错误的极端值
                chgPct=raw if abs(raw)<=200.0 else 0.0
                return {"last":price,"prevClose":price/(1+chgPct/100) if chgPct else price,"chgPct":round(chgPct,2)}
    except Exception:
        pass
    return None

def em_hk(symbol):
    """东方财富实时报价（港股覆盖最全，含杠杆ETF）。symbol 形如 7709.HK。价格×1000、涨跌幅×100。"""
    code = symbol.replace(".HK", "").zfill(5)
    try:
        url = f"https://push2.eastmoney.com/api/qt/stock/get?secid=116.{code}&fields=f43,f60,f170,f57,f58"
        d = _get(url)
        dd = (d or {}).get("data") or {}
        if not dd:
            return None
        last = dd.get("f43")
        if last is None:
            return None
        last = last / 1000.0
        prev = dd.get("f60")
        prev = prev / 1000.0 if prev else last
        pct = (dd.get("f170") or 0) / 100.0
        return {"last": round(last, 3), "prevClose": round(prev, 3), "chgPct": round(pct, 2)}
    except Exception:
        return None

def history(symbol, range_="3mo", interval="1d"):
    """返回 (closes, volumes)，同一时间序列，保证时间维度一致"""
    for attempt in range(3):
        r=yahoo_chart(symbol,range_,interval)
        if r:
            ts=r["timestamp"]; q=r["indicators"]["quote"][0]
            closes=[c for t,c in zip(ts,q.get("close",[])) if c is not None]
            vols=[v for t,v in zip(ts,q.get("volume",[])) if v is not None]
            if closes: return closes, vols
        time.sleep(0.5)
    return [], []

def stats(closes, volumes=None):
    if len(closes)<2: return {}
    last=closes[-1]
    ma20=sum(closes[-20:])/min(20,len(closes))
    ma60=sum(closes[-60:])/min(60,len(closes)) if len(closes)>=60 else None
    rets=[(closes[i]/closes[i-1]-1) for i in range(1,len(closes))]
    vol20=sum(abs(x) for x in rets[-20:])/min(20,len(rets))*100  # ATR% 近似
    ret20=(last/ma20-1)*100 if ma20 else 0
    ret60=(last/ma60-1)*100 if ma60 else None
    vol_ratio=None; vol_state="—"
    if volumes and len(volumes)>=6:
        recent=volumes[-1]
        win=volumes[-21:-1] if len(volumes)>=21 else volumes[:-1]
        avg=sum(win)/len(win) if win else 0
        if avg>0:
            vol_ratio=round(recent/avg,2)
            vol_state="放量" if vol_ratio>=1.2 else ("缩量" if vol_ratio<=0.8 else "平量")
    return {"ma20":round(ma20,3),"ma60":round(ma60,3) if ma60 else None,
            "atrPct":round(vol20,2),"ret20":round(ret20,2),
            "ret60":round(ret60,2) if ret60 is not None else None,
            "last_close":round(closes[-1],4),
            "vol_ratio":vol_ratio,"vol_state":vol_state,
            "n":len(closes)}

NAMES = {
    "MU":"美光","AAOI":"应用光电","GOOGL":"谷歌","MSFT":"微软","AMZN":"亚马逊","MRVL":"迈威尔",
    "LITE":"Lumentum","SNDK":"闪迪","NVDA":"英伟达","ORCL":"甲骨文","SPCX":"标普500ETF","SKHY":"SK海力士","TSLA":"特斯拉",
    "0700.HK":"腾讯控股","0883.HK":"中国海洋石油","3750.HK":"宁德时代","07709.HK":"南方两倍做多海力士","00981.HK":"中芯国际",
    "688809.SS":"强一股份","300408.SZ":"三环集团","300679.SZ":"电连技术","000426.SZ":"兴业银锡","002624.SZ":"完美世界",
    "601872.SS":"招商轮船","601975.SS":"招商南油","002258.SZ":"利尔化学","001331.SZ":"胜通能源","600150.SS":"中国船舶",
    "00293.HK":"国泰航空","03690.HK":"美团-W","01138.HK":"中远海能","03968.HK":"招商银行",
    "EUV":"Corgi Lithography","RKLB":"Rocket Lab","GEV":"GE Vernova","FUTU":"富途","UNH":"联合健康",
    "NVO":"诺和诺德","NFLX":"Netflix","JNJ":"强生","INTU":"Intuit",
}
# 展示代码 -> Yahoo 抓取代码（HKEX 5 位港股去 1 个前导零：00981→0981、07709→7709…）
YHOO = {"03690.HK":"3690.HK","07709.HK":"7709.HK","00981.HK":"0981.HK",
        "00293.HK":"0293.HK","01138.HK":"1138.HK","03968.HK":"3968.HK"}
WATCHLIST = [(s, YHOO.get(s, s), NAMES[s]) for s in NAMES.keys()]

def market_of(s):
    if s.endswith(".HK"): return "港股"
    if s.endswith((".SS",".SZ")): return "A股"
    return "美股"

# ---------------------------------------------------------------------------
# 衍生字段（与 PWA computeRec 同源算法，服务端预计算，保证时间一致）
# ---------------------------------------------------------------------------
def clamp(x,a,b): return max(a,min(b,x))

def fmt_stop(last, atrPct):
    if last is None or atrPct is None: return None
    sp=max(atrPct*2.2,7)/100.0
    return round(last*(1-sp),2)

def buy_levels(last, ma20, ma60, atrPct):
    if last is None: return None, None
    a = atrPct if atrPct else 3.0
    ideal = round(last*(1 - 0.5*a/100.0), 2)
    add = round(last*(1 - a/100.0), 2)
    return ideal, add

def factors(e):
    bull=[]; bear=[]
    last=e.get("last"); ma20=e.get("hist_ma20"); ma60=e.get("hist_ma60")
    chg=e.get("chgPct") or 0; ret20=e.get("hist_ret20") or 0
    atr=e.get("hist_atrPct"); pe=e.get("pe")
    if last and ma20:
        if last>ma20: bull.append(f"现价站上20日线({ma20})，短期趋势向上")
        else: bear.append(f"现价跌破20日线({ma20})，短期承压")
    if ma20 and ma60:
        if ma20>ma60: bull.append("20日线 > 60日线，多头排列")
        else: bear.append("20日线 < 60日线，空头排列")
    if chg>0: bull.append(f"当日上涨 {chg:.2f}%")
    else: bear.append(f"当日下跌 {abs(chg):.2f}%")
    if ret20>0: bull.append(f"近20日收益 +{ret20:.1f}%")
    else: bear.append(f"近20日收益 {ret20:.1f}%")
    if atr is not None:
        if atr<4: bull.append(f"ATR {atr:.1f}% 波动温和")
        elif atr>6: bear.append(f"ATR {atr:.1f}% 波动放大，需收紧风控")
    if pe is not None:
        if 0<pe<30: bull.append(f"PE {pe:.1f} 估值合理")
        elif pe>=50: bear.append(f"PE {pe:.1f} 估值偏高")
    return bull[:4], bear[:4]

def score(e):
    last=e.get("last"); chg=e.get("chgPct") or 0
    ma20=e.get("hist_ma20"); ma60=e.get("hist_ma60"); atr=e.get("hist_atrPct"); ret20=e.get("hist_ret20")
    pe=e.get("pe")
    tech=50
    if ma20 is not None:
        tech=50+(ret20 or 0)*1.5+(10 if last>ma20 else 0)+(10 if (ma20 and ma60 and ma20>ma60) else 0)-(8 if last<ma20 else 0)
        tech=clamp(tech,0,100)
    val=50
    if pe is not None:
        if pe>0 and pe<15: val=85
        elif pe<25: val=70
        elif pe<35: val=58
        elif pe<50: val=48
        else: val=38
        if pe<=0: val=45
    mom=clamp(50+chg*3,0,100)
    comp=clamp(0.5*tech+0.25*val+0.25*mom,0,100)
    bias="看多" if comp>=60 else ("看空" if comp<=40 else "震荡")
    return dict(tech=round(tech),val=round(val),mom=round(mom),comp=round(comp),bias=bias)

def enrich(e):
    a=e.get("hist_atrPct")
    e["stop_loss"]=fmt_stop(e.get("last"),a)
    ib,ab=buy_levels(e.get("last"),e.get("hist_ma20"),e.get("hist_ma60"),a)
    e["ideal_buy"]=ib; e["add_buy"]=ab
    b,br=factors(e); e["bullish"]=b; e["bearish"]=br
    e.update(score(e))
    return e

def market_sentiment(vix):
    if vix is None: return {"score":50,"label":"中性","note":"VIX 暂缺","vix":None}
    if vix<15: s,l,n=72,"乐观/贪婪","VIX<15 避险情绪低，风险资产占优"
    elif vix<20: s,l,n=60,"中性偏多","VIX 15-20 市场平稳"
    elif vix<25: s,l,n=42,"谨慎","VIX 20-25 波动上升，降低杠杆"
    else: s,l,n=28,"恐慌","VIX>25 避险主导，控制仓位"
    return {"score":s,"label":l,"note":n,"vix":round(vix,2)}

snap={"generated_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"indices":{},"panic":{},"watchlist":[],"risk":{},"vix_history":{},"sentiment":{}}

INDICES={"000001.SS":{"name":"上证指数","yf":"000001.SS"},"^HSI":{"name":"恒生指数","yf":"^HSI"},
    "^GSPC":{"name":"标普500","yf":"^GSPC"},"^IXIC":{"name":"纳斯达克","yf":"^IXIC"},"^DJI":{"name":"道琼斯","yf":"^DJI"}}
print("== indices ==")
for k,v in INDICES.items():
    q=quote(v["yf"]); rec={"name":v["name"],**(q or {})}
    try:
        h,_=history(v["yf"],"1mo","1d")
        if len(h)>=2: rec["chg1m"]=round((h[-1]-h[0])/h[0]*100,2)
    except Exception: pass
    snap["indices"][k]=rec; print(" ",v["name"],rec.get("chgPct"),rec.get("chg1m"))
print("== panic ==")
for s in ("^VIX","^VXN"):
    q=quote(s); snap["panic"][s.strip("^")]=q or {}; print(" ",s,q)
vixv = (snap["panic"].get("VIX",{}) or {}).get("last")
snap["sentiment"]=market_sentiment(vixv)
print("== sentiment ==", snap["sentiment"])
print("== watchlist (",len(WATCHLIST),") ==")
for disp, yf, nm in WATCHLIST:
    e={"symbol":disp,"yf":yf,"name":nm,"market":market_of(disp)}
    # 港股优先用东方财富实时报价（覆盖最全、含杠杆ETF），其次腾讯，最后 Yahoo
    if disp.endswith(".HK"):
        q = em_hk(disp) or gtimg_hk(disp) or quote(yf)
    else:
        q = quote(yf)
    if q and q.get("last") is not None:
        e.update({"last":q["last"],"chgPct":q.get("chgPct") or 0.0})
    else:
        e.update({"last":None,"chgPct":0.0})
    e["pe"]=None  # PE 由 PWA 客户端经代理实时补抓（服务端 Yahoo 需鉴权）
    try:
        cl,vols=history(yf,"3mo","1d"); st=stats(cl,vols)
        e.update({f"hist_{k}":v for k,v in st.items()})
    except Exception: pass
    enrich(e)
    snap["watchlist"].append(e)
    print(f"  {disp:10s} {e['name'][:12]:12s} last={e.get('last')} chg={e.get('chgPct')} vol={e.get('hist_vol_state')} comp={e.get('comp')} bias={e.get('bias')} stop={e.get('stop_loss')}")
    time.sleep(0.35)
print("== risk extras ==")
for s,nm in (("^TNX","10Y"),("DX-Y.NYB","DXY")):
    q=quote(s); snap["risk"][nm]=q or {}; print(" ",nm,q)
print("== vix/vxn 10y weekly ==")
vh={}
for s in ("^VIX","^VXN"):
    h={}
    try:
        rr=yahoo_chart(s,"10y","1wk")
        if rr:
            for t,c in zip(rr["timestamp"],rr["indicators"]["quote"][0]["close"]):
                if c is None: continue
                h[datetime.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d")]=round(float(c),2)
    except Exception: pass
    if h: vh[s.strip("^")]=h
snap["vix_history"]=vh

with open("pwa/snapshot.json","w",encoding="utf-8") as f:
    json.dump(snap,f,ensure_ascii=False,indent=1)
print("WROTE pwa/snapshot.json  watchlist=",len(snap["watchlist"]))
