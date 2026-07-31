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
            # 兜底：meta 价格为 None 时取时间序列末值
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
                price=float(parts[3]); chgPct=float(parts[32])
                return {"last":price,"prevClose":price/(1+chgPct/100) if chgPct else price,"chgPct":round(chgPct,2)}
    except Exception:
        pass
    return None

def history(symbol, range_="3mo", interval="1d"):
    for attempt in range(3):
        r=yahoo_chart(symbol,range_,interval)
        if r:
            ts=r["timestamp"]; closes=r["indicators"]["quote"][0]["close"]
            out=[c for t,c in zip(ts,closes) if c is not None]
            if out: return out
        time.sleep(0.5)
    return []

def stats(closes):
    if len(closes)<2: return {}
    last=closes[-1]
    ma20=sum(closes[-20:])/min(20,len(closes))
    ma60=sum(closes[-60:])/min(60,len(closes)) if len(closes)>=60 else None
    rets=[(closes[i]/closes[i-1]-1) for i in range(1,len(closes))]
    vol20=sum(abs(x) for x in rets[-20:])/min(20,len(rets))*100  # ATR% 近似
    ret20=(last/ma20-1)*100 if ma20 else 0
    ret60=(last/ma60-1)*100 if ma60 else None
    return {"ma20":round(ma20,3),"ma60":round(ma60,3) if ma60 else None,
            "atrPct":round(vol20,2),"ret20":round(ret20,2),
            "ret60":round(ret60,2) if ret60 is not None else None,
            "n":len(closes)}

NAMES = {
    "MU":"美光","AAOI":"应用光电","GOOGL":"谷歌","MSFT":"微软","AMZN":"亚马逊","MRVL":"迈威尔",
    "LITE":"Lumentum","SNDK":"闪迪","NVDA":"英伟达","ORCL":"甲骨文","SPCX":"标普500ETF","SKHY":"高收益债ETF","TSLA":"特斯拉",
    "0700.HK":"腾讯控股","0883.HK":"中国海洋石油","3750.HK":"锂业ETF","07709.HK":"南方两倍做多海力士","00981.HK":"中芯国际",
    "688809.SS":"豪威股份","300408.SZ":"三环集团","300679.SZ":"电连技术","000426.SZ":"兴业银锡","002624.SZ":"完美世界",
    "601872.SS":"招商轮船","601975.SS":"招商轮船","002258.SZ":"利尔化学","001331.SZ":"胜通能源","600150.SS":"中国船舶",
    "00293.HK":"国泰航空","03690.HK":"美团-W","01138.HK":"中远海能","03968.HK":"招商银行",
    "EUV":"Corgi Lithography","RKLB":"Rocket Lab","GEV":"GE Vernova","FUTU":"富途","UNH":"联合健康",
    "NVO":"诺和诺德","NFLX":"Netflix","JNJ":"强生","INTU":"Intuit",
}
# 展示代码 -> Yahoo 抓取代码（个别港股 Yahoo 不带前导零）
YHOO = {"03690.HK":"3690.HK"}
WATCHLIST = [(s, YHOO.get(s, s), NAMES[s]) for s in NAMES.keys()]

def market_of(s):
    if s.endswith(".HK"): return "港股"
    if s.endswith((".SS",".SZ")): return "A股"
    return "美股"

snap={"generated_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"indices":{},"panic":{},"watchlist":[],"risk":{},"vix_history":{}}

INDICES={"000001.SS":{"name":"上证指数","yf":"000001.SS"},"^HSI":{"name":"恒生指数","yf":"^HSI"},
    "^GSPC":{"name":"标普500","yf":"^GSPC"},"^IXIC":{"name":"纳斯达克","yf":"^IXIC"},"^DJI":{"name":"道琼斯","yf":"^DJI"}}
print("== indices ==")
for k,v in INDICES.items():
    q=quote(v["yf"]); rec={"name":v["name"],**(q or {})}
    try:
        h=history(v["yf"],"1mo","1d")
        if len(h)>=2: rec["chg1m"]=round((h[-1]-h[0])/h[0]*100,2)
    except Exception: pass
    snap["indices"][k]=rec; print(" ",v["name"],rec.get("chgPct"),rec.get("chg1m"))
print("== panic ==")
for s in ("^VIX","^VXN"):
    q=quote(s); snap["panic"][s.strip("^")]=q or {}; print(" ",s,q)
print("== watchlist (41) ==")
for disp, yf, nm in WATCHLIST:
    e={"symbol":disp,"yf":yf,"name":nm,"market":market_of(disp)}
    q=quote(yf)
    if not (q and q.get("last") is not None) and disp.endswith(".HK"):
        q=gtimg_hk(disp)
    if q and q.get("last") is not None:
        e.update({"last":q["last"],"chgPct":q.get("chgPct") or 0.0})
    else:
        e.update({"last":None,"chgPct":0.0})
    e["pe"]=None  # PE 由 PWA 客户端经代理实时补抓（服务端 Yahoo 需鉴权）
    try:
        cl=history(yf,"3mo","1d"); st=stats(cl)
        e.update({f"hist_{k}":v for k,v in st.items()})
    except Exception: pass
    snap["watchlist"].append(e)
    print(f"  {disp:10s} {e['name'][:14]:14s} last={e.get('last')} atr={e.get('hist_atrPct')} ret20={e.get('hist_ret20')}")
    time.sleep(0.35)
print("== risk extras ==")
for s,nm in (("^TNX","10Y"),("DX-Y.NYB","DXY")):
    q=quote(s); snap["risk"][nm]=q or {}; print(" ",nm,q)
print("== vix/vxn 10y weekly ==")
vh={}
for s in ("^VIX","^VXN"):
    h={}
    try:
        r=history(s,"10y","1wk")
        # rebuild dated dict
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
