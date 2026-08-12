"""
PIB scaled crawl (stdlib only) — bounded, polite, resumable.

Enumerates N days of National (reg=48) releases via the Allrel date postback,
expands each to its ReleaseLang siblings, extracts structured native-Unicode
records, groups into parallel release sets. Appends incrementally so a restart
resumes; rate-limited; hard fetch cap. Provenance: PIB reproduction policy.
"""
import re, json, time, gzip, sys, urllib.request, urllib.parse
from pathlib import Path
from datetime import date, timedelta
from collections import Counter

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
ALLREL = "https://www.pib.gov.in/Allrel.aspx?reg=48&lang=1"
REL = "https://www.pib.gov.in/PressReleasePage.aspx?PRID=%s"
DELAY = 1.2
DAYS = 7
MAX_FETCHES = 6000          # hard safety bound
OUT = Path.home() / "quanfire-ai-data/embedding/gov-indic/corpus/gov-corpus.jsonl"

_LANG = {"english":"en","hindi":"hi","हिन्दी":"hi","हिंदी":"hi","urdu":"ur","اردو":"ur",
 "marathi":"mr","मराठी":"mr","telugu":"te","తెలుగు":"te","tamil":"ta","தமிழ்":"ta",
 "bengali":"bn","বাংলা":"bn","gujarati":"gu","ગુજરાતી":"gu","kannada":"kn","ಕನ್ನಡ":"kn",
 "malayalam":"ml","മലയാളം":"ml","punjabi":"pa","ਪੰਜਾਬੀ":"pa","odia":"or","oriya":"or",
 "ଓଡ଼ିଆ":"or","assamese":"as","অসমীয়া":"as","manipuri":"mni","nepali":"ne"}

def guess_lang(t):
    c = Counter()
    for ch in t:
        o = ord(ch)
        if 0x900<=o<=0x97F: c["hi"]+=1
        elif 0x980<=o<=0x9FF: c["bn"]+=1
        elif 0xB80<=o<=0xBFF: c["ta"]+=1
        elif 0xC00<=o<=0xC7F: c["te"]+=1
        elif 0xA80<=o<=0xAFF: c["gu"]+=1
        elif 0xC80<=o<=0xCFF: c["kn"]+=1
        elif 0xD00<=o<=0xD7F: c["ml"]+=1
        elif 0xA00<=o<=0xA7F: c["pa"]+=1
        elif 0xB00<=o<=0xB7F: c["or"]+=1
        elif 0x600<=o<=0x6FF: c["ur"]+=1
        elif 65<=o<=122: c["en"]+=1
    return c.most_common(1)[0][0] if c else "??"

_fetches = 0
def get(url, data=None):
    global _fetches
    _fetches += 1
    headers = {"User-Agent": UA, "Accept-Language": "en"}
    body = None
    if data is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=45) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return raw.decode("utf-8", "replace")

def hidden(html):
    out = {}
    for m in re.finditer(r'<input[^>]*type="hidden"[^>]*>', html):
        nm = re.search(r'name="([^"]+)"', m.group(0)); vl = re.search(r'value="([^"]*)"', m.group(0))
        if nm: out[nm.group(1)] = vl.group(1) if vl else ""
    return out

def strip_tags(s):
    s = re.sub(r"<script.*?</script>|<style.*?</style>", " ", s, flags=re.S|re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("&nbsp;"," ").replace("&amp;","&").replace("&#39;","'").replace("&quot;",'"')
    return re.sub(r"[ \t]+"," ", re.sub(r"\s*\n\s*","\n", s)).strip()

def by_id(html, eid, closing):
    m = re.search(rf'id="{eid}"[^>]*>(.*?)</{closing}>', html, re.S)
    return strip_tags(m.group(1)) if m else ""

def parse_siblings(html):
    m = re.search(r'class="ReleaseLang">(.*?)</div>', html, re.S); out={}
    if m:
        for prid,label in re.findall(r"PRID=(\d+)'[^>]*>\s*([^<]+?)\s*</a>", m.group(1)):
            out[prid] = _LANG.get(label.strip().lower(), label.strip().lower())
    return out

def parse_body(html):
    start = html.find('id="PrDateTime"')
    if start < 0: start = html.find("innner-page-main-about-us-content-right-part")
    region = html[start:] if start>=0 else html
    cut = region.find('class="ReleaseLang"')
    if cut>0: region = region[:cut]
    region = re.sub(r'id="PrDateTime".*?</div>'," ", region, flags=re.S)
    return strip_tags(region)

def parse_release(html):
    return {"title": by_id(html,"Titleh2","h2"), "body": parse_body(html),
            "ministry": by_id(html,"MinistryName","div"), "subtitle": by_id(html,"ltrSubtitle","span"),
            "date": by_id(html,"PrDateTime","div").replace("प्रविष्टि तिथि:","").strip()}

def anchors_for(d):
    base = get(ALLREL); time.sleep(DELAY)
    form = hidden(base)
    form.update({"__EVENTTARGET":"ctl00$ContentPlaceHolder1$ddlYear","__EVENTARGUMENT":"",
        "ctl00$Bar1$ddlregion":"48","ctl00$Bar1$ddlLang":"1","ctl00$ContentPlaceHolder1$ddlMinistry":"0",
        "ctl00$ContentPlaceHolder1$ddlday":str(d.day),"ctl00$ContentPlaceHolder1$ddlMonth":str(d.month),
        "ctl00$ContentPlaceHolder1$ddlYear":str(d.year)})
    html = get(ALLREL, form); time.sleep(DELAY)
    return list(dict.fromkeys(re.findall(r"PRID=(\d+)", html)))

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen = set(); ngroups = 0
    if OUT.exists():
        for line in OUT.open(encoding="utf-8"):
            if line.strip():
                g = json.loads(line); ngroups += 1
                seen |= {v["prid"] for v in g["langs"].values()}
        print(f"[resume] {ngroups} groups, {len(seen)} PRIDs already done", flush=True)
    out = OUT.open("a", encoding="utf-8")
    today = date.today()
    for i in range(DAYS):
        if _fetches >= MAX_FETCHES: print("[cap] fetch cap reached", flush=True); break
        d = today - timedelta(days=i)
        try:
            anchors = anchors_for(d)
        except Exception as e:
            print(f"[{d}] anchor-list ERR {type(e).__name__}", flush=True); continue
        new = [a for a in anchors if a not in seen]
        print(f"[{d}] {len(anchors)} anchors, {len(new)} new (fetches={_fetches})", flush=True)
        for ap in new:
            if _fetches >= MAX_FETCHES: break
            if ap in seen: continue
            try:
                html = get(REL % ap); time.sleep(DELAY)
                sibs = parse_siblings(html)
                pages = {ap: html}
                for sp in sibs:
                    if sp in seen or sp in pages: continue
                    pages[sp] = get(REL % sp); time.sleep(DELAY)
                label = dict(sibs)
                for p, h in pages.items():
                    for k, v in parse_siblings(h).items(): label.setdefault(k, v)
                rec = {"group": min(pages, key=int), "langs": {}}
                for p, h in pages.items():
                    rel = parse_release(h); rel["prid"] = p
                    lang = label.get(p) or guess_lang(rel["title"] + rel["body"])
                    rec["langs"][lang] = rel
                out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
                seen |= set(pages); ngroups += 1
            except Exception as e:
                print(f"  [{ap}] ERR {type(e).__name__} {getattr(e,'code','')}", flush=True)
    out.close()
    # summary
    groups = [json.loads(l) for l in OUT.open(encoding="utf-8") if l.strip()]
    langc = Counter(l for g in groups for l in g["langs"])
    mult = sum(1 for g in groups if len(g["langs"]) > 1)
    print(f"\n[DONE] groups={len(groups)} multilingual={mult} fetches={_fetches}")
    print(f"  langs={dict(langc)}")
    print(f"  wrote {OUT}")

if __name__ == "__main__":
    main()
