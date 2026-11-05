"""Discovery probe (run in CI, which has open internet): lists candidate
official RBI documents so fetch URLs are confirmed against the live site."""
import re
import urllib.request
from html import unescape

UA = {"User-Agent": "Mozilla/5.0 (compatible; lending-policy-rag/1.0; +https://github.com/yash9518/lending-policy-rag)"}
BASE = "https://www.rbi.org.in"
KEYWORDS = re.compile(r"know your customer|fair practice|digital lending|penal charge|key fact|floating|"
                      r"release of movable|ombudsman|housing finance|foreclosure|pre-payment|prepayment|"
                      r"income recognition|asset classification|scale based|responsible lending|"
                      r"credit information|interest rate on advances|loans and advances", re.I)


def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "ignore")


def text_of(html):
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", html))).strip()


def subject(html):
    t = text_of(html)
    ref = re.search(r"RBI/\S+", t)
    m = re.search(r"(?:Madam\s*/\s*Sir|Sir\s*/\s*Madam|Dear Sir|Madam/ Sir|Sir/ Madam)[,.]?\s*(.{0,140})", t)
    date = re.search(r"((?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4})", t)
    return (ref.group(0) if ref else None), (date.group(1) if date else None), (m.group(1) if m else t[:140])


for page in ["/Scripts/BS_ViewMasterDirections.aspx", "/Scripts/BS_ViewMasterCirculardetails.aspx"]:
    html = get(BASE + page)
    for href, inner in re.findall(r"<a\b[^>]*?href\s*=\s*['\"]([^'\"]+)['\"][^>]*>(.*?)</a>", html, re.S | re.I):
        label = text_of(inner)
        if KEYWORDS.search(label):
            print(f"LIST {page.split('/')[-1]} | {label[:170]} | {href}")

for i in list(range(12526, 12534)) + list(range(12176, 12200)) + list(range(12820, 12840)):
    try:
        ref, date, subj = subject(get(f"{BASE}/Scripts/NotificationUser.aspx?Id={i}"))
        if KEYWORDS.search(subj or ""):
            print(f"NOTIF Id={i} | {ref} | {date} | {subj}")
    except Exception as exc:
        print(f"NOTIF-ERROR Id={i}: {exc}")
