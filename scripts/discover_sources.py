"""Discovery probe (run in CI, which has open internet): lists candidate
official RBI documents and checks the HMDA Data Browser API, so fetch URLs
are confirmed against the live sites rather than guessed."""
import re
import sys
import urllib.request
from html import unescape

UA = {"User-Agent": "Mozilla/5.0 (compatible; lending-policy-rag/1.0; +https://github.com/yash9518/lending-policy-rag)"}
KEYWORDS = re.compile(r"know your customer|kyc|fair practice|digital lending|penal charge|key fact|"
                      r"floating interest|reset of floating|release of movable|integrated ombudsman|"
                      r"housing finance|foreclosure|pre-payment|prepayment|income recognition|"
                      r"scale based regulation|responsible lending|credit information", re.I)


def get(url, limit=4_000_000):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, r.headers.get("content-type", ""), r.read(limit)


def links(html, base):
    for href, text in re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.S | re.I):
        text = unescape(re.sub(r"<[^>]+>|\s+", " ", text)).strip()
        if href.startswith("/"):
            href = base + href
        elif not href.startswith("http"):
            href = base + "/Scripts/" + href
        yield text, href


def title(html):
    m = re.search(r"<title>(.*?)</title>", html, re.S | re.I)
    return unescape(m.group(1)).strip() if m else ""


def main():
    base = "https://www.rbi.org.in"
    for page in ["/Scripts/BS_ViewMasterDirections.aspx", "/Scripts/BS_ViewMasterCirculardetails.aspx",
                 "/Scripts/NotificationUser.aspx"]:
        try:
            status, ctype, body = get(base + page)
            html = body.decode("utf-8", "ignore")
            print(f"PAGE {status} {page} bytes={len(body)} title={title(html)!r}")
            seen = set()
            for text, href in links(html, base):
                if KEYWORDS.search(text) and href not in seen:
                    seen.add(href)
                    print(f"  LINK {text[:160]!r} -> {href}")
        except Exception as exc:
            print(f"PAGE-ERROR {page}: {type(exc).__name__}: {exc}")

    # Candidate ids recalled from memory: verify by title before any use.
    candidates = [
        "/Scripts/BS_ViewMasDirections.aspx?id=11566",
        "/Scripts/BS_ViewMasDirections.aspx?id=12550",
        "/Scripts/NotificationUser.aspx?Id=12527",
        "/Scripts/NotificationUser.aspx?Id=12528",
        "/Scripts/NotificationUser.aspx?Id=12535",
        "/Scripts/NotificationUser.aspx?Id=12663",
        "/Scripts/NotificationUser.aspx?Id=12382",
        "/Scripts/NotificationUser.aspx?Id=12175",
    ]
    for c in candidates:
        try:
            status, ctype, body = get(base + c)
            html = body.decode("utf-8", "ignore")
            head = re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", html)))
            m = re.search(r"(RBI/20\d\d-\d\d/\d+)", head)
            print(f"CAND {status} {c} title={title(html)[:100]!r} ref={m.group(1) if m else None} "
                  f"snippet={head[head.find('RBI/'):head.find('RBI/') + 300]!r}")
        except Exception as exc:
            print(f"CAND-ERROR {c}: {type(exc).__name__}: {exc}")

    for url in [
        "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states=DE&years=2023&actions_taken=1,3",
        "https://ffiec.cfpb.gov/v2/data-browser-api/view/aggregations?states=DE&years=2023&actions_taken=1,3",
    ]:
        try:
            status, ctype, body = get(url, limit=3000)
            print(f"HMDA {status} {ctype} {url}\n{body.decode('utf-8', 'ignore')[:1500]}")
        except Exception as exc:
            print(f"HMDA-ERROR {url}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    sys.exit(main())
