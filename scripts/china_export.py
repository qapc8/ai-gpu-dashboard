"""China and US export-control intelligence for AI accelerators.

Two kinds of data, kept strictly apart:

  rules            LIVE. Pulled from the Federal Register API (free, no key):
                   BIS (Commerce) rules / notices on advanced computing ICs,
                   AI export controls, semiconductors and China-related Entity
                   List actions, plus presidential documents on semiconductors
                   and AI infrastructure. Last ~24 months, filtered with
                   explicit keyword rules (below) to keep out unrelated BIS
                   traffic (Russia sanctions, denial orders, OMB paperwork ...).

  china_prices     CURATED, citation-backed (scripts/curated/china_export.json).
  domestic_supply  Every entry carries the publication it came from, the url
  status           and the as_of date of the report. Nothing in this module
                   invents or interpolates a number; there is no free live
                   feed for grey-market H100/B200 prices in Shenzhen or Huawei
                   Ascend output, so those are press-reported figures only.

If the Federal Register cannot be reached, the previous `rules` list is kept
and flagged `rules_stale: true`; if the curated file cannot be read, the
previous curated lists are kept and flagged `curated_stale: true`.

`policy_events(section)` turns rules into items for the events calendar
(scripts/events.py imports it).

Stdlib only.
"""

import json
import os
import re
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone

USER_AGENT = "ai-gpu-dashboard/1.0 (market research)"
TIMEOUT = 30

FR_API = "https://www.federalregister.gov/api/v1/documents.json"
FR_DOCS_URL = "https://www.federalregister.gov/developers/documentation/api/v1"
BIS_SLUG = "industry-and-security-bureau"
LOOKBACK_DAYS = 730

CURATED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "curated", "china_export.json")

SRC_FR = "Federal Register"
SRC_CURATED = "China Export (curated)"
SOURCES = [SRC_FR, SRC_CURATED]

# --- Federal Register selection rules -------------------------------------
# BIS full-text searches. A hit is kept when title+abstract matches RELEVANT_RE
# and the title does not match NOISE_RE.
BIS_TERMS = ['"advanced computing"', '"integrated circuits"', '"artificial intelligence"',
             "semiconductor", "supercomputer"]
RELEVANT_RE = re.compile(
    r"advanced computing|integrated circuit|artificial intelligence|\bAI\b|semiconductor|"
    r"supercomput|model weights|data center|validated end.user", re.I)
NOISE_RE = re.compile(
    r"connected vehicle|unmanned aircraft|biotechnolog|laboratory equipment|polysilicon|"
    r"information collection|denying export privileges|temporarily denying|in the matter of",
    re.I)
# Entity List rules: only those naming China / PRC (or the affiliates rule that
# extends Entity List restrictions to 50%-owned subsidiaries).
ENTITY_TERM = '"entity list"'
CHINA_RE = re.compile(r"China|People's Republic|\bPRC\b|Hong Kong|Macau", re.I)
AFFILIATES_RE = re.compile(r"affiliates of certain listed entities", re.I)
# Presidential documents: whitelist of title patterns.
PRES_TERMS = ["semiconductors", '"artificial intelligence"']
PRES_TITLE_RE = re.compile(
    r"semiconductor|AI technology stack|leadership in artificial intelligence|"
    r"artificial intelligence infrastructure|national policy framework for artificial intelligence|"
    r"artificial intelligence innovation and security|super ?intelligence|data center", re.I)

FR_FIELDS = ["title", "type", "abstract", "document_number", "html_url",
             "publication_date", "effective_on", "agencies", "subtype"]

TYPE_MAP = {"Rule": "rule", "Proposed Rule": "proposed_rule", "Notice": "notice",
            "Presidential Document": "presidential"}

_MONTHS = ("January February March April May June July August September "
           "October November December").split()
_DEADLINE_RE = re.compile(
    r"(?:through|until|end(?:s|ing)? on|expires? on)\s+((?:%s) \d{1,2}, \d{4})" % "|".join(_MONTHS))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _get_json(url, params=None):
    if params:
        url = url + "?" + urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _fr_search(term, since, agency=None, doc_type=None):
    params = [("per_page", "200"), ("order", "newest"),
              ("conditions[term]", term),
              ("conditions[publication_date][gte]", since)]
    if agency:
        params.append(("conditions[agencies][]", agency))
    if doc_type:
        params.append(("conditions[type][]", doc_type))
    params += [("fields[]", f) for f in FR_FIELDS]
    payload = _get_json(FR_API, params)
    if "results" not in payload:
        if payload.get("count") == 0:
            return []
        raise RuntimeError(f"Federal Register: unexpected payload for {term!r}: "
                           f"{str(payload)[:200]}")
    return payload["results"]


def _agency_name(doc):
    if doc.get("type") == "Presidential Document":
        return "Executive Office of the President"
    names = [a.get("name") or a.get("raw_name") for a in doc.get("agencies") or []]
    for n in names:
        if n and "Industry and Security" in n:
            return "Bureau of Industry and Security"
    return names[-1] if names else ""


def _rule_from_doc(doc):
    abstract = re.sub(r"\s+", " ", doc.get("abstract") or "").strip()
    if len(abstract) > 700:
        abstract = abstract[:697].rsplit(" ", 1)[0] + "..."
    out = {
        "date": doc.get("publication_date"),
        "title": doc.get("title"),
        "agency": _agency_name(doc),
        "doc_number": doc.get("document_number"),
        "type": TYPE_MAP.get(doc.get("type"), (doc.get("type") or "").lower().replace(" ", "_")),
        "abstract": abstract,
        "url": doc.get("html_url"),
    }
    if doc.get("effective_on"):
        out["effective_on"] = doc["effective_on"]
    if doc.get("subtype"):
        out["subtype"] = doc["subtype"]
    return out


# --------------------------------------------------------------------------
# fetchers
# --------------------------------------------------------------------------

def fetch_federal_register(today=None):
    """Relevant BIS / presidential documents from the last ~24 months, newest first.
    Raises if every query fails or nothing relevant is found."""
    today = today or datetime.now(timezone.utc).date()
    since = (today - timedelta(days=LOOKBACK_DAYS)).isoformat()
    docs, errors, ok_queries = {}, [], 0

    def run(term, keep, **kw):
        nonlocal ok_queries
        try:
            res = _fr_search(term, since, **kw)
            ok_queries += 1
        except Exception as e:  # noqa: BLE001 -- one bad query must not kill the rest
            errors.append(f"{term}: {e}")
            return
        for d in res:
            if d.get("document_number") and keep(d):
                docs[d["document_number"]] = d

    def bis_keep(d):
        text = f"{d.get('title') or ''} {d.get('abstract') or ''}"
        return bool(RELEVANT_RE.search(text)) and not NOISE_RE.search(d.get("title") or "")

    def entity_keep(d):
        title = d.get("title") or ""
        if NOISE_RE.search(title):
            return False
        return bool(AFFILIATES_RE.search(title) or CHINA_RE.search(d.get("abstract") or ""))

    def pres_keep(d):
        return bool(PRES_TITLE_RE.search(d.get("title") or ""))

    for t in BIS_TERMS:
        run(t, bis_keep, agency=BIS_SLUG)
    run(ENTITY_TERM, entity_keep, agency=BIS_SLUG, doc_type="RULE")
    for t in PRES_TERMS:
        run(t, pres_keep, doc_type="PRESDOCU")

    if ok_queries == 0:
        raise RuntimeError("all Federal Register queries failed: " + "; ".join(errors)[:300])
    rules = [_rule_from_doc(d) for d in docs.values()]
    if not rules:
        raise RuntimeError("Federal Register returned no relevant documents")
    rules.sort(key=lambda r: (r["date"] or "", r["doc_number"] or ""), reverse=True)
    return rules, errors


_CURATED_REQ = {
    "china_prices": ("item", "price", "currency", "unit", "as_of", "source", "url"),
    "domestic_supply": ("item", "metric", "value", "as_of", "source", "url"),
    "status": ("topic", "status", "as_of", "source", "url"),
}


def load_curated(path=CURATED_PATH):
    """Curated lists; entries missing a citation field are dropped, never patched."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    out, dropped = {}, 0
    for key, required in _CURATED_REQ.items():
        rows = []
        for e in raw.get(key) or []:
            if all(e.get(f) not in (None, "") for f in required) and \
                    str(e.get("url", "")).startswith("http"):
                rows.append(e)
            else:
                dropped += 1
        rows.sort(key=lambda e: str(e.get("as_of") or ""), reverse=True)
        out[key] = rows
    if not any(out.values()):
        raise RuntimeError(f"no valid entries in {os.path.basename(path)}")
    return out, dropped


# --------------------------------------------------------------------------
# events helper
# --------------------------------------------------------------------------

POLICY_IMPACT_RULE = (
    "policy: high = BIS rule whose title names advanced computing, integrated circuits, "
    "AI / artificial intelligence, semiconductors or the Entity-List affiliates rule, or a "
    "presidential document on semiconductor imports / AI exports (and the effective dates and "
    "deadlines of those); medium = China-related Entity List / Unverified List / VEU rules and "
    "other presidential AI or data-center orders; low = public briefings, notices, everything else.")
_HIGH_POLICY_RE = re.compile(
    r"advanced computing|integrated circuit|artificial intelligence|\bAI\b|semiconductor|"
    r"affiliates of certain listed entities", re.I)
_HIGH_PRES_RE = re.compile(r"semiconductor|export|technology stack", re.I)
_LOW_RE = re.compile(r"public briefing|request for public comments", re.I)
_MED_POLICY_RE = re.compile(r"entity list|unverified list|validated end.user", re.I)
_CHIP_TICKERS_RE = re.compile(r"advanced computing|integrated circuit|artificial intelligence|"
                              r"\bAI\b|affiliates of certain listed", re.I)


_AGENCY_SHORT = {"Bureau of Industry and Security": "BIS",
                 "Executive Office of the President": "White House"}


def _policy_impact(rule):
    title = rule.get("title") or ""
    kind = rule.get("type")
    if _LOW_RE.search(title):
        return "low"
    if kind == "presidential":
        return "high" if _HIGH_PRES_RE.search(title) else "medium"
    if kind == "rule" and _HIGH_POLICY_RE.search(title):
        return "high"
    if kind in ("rule", "proposed_rule") and \
            (_MED_POLICY_RE.search(title) or _HIGH_POLICY_RE.search(title)):
        return "medium"
    return "low"


def _policy_tickers(rule):
    title = rule.get("title") or ""
    if re.search(r"semiconductor", title, re.I) and rule.get("type") == "presidential":
        return ["NVDA", "AMD", "TSM", "MU", "AVGO"]
    if _CHIP_TICKERS_RE.search(title) or re.search(r"H200|H20\b", rule.get("abstract") or ""):
        return ["NVDA", "AMD"]
    return []


def _parse_long_date(s):
    try:
        return datetime.strptime(s, "%B %d, %Y").date().isoformat()
    except ValueError:
        return None


def policy_events(section, today=None):
    """Federal Register rules -> events-schema items (type 'policy').

    Besides the publication itself, emits rule-derived future dates:
      * an 'effective' item when effective_on is after publication, and
      * a 'deadline' item for suspensions/extensions whose abstract states an
        end date ("... through January 10, 2027").
    """
    today = today or datetime.now(timezone.utc).date()
    items = []
    for r in (section or {}).get("rules") or []:
        if not r.get("date") or not r.get("doc_number"):
            continue
        impact, tickers = _policy_impact(r), _policy_tickers(r)
        base = {"type": "policy", "tickers": tickers, "impact": impact,
                "source": SRC_FR, "url": r.get("url"), "date_confirmed": True}
        items.append(dict(base, id=f"fr-{r['doc_number']}", date=r["date"],
                          title=f"{_AGENCY_SHORT.get(r.get('agency'), r.get('agency') or 'US')}: {r['title']}"))
        eff = r.get("effective_on")
        if eff and eff > r["date"] and eff >= (today - timedelta(days=548)).isoformat():
            items.append(dict(base, id=f"fr-{r['doc_number']}-effective", date=eff,
                              title=f"Takes effect: {r['title']}"))
        if re.search(r"suspension|extension|deadline|stay", r.get("title") or "", re.I):
            m = _DEADLINE_RE.search(r.get("abstract") or "")
            d = _parse_long_date(m.group(1)) if m else None
            if d and d > r["date"]:
                items.append(dict(base, id=f"fr-{r['doc_number']}-deadline", date=d,
                                  title=f"Deadline: {r['title']} (ends {m.group(1)})"))
    return items


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build_china_export(data, log_info=lambda m: None, log_ok=lambda a, b="": None,
                       log_fail=lambda a, b="": None):
    try:
        prev = data.get("china_export") or {}
        now = datetime.now(timezone.utc)
        section = {
            "as_of": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "sources": {},
            "rules": prev.get("rules") or [],
            "china_prices": prev.get("china_prices") or [],
            "domestic_supply": prev.get("domestic_supply") or [],
            "status": prev.get("status") or [],
        }

        # live: Federal Register
        try:
            rules, errs = fetch_federal_register(now.date())
            section["rules"] = rules
            note = f"{len(rules)} BIS/presidential documents since " \
                   f"{(now.date() - timedelta(days=LOOKBACK_DAYS)).isoformat()}"
            if errs:
                note += f"; {len(errs)} sub-queries failed"
                log_info(f"Federal Register partial failures: {'; '.join(errs)[:300]}")
            section["sources"][SRC_FR] = {"ok": True, "url": FR_DOCS_URL, "note": note}
            log_ok(SRC_FR, note)
        except Exception as e:  # noqa: BLE001
            section["rules_stale"] = True
            section["sources"][SRC_FR] = {"ok": False, "url": FR_DOCS_URL,
                                          "note": f"fetch failed, kept previous: {e}"[:300]}
            log_fail(SRC_FR, str(e)[:200])

        # curated
        rel = "scripts/curated/china_export.json"
        try:
            cur, dropped = load_curated()
            section.update(cur)
            note = (f"{len(cur['china_prices'])} prices, {len(cur['domestic_supply'])} supply, "
                    f"{len(cur['status'])} status entries; each cites source/url/as_of")
            if dropped:
                note += f"; {dropped} entries dropped for missing citation fields"
            section["sources"][SRC_CURATED] = {"ok": True, "url": rel, "note": note}
            log_ok(SRC_CURATED, note)
        except Exception as e:  # noqa: BLE001
            section["curated_stale"] = True
            section["sources"][SRC_CURATED] = {"ok": False, "url": rel,
                                               "note": f"load failed, kept previous: {e}"[:300]}
            log_fail(SRC_CURATED, str(e)[:200])

        data["china_export"] = section
    except Exception as e:  # noqa: BLE001 -- contract: never raise
        log_fail(SRC_FR, f"build_china_export crashed: {e}"[:200])
    return data


if __name__ == "__main__":
    repo = "/Users/quentinreyes/ai-gpu-dashboard-rm"
    out_path = ("/private/tmp/claude-501/-Users-quentinreyes/"
                "1c7063bc-b4c1-4db6-b232-f92320644371/scratchpad/china_export_sample.json")
    with open(os.path.join(repo, "data.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    d = build_china_export(d, log_info=lambda m: print("[INFO]", m),
                           log_ok=lambda s, m="": print("[ OK ]", s, m),
                           log_fail=lambda s, m="": print("[FAIL]", s, m))
    sec = d.get("china_export", {})
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"china_export": sec, "_policy_events": policy_events(sec)}, fh, indent=2)
    print("rules:", len(sec.get("rules", [])), "->", out_path)
    for r in sec.get("rules", []):
        print(" ", r["date"], r["type"], r["doc_number"], r["title"][:90])
