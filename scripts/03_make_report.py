import os
import html as html_lib
import pandas as pd
import plotly.express as px
import plotly.io as pio
import json
from datetime import datetime
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
from urllib.parse import urlparse
import shutil

# Use JSON renderer for headless builds; we'll embed HTML fragments manually
pio.renderers.default = "json"

# Load data
df = pd.read_csv("site/indicators.csv", parse_dates=["date"]).sort_values("date")
os.makedirs("site", exist_ok=True)

# --- Build figures (interactive) ---
fig1 = px.line(
    df, x="date", y=["S_M", "T_L"],
    title="Money Entropy (S_M) and Liquidity Temperature (T_L)",
    labels={"value": "Index value", "date": "Date", "variable": "Series"}
)
fig2 = px.line(
    df, x="date", y="loop_area",
    title="Policy/Regulatory Loop 'Dissipation' (Streaming Estimator)",
    labels={"loop_area": "Loop area (arb. units)", "date": "Date"}
)
fig3 = px.line(
    df, x="date", y="X_C",
    title="Credit Exergy Ceiling: X_C = U − T0 · S_M",
    labels={"X_C": "X_C (arb. units)", "date": "Date"}
)

# --- Save static PNG fallbacks (for crawlers / no-JS). Requires kaleido; continue if missing.
png_fallback_ok = False
try:
    fig1.write_image("site/fig1.png", scale=2, width=1280, height=720)
    fig2.write_image("site/fig2.png", scale=2, width=1280, height=720)
    fig3.write_image("site/fig3.png", scale=2, width=1280, height=720)
    png_fallback_ok = True
except Exception as e:
    print("PNG export skipped:", e)

# --- Text summary (for SEO / no-JS) ---
last = df.iloc[-1]
latest_date = last["date"].strftime("%Y-%m-%d")
def fmt(v):
    try:
        return f"{float(v):.4g}"
    except Exception:
        return "n/a"

summary_items = [
    f"Latest date: {latest_date}",
    f"S_M (money entropy): {fmt(last.get('S_M'))}",
    f"T_L (liquidity temperature): {fmt(last.get('T_L'))}",
    f"Loop area (streaming): {fmt(last.get('loop_area'))}",
    f"X_C (credit exergy ceiling): {fmt(last.get('X_C'))}",
]
summary_html = "<ul>" + "".join(f"<li>{html_lib.escape(s)}</li>" for s in summary_items) + "</ul>"

# Recent mini table (up to last 6 rows)
tail = df[["date", "S_M", "T_L", "loop_area", "X_C"]].tail(6).copy()
tail["date"] = tail["date"].dt.strftime("%Y-%m-%d")
table_html = tail.to_html(index=False, border=0, classes="mini", escape=True)

# --- Compose final HTML with meta + noscript fallbacks ---
head = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Thermo-Credit Monitor</title>
  <meta name="description" content="Monthly thermo-credit indicators: liquidity temperature (T_L), monetary entropy (S_M), policy loop dissipation, and credit exergy ceiling (X_C).">
  <link rel="canonical" href="https://toppymicros.com/2025_11_Thermo_Credit/report.html">
  <meta property="og:title" content="Thermo-Credit Monitor">
  <meta property="og:description" content="Latest T_L, S_M, loop dissipation, and X_C with textual fallback and images for no-JS/SEO.">
  <meta property="og:type" content="article">
  <meta property="og:url" content="https://toppymicros.com/2025_11_Thermo_Credit/report.html">
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;line-height:1.5;margin:1.25rem}
    h1{font-size:1.6rem;margin:0 0 .5rem}
    h2{font-size:1.1rem;margin:1.25rem 0 .5rem}
    .wrap{max-width:1100px;margin:0 auto}
    .note{color:#333;margin:.5rem 0 1rem}
    figure{margin:1rem 0}
    figcaption{font-size:.9rem;color:#555}
    table.mini{border-collapse:collapse;margin:.5rem 0}
    table.mini td,table.mini th{padding:.25rem .5rem;border-bottom:1px solid #ddd;text-align:right}
    table.mini th:first-child,table.mini td:first-child{text-align:left}
    .grid{display:grid;grid-template-columns:1fr;gap:1rem}
    @media(min-width:900px){.grid{grid-template-columns:1fr 1fr}}
    .visually-hidden{position:absolute;left:-10000px;top:auto;width:1px;height:1px;overflow:hidden}
  </style>
</head>
<body>
<div class="wrap">
  <h1>Thermo-Credit Monitor</h1>
  <p class="note">Below charts are interactive (Plotly). A textual summary and PNG fallbacks are included for crawlers and no‑JS browsers.</p>
"""

# Interactive charts (HTML fragments) with accessible wrappers
figs_html = []
for idx, (fig, title, alt) in enumerate([
    (fig1, "S_M & T_L", "Time series of monetary entropy S_M and liquidity temperature T_L."),
    (fig2, "Policy Loop Dissipation", "Streaming estimate of policy/regulatory loop area."),
    (fig3, "Credit Exergy Ceiling", "Exergy ceiling X_C = U − T0 · S_M."),
], start=1):
    figs_html.append(
        f"<figure aria-label='{html_lib.escape(alt)}'>"
        f"{fig.to_html(full_html=False, include_plotlyjs='cdn')}"
        f"<figcaption>{html_lib.escape(title)}</figcaption></figure>"
    )
figs_html = "\n".join(figs_html)

# noscript fallback with static PNGs
if png_fallback_ok:
    noscript = """
    <noscript>
      <h2>Static summary images (JavaScript disabled)</h2>
      <figure><img src="fig1.png" alt="S_M and T_L over time" width="100%"/><figcaption>S_M &amp; T_L</figcaption></figure>
      <figure><img src="fig2.png" alt="Policy/regulatory loop area (streaming)" width="100%"/><figcaption>Policy Loop Dissipation</figcaption></figure>
      <figure><img src="fig3.png" alt="Credit exergy ceiling X_C" width="100%"/><figcaption>Credit Exergy Ceiling</figcaption></figure>
    </noscript>
    """
else:
    noscript = """
    <noscript>
      <p>JavaScript is disabled. Static images were not generated on this run, but the textual summary and mini table below provide the latest values.</p>
    </noscript>
    """

footer = f"""
  <h2>Latest snapshot</h2>
  {summary_html}
  <h2>Recent values</h2>
  {table_html}
</div>
</body>
</html>
"""

final_html = head + figs_html + noscript + footer
with open("site/report.html", "w", encoding="utf-8") as f:
    f.write(final_html)

print("Wrote site/report.html (with interactive charts and fallbacks)")

# --- Monthly Archive / RSS / Sitemap ---

DEFAULT_BASE = "https://toppymicros.com/2025_11_Thermo_Credit"
def _validated_base_url(raw: str) -> str:
    """
    Return a safe base URL limited to our domains and https only.
    Falls back to DEFAULT_BASE if invalid.
    """
    try:
        u = urlparse((raw or "").strip())
        if u.scheme != "https":
            return DEFAULT_BASE
        host = (u.netloc or "").lower()
        # allowlist: primary custom domain and GitHub Pages
        allow = {"toppymicros.com", "toppymicroservices.github.io"}
        if host not in allow:
            return DEFAULT_BASE
        # normalize path to our prefix
        path = u.path.rstrip("/")
        if not path.endswith("/2025_11_Thermo_Credit"):
            path = "/2025_11_Thermo_Credit"
        return f"https://{host}{path}"
    except Exception:
        return DEFAULT_BASE

base_url = _validated_base_url(os.getenv("TMS_BASE_URL", DEFAULT_BASE))
month_key = last["date"].strftime("%Y-%m")
month_dir = os.path.join("site", month_key)
os.makedirs(month_dir, exist_ok=True)

# Duplicate PNGs into the month folder (so past months keep their own images)
if png_fallback_ok:
    try:
        shutil.copyfile("site/fig1.png", os.path.join(month_dir, "fig1.png"))
        shutil.copyfile("site/fig2.png", os.path.join(month_dir, "fig2.png"))
        shutil.copyfile("site/fig3.png", os.path.join(month_dir, "fig3.png"))
    except Exception as _e:
        pass

# Build a month-specific page (reusing the same charts/summary)
month_head = head.replace(
    "<title>Thermo-Credit Monitor</title>",
    f"<title>Thermo-Credit Monitor – {month_key}</title>"
).replace(
    'href="https://toppymicros.com/2025_11_Thermo_Credit/report.html"',
    f'href="https://toppymicros.com/2025_11_Thermo_Credit/{month_key}/"'
).replace(
    'content="https://toppymicros.com/2025_11_Thermo_Credit/report.html"',
    f'content="https://toppymicros.com/2025_11_Thermo_Credit/{month_key}/"'
)
month_noscript = noscript.replace('src="fig', 'src="../fig') if png_fallback_ok else noscript
month_html = month_head + figs_html + month_noscript + footer
with open(os.path.join(month_dir, "index.html"), "w", encoding="utf-8") as f:
    f.write(month_html)

# Archive JSON: load previous if available (local first, then remote), append this month, deduplicate
archive_path = "site/archive.json"
archive = []
# Try local
if os.path.exists(archive_path):
    try:
        with open(archive_path, "r", encoding="utf-8") as fp:
            archive = json.load(fp)
    except Exception:
        archive = []
# If local empty, try remote (best-effort)
if not archive:
    try:
        with urlopen(f"{base_url}/archive.json", timeout=10) as r:
            ctype = (r.headers.get("Content-Type") or "").lower()
            raw = r.read(524288)  # cap to 512 KiB
            if "application/json" in ctype or raw.strip().startswith(b"["):
                archive = json.loads(raw.decode("utf-8", errors="strict"))
    except Exception:
        pass

# Normalize to list of dicts
if not isinstance(archive, list):
    archive = []

entry = {
    "month": month_key,
    "url": f"{base_url}/{month_key}/",
    "lastmod": last["date"].strftime("%Y-%m-%d"),
    "title": f"Thermo-Credit Monitor {month_key}",
    "summary": summary_items
}
# Upsert by month
by_month = {e.get("month"): e for e in archive if isinstance(e, dict)}
by_month[month_key] = entry
archive = sorted(by_month.values(), key=lambda e: e.get("month", ""), reverse=True)

with open(archive_path, "w", encoding="utf-8") as fp:
    json.dump(archive, fp, ensure_ascii=False, indent=2)

# RSS (Atom-like RSS 2.0)
def rss_escape(s: str) -> str:
    return (s.replace("&","&amp;")
             .replace("<","&lt;")
             .replace(">","&gt;"))

rss_items = []
for e in archive[:24]:  # up to last two years
    pub = datetime.strptime(e["month"] + "-01", "%Y-%m-%d")
    pub_rfc822 = pub.strftime("%a, %d %b %Y 00:00:00 +0000")
    rss_items.append(
        f"<item><title>{rss_escape(e['title'])}</title>"
        f"<link>{rss_escape(e['url'])}</link>"
        f"<guid>{rss_escape(e['url'])}</guid>"
        f"<pubDate>{pub_rfc822}</pubDate>"
        f"<description>{rss_escape(' – '.join(map(str, e.get('summary', [])[:2])))}</description>"
        f"</item>"
    )

rss_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Thermo-Credit Monitor</title>
    <link>{base_url}/</link>
    <description>Monthly thermo-credit indicators: T_L, S_M, loop dissipation, X_C.</description>
    <language>en</language>
    {''.join(rss_items)}
  </channel>
</rss>"""
with open("site/feed.xml", "w", encoding="utf-8") as fp:
    fp.write(rss_xml)

# Sitemap
urls = [
    f"{base_url}/",
    f"{base_url}/report.html",
    f"{base_url}/feed.xml",
] + [f"{base_url}/{e['month']}/" for e in archive]
today = datetime.utcnow().strftime("%Y-%m-%d")
urlset = "".join(
    f"<url><loc>{rss_escape(u)}</loc><lastmod>{today}</lastmod></url>"
    for u in urls
)
sitemap = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  {urlset}
</urlset>"""
with open("site/sitemap.xml", "w", encoding="utf-8") as fp:
    fp.write(sitemap)

# robots.txt (declare sitemap)
with open("site/robots.txt", "w", encoding="utf-8") as fp:
    fp.write(f"User-agent: *\nAllow: /\nSitemap: {base_url}/sitemap.xml\n")

print("Wrote monthly archive, feed.xml, sitemap.xml, and robots.txt")
