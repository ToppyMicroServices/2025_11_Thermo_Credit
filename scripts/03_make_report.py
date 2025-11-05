import pandas as pd, plotly.express as px, plotly.io as pio
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
pio.renderers.default = "json"

df = pd.read_csv("site/indicators.csv", parse_dates=["date"])

fig1 = px.line(df, x="date", y=["S_M","T_L"], title="Money Entropy S_M and Liquidity Temperature T_L")
fig2 = px.line(df, x="date", y="loop_area", title="Regulatory Loop 'Area' (Streaming Estimator)")
fig3 = px.line(df, x="date", y="X_C", title="Credit Exergy X_C = U - T0*S_M")

html = "\n".join([f.to_html(full_html=False, include_plotlyjs='cdn') for f in [fig1, fig2, fig3]])
with open("site/report.html","w",encoding="utf-8") as f:
    f.write(f"<!doctype html><meta charset='utf-8'><title>Thermo-Credit Monitor</title>{html}")
print("Wrote site/report.html")
