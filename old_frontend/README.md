# Streamlit UI — Serving A Nation

A polished, three-tab UI shell. **All data is mocked.** Same function signatures will be wired to Delta tables and the agent layer on day 5 of the build.

## Run locally

```bash
cd ~/Desktop/Hack_nation-V2
python -m venv .venv
source .venv/bin/activate
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

Browser opens at `http://localhost:8501`.

## Tabs

1. **Patient Finder** — natural-language chat, sample queries, mock agent trace, ranked facilities on a map, expandable claim-level citations with validator status.
2. **Medical Desert Map** — bubble map of high-acuity coverage per district (population-weighted), specialty selector, bottom-10 districts table, 95% bootstrap CI plot.
3. **Trust & Data-Quality Audit** — KPI tiles, filterable audit table, per-facility drilldown with trust-component breakdown and validator findings.

## What's mocked vs. real

| Element | Now | Day 5 swap |
|---|---|---|
| Facility list | 25 hand-crafted seeds | `spark.read.table("workspace.hack_nation.gold_facilities").toPandas()` |
| Agent reasoning trace | Hardcoded 7-step list | MLflow trace span dump |
| Recommendation ranking | Hand-rolled scorer | Real planner → vector → graph → ranker pipeline |
| District desert metrics | Random with realistic ranges | Bootstrap CIs from `gold_district_coverage` table |
| Trust score | Computed live from mock components | Computed live from real `gold_trust_components` table (same formula) |

## Demo seeds intentionally embedded in the mock data

- **Aurangabad/UP geographic bug** — facility 4 is in Aurangabad but listed in Uttar Pradesh. Tab 3 surfaces it via the `location_coherence` component.
- **Advanced-surgery-without-equipment contradiction** — facilities at indices 3, 10, 17 claim advanced surgery with empty equipment arrays. Validator flags them.
- **Trauma-without-defibrillator** — applied wherever trauma is in the specialty pool but defibrillator missing.

## Style notes

- Light theme, deep-blue brand palette (`#0b3d91`).
- Inter typeface (Google Fonts) for UI, JetBrains Mono for citations.
- Trust badges use a fixed semantic palette: green ≥0.70, yellow 0.45–0.70, red <0.45.
- No emojis anywhere — medical-professional aesthetic.
- Maps use `mapbox_style="open-street-map"` so no Mapbox token is required.

## Wiring to real data later

Every "mock" function returns the same shape it'll return in production. The swap on day 5 is mechanical:

```python
# before
@st.cache_data(show_spinner=False)
def load_facilities() -> list[Facility]:
    return _mock_facilities()

# after — same return type
@st.cache_data(show_spinner=False)
def load_facilities() -> list[Facility]:
    sdf = spark.read.table("workspace.hack_nation.gold_facilities")
    return [Facility(**row.asDict(recursive=True)) for row in sdf.collect()]
```
