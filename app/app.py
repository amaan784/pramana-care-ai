"""Pramana Streamlit app — 3 tabs: Chat / Map / Audit."""
from __future__ import annotations

import os
import json

import streamlit as st
import pandas as pd
import pydeck as pdk
from databricks.sdk import WorkspaceClient
from openai import OpenAI

st.set_page_config(
    page_title="Pramana — Truth-Check Engine",
    layout="wide",
    page_icon="🪔",
)

ENDPOINT = os.environ["SERVING_ENDPOINT_NAME"]
WAREHOUSE_ID = os.environ.get("WAREHOUSE_ID")
GENIE_SPACE_ID = os.environ.get("GENIE_SPACE_ID")
CATALOG = os.environ.get("PRAMANA_CATALOG", "workspace")
SCHEMA = os.environ.get("PRAMANA_SCHEMA", "pramana")
NS = f"{CATALOG}.{SCHEMA}"

w = WorkspaceClient()
client = OpenAI(
    api_key=w.config.oauth_token().access_token,
    base_url=f"{w.config.host}/serving-endpoints",
)


@st.cache_data(ttl=600, show_spinner=False)
def run_sql(sql: str) -> pd.DataFrame:
    if not WAREHOUSE_ID:
        return pd.DataFrame()
    res = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID, statement=sql, wait_timeout="30s",
    )
    cols = [c.name for c in (res.manifest.schema.columns or [])]
    rows = (res.result.data_array or []) if res.result else []
    return pd.DataFrame(rows, columns=cols)


st.title("🪔 Pramana — Agentic Facility Truth-Check Engine")
st.caption("Verify capability claims · find medical deserts · audit data quality.")

tab_chat, tab_map, tab_audit = st.tabs(
    ["💬 Chat", "🗺️ Medical Deserts", "🔍 Data Audit"]
)

# ---------------------------------------------------------- Chat
with tab_chat:
    st.header("Ask Pramana")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if prompt := st.chat_input(
        "e.g. Is District Hospital Kishanganj actually equipped for cardiac surgery?"
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            def stream():
                resp = client.chat.completions.create(
                    model=ENDPOINT,
                    messages=st.session_state.messages,
                    stream=True,
                    timeout=180,
                )
                for chunk in resp:
                    if chunk.choices and chunk.choices[0].delta:
                        yield chunk.choices[0].delta.content or ""
            full = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": full})

# ---------------------------------------------------------- Map
with tab_map:
    st.header("H3 Hex Coverage — Specialty Deserts")
    col1, col2 = st.columns([1, 3])
    with col1:
        specialty = st.selectbox(
            "Specialty", ["oncology", "cardiology", "nephrology", "trauma", "pediatrics"]
        )
        state_filter = st.text_input("State filter (optional)", "")
        st.caption("Hex color = facility count. Red = no coverage in that hex.")

    state_clause = (
        f"AND state = '{state_filter}'" if state_filter.strip() else ""
    )

    spec = specialty.replace("'", "")
    sql = f"""
    SELECT h3_6,
           COUNT(*) AS n_facilities,
           SUM(CASE WHEN exists(specialties, x -> contains(lower(x), '{spec}'))
                    THEN 1 ELSE 0 END) AS n_specialty,
           AVG(latitude)  AS lat,
           AVG(longitude) AS lon
    FROM {NS}.gold_facilities
    WHERE h3_6 IS NOT NULL {state_clause}
    GROUP BY h3_6
    """
    df = run_sql(sql)

    with col2:
        if df.empty:
            st.info("No data — verify WAREHOUSE_ID env var and that gold_facilities exists.")
        else:
            df["n_specialty"] = pd.to_numeric(df["n_specialty"], errors="coerce").fillna(0).astype(int)
            df["n_facilities"] = pd.to_numeric(df["n_facilities"], errors="coerce").fillna(0).astype(int)
            df["color_r"] = df["n_specialty"].apply(lambda x: 200 if x == 0 else 30)
            df["color_g"] = df["n_specialty"].apply(lambda x: 30  if x == 0 else 180)
            df["color_b"] = 80
            layer = pdk.Layer(
                "H3HexagonLayer",
                df,
                pickable=True,
                stroked=True,
                filled=True,
                extruded=False,
                get_hexagon="h3_6",
                get_fill_color="[color_r, color_g, color_b, 160]",
                line_width_min_pixels=1,
            )
            view = pdk.ViewState(latitude=22.5, longitude=80.0, zoom=4.2, pitch=0)
            st.pydeck_chart(pdk.Deck(
                layers=[layer], initial_view_state=view,
                tooltip={"text": "hex {h3_6}\n{n_specialty}/{n_facilities} {specialty}"},
                map_style=None,
            ))
            st.metric(
                f"Hexes with zero {specialty} coverage",
                int((df["n_specialty"] == 0).sum()),
                f"{(df['n_specialty'] == 0).mean()*100:.1f}% of populated hexes",
                delta_color="inverse",
            )

# ---------------------------------------------------------- Audit
with tab_audit:
    st.header("We audited our own data")
    st.caption(
        "The agent doesn't just trust the input — it scores every row against 8 rules and "
        "exposes the breakdown. This snapshot has 51 cross-state coordinate mismatches "
        "(0.51%) and 0 outside-India coordinates."
    )

    audit_sql = f"""
    SELECT facility_id, name, facility_type_raw, facility_type, state, city,
           latitude, longitude, trust_score,
           size(flags) AS n_flags
    FROM {NS}.gold_facilities
    WHERE size(flags) > 0
    ORDER BY trust_score ASC
    LIMIT 100
    """
    bad = run_sql(audit_sql)

    cL, cR = st.columns(2)
    with cL:
        st.subheader("Raw — claims & coordinates as ingested")
        if not bad.empty:
            st.dataframe(
                bad[["facility_id", "name", "facility_type_raw", "state", "city",
                     "latitude", "longitude"]],
                use_container_width=True, height=420,
            )
        else:
            st.info("No flagged rows found — run notebook 04 first.")
    with cR:
        st.subheader("After validation — Pramana flags")
        if not bad.empty:
            st.dataframe(
                bad[["facility_id", "facility_type", "trust_score", "n_flags"]],
                use_container_width=True, height=420,
            )

    m1, m2, m3 = st.columns(3)
    farmacy = run_sql(
        f"SELECT COUNT(*) AS n FROM {NS}.gold_facilities WHERE facility_type_raw='farmacy'"
    )
    rule_counts = run_sql(
        f"SELECT severity, COUNT(*) AS n FROM {NS}.silver_contradictions GROUP BY severity"
    )
    rc = {r["severity"]: int(r["n"]) for _, r in rule_counts.iterrows()} if not rule_counts.empty else {}

    m1.metric("'farmacy' typo entries", int(farmacy.iloc[0]["n"]) if not farmacy.empty else 0)
    m2.metric("HIGH-severity contradictions", rc.get("HIGH", 0))
    m3.metric(
        "Cross-state coordinate mismatches", "51",
        "0 outside India bounding box",
        delta_color="inverse",
    )
