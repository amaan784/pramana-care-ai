"""PramanaCare.ai — Streamlit app.

Polished four-tab UI ported from old_frontend/streamlit_app.py and wired to the
real Databricks backend (model serving endpoint + SQL warehouse + gold tables).
The backend (src/, notebooks/, databricks.yml, app.yaml) is unchanged; only this
file and app/requirements.txt were touched.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from databricks.sdk import WorkspaceClient
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PramanaCare — Verified healthcare for 1.4 billion",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# Backend wiring (unchanged from previous app/app.py)
# ──────────────────────────────────────────────────────────────────────────────

ENDPOINT = os.environ.get("SERVING_ENDPOINT_NAME", "")
WAREHOUSE_ID = os.environ.get("WAREHOUSE_ID")
GENIE_SPACE_ID = os.environ.get("GENIE_SPACE_ID")
CATALOG = os.environ.get("PRAMANA_CATALOG", "workspace")
SCHEMA = os.environ.get("PRAMANA_SCHEMA", "pramana")
NS = f"{CATALOG}.{SCHEMA}"

@st.cache_resource(show_spinner=False)
def _workspace_client() -> WorkspaceClient:
    return WorkspaceClient()


def _serving_api_token() -> str:
    """Return the best available token for OpenAI-compatible serving calls.

    Tries OAuth (Databricks Apps), then SDK config token, then DATABRICKS_TOKEN.
    """
    w = _workspace_client()
    try:
        return w.config.oauth_token().access_token
    except Exception:
        pass
    token = getattr(w.config, "token", None)
    if token:
        return token
    token = os.environ.get("DATABRICKS_TOKEN")
    if token:
        return token
    raise RuntimeError("No Databricks token available for serving endpoint calls.")


@st.cache_resource(show_spinner=False)
def _openai_client() -> OpenAI | None:
    if not ENDPOINT:
        return None
    w = _workspace_client()
    return OpenAI(
        api_key=_serving_api_token(),
        base_url=f"{w.config.host}/serving-endpoints",
    )


@st.cache_data(ttl=600, show_spinner=False)
def run_sql(sql: str) -> pd.DataFrame:
    if not WAREHOUSE_ID:
        st.warning("SQL warehouse is not configured. Attach `pramana-sql-warehouse` to the app.")
        return pd.DataFrame()
    w = _workspace_client()
    try:
        res = w.statement_execution.execute_statement(
            warehouse_id=WAREHOUSE_ID,
            statement=sql,
            wait_timeout="30s",
        )
        for _ in range(24):
            state = str(getattr(getattr(res, "status", None), "state", "") or "").upper()
            if "SUCCEEDED" in state or getattr(res, "manifest", None) is not None:
                break
            if any(s in state for s in ("FAILED", "CANCELED", "CLOSED")):
                message = getattr(getattr(res, "status", None), "error", None)
                st.warning(f"SQL statement did not complete: {state} {message or ''}")
                return pd.DataFrame()
            statement_id = getattr(res, "statement_id", None)
            if not statement_id:
                break
            time.sleep(1)
            res = w.statement_execution.get_statement(statement_id)

        manifest = getattr(res, "manifest", None)
        schema = getattr(manifest, "schema", None)
        columns = getattr(schema, "columns", None) or []
        if not columns:
            st.warning("SQL query returned no schema. Check warehouse permissions and table availability.")
            return pd.DataFrame()
        cols = [c.name for c in columns]
        rows = (res.result.data_array or []) if res.result else []
        return pd.DataFrame(rows, columns=cols)
    except Exception as e:  # noqa: BLE001
        st.warning(f"SQL query failed: {type(e).__name__}: {e}")
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# Logo (inline SVG)
# ──────────────────────────────────────────────────────────────────────────────

LOGO_DARK_BG = """
<svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg" aria-label="PramanaCare logo">
  <defs>
    <linearGradient id="lg-bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#2d4a3a"/>
      <stop offset="100%" stop-color="#1a3429"/>
    </linearGradient>
  </defs>
  <rect width="48" height="48" rx="12" fill="url(#lg-bg)"/>
  <path d="M14 34 L34 34 L24 14 Z" fill="none" stroke="#f3e8c8" stroke-width="1.6" stroke-linejoin="round" opacity="0.55"/>
  <path d="M24 27 L14 34 M24 27 L34 34 M24 27 L24 14" stroke="#f3e8c8" stroke-width="1" opacity="0.35"/>
  <circle cx="14" cy="34" r="3.2" fill="#f7f1e3"/>
  <circle cx="34" cy="34" r="3.2" fill="#f7f1e3"/>
  <circle cx="24" cy="14" r="3.2" fill="#f7f1e3"/>
  <circle cx="24" cy="27" r="3.6" fill="#d97757"/>
  <circle cx="24" cy="27" r="3.6" fill="none" stroke="#d97757" stroke-width="1" opacity="0.4">
    <animate attributeName="r" values="3.6;6.4;3.6" dur="2.4s" repeatCount="indefinite"/>
    <animate attributeName="opacity" values="0.55;0;0.55" dur="2.4s" repeatCount="indefinite"/>
  </circle>
</svg>
"""

LOGO_LIGHT_BG = """
<svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg" aria-label="PramanaCare logo">
  <rect width="48" height="48" rx="12" fill="#2d4a3a"/>
  <path d="M14 34 L34 34 L24 14 Z" fill="none" stroke="#f3e8c8" stroke-width="1.6" stroke-linejoin="round" opacity="0.55"/>
  <path d="M24 27 L14 34 M24 27 L34 34 M24 27 L24 14" stroke="#f3e8c8" stroke-width="1" opacity="0.35"/>
  <circle cx="14" cy="34" r="3.2" fill="#f7f1e3"/>
  <circle cx="34" cy="34" r="3.2" fill="#f7f1e3"/>
  <circle cx="24" cy="14" r="3.2" fill="#f7f1e3"/>
  <circle cx="24" cy="27" r="3.6" fill="#d97757"/>
</svg>
"""

# ──────────────────────────────────────────────────────────────────────────────
# CSS — cream/sage/terracotta system (verbatim from old_frontend)
# ──────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg:            #f7f1e3;
        --bg-2:          #f0e9d6;
        --surface:       #fdfaf2;
        --surface-alt:   #f3ecda;
        --border:        #e6dcc4;
        --border-strong: #d4c8a8;
        --text:          #1c1815;
        --text-2:        #5e554a;
        --text-3:        #8f8475;
        --brand:         #2d4a3a;
        --brand-2:       #1a3429;
        --brand-3:       #3d6b54;
        --accent:        #c4623a;
        --accent-2:      #d97757;
        --accent-soft:   #f5dccb;
        --gold:          #b8924a;
        --good:          #2d4a3a;
        --good-soft:     #d8e4d4;
        --warn:          #b8843c;
        --warn-soft:     #f5e6c4;
        --bad:           #a8412a;
        --bad-soft:      #f3d6c8;
        --shadow-sm:     0 1px 2px rgba(28, 24, 21, 0.04), 0 1px 1px rgba(28,24,21,0.03);
        --shadow:        0 4px 16px rgba(28, 24, 21, 0.06), 0 1px 2px rgba(28,24,21,0.04);
        --shadow-lg:     0 32px 64px -16px rgba(28, 24, 21, 0.18);
        --radius-sm:     8px;
        --radius:        16px;
        --radius-lg:     24px;
    }

    html, body, .stApp, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: var(--bg) !important;
        color: var(--text);
        -webkit-font-smoothing: antialiased;
    }
    .stApp { background: var(--bg) !important; }

    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
    [data-testid="stToolbar"] { display: none; }

    .block-container {
        padding-top: 1.0rem !important;
        padding-bottom: 4rem !important;
        max-width: 1500px !important;
    }

    h1, h2, h3, h4 {
        font-family: 'Fraunces', 'Inter', serif !important;
        font-weight: 500 !important;
        letter-spacing: -0.025em !important;
        color: var(--text);
    }
    h1 { font-size: 56px !important; line-height: 1.04; }
    h2 { font-size: 32px !important; line-height: 1.15; margin: 0; }
    h3 { font-size: 20px !important; line-height: 1.3; margin: 0; }

    .topnav {
        display: flex; align-items: center; justify-content: space-between;
        padding: 14px 22px;
        margin-bottom: 18px;
        background: rgba(253, 250, 242, 0.72);
        backdrop-filter: saturate(180%) blur(14px);
        -webkit-backdrop-filter: saturate(180%) blur(14px);
        border: 1px solid var(--border);
        border-radius: 999px;
        box-shadow: var(--shadow-sm);
    }
    .topnav-left  { display: flex; align-items: center; gap: 12px; }
    .topnav-right { display: flex; align-items: center; gap: 18px; font-size: 13px; color: var(--text-2); }
    .topnav-logo  { width: 32px; height: 32px; }
    .topnav-name  {
        font-family: 'Fraunces', serif; font-size: 18px; font-weight: 600;
        color: var(--text); letter-spacing: -0.015em;
    }
    .topnav-name .dot { color: var(--accent); }
    .topnav-link  { color: var(--text-2); text-decoration: none; font-weight: 500; }
    .topnav-link:hover { color: var(--text); }
    .topnav-pill  {
        background: var(--brand); color: var(--bg);
        padding: 7px 16px; border-radius: 999px;
        font-size: 13px; font-weight: 500;
        display: inline-flex; align-items: center; gap: 8px;
    }
    .live-dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: #6ee7b7;
        box-shadow: 0 0 0 0 rgba(110, 231, 183, 0.75);
        animation: pulse 1.8s infinite;
    }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(110, 231, 183, 0.85); }
        70%  { box-shadow: 0 0 0 10px rgba(110, 231, 183, 0); }
        100% { box-shadow: 0 0 0 0 rgba(110, 231, 183, 0); }
    }

    .home-hero {
        position: relative;
        padding: 96px 48px 80px 48px;
        margin-bottom: 28px;
        border-radius: var(--radius-lg);
        background:
            radial-gradient(800px 400px at 10% 0%, rgba(196, 98, 58, 0.18) 0%, transparent 60%),
            radial-gradient(900px 500px at 100% 100%, rgba(45, 74, 58, 0.22) 0%, transparent 55%),
            radial-gradient(600px 300px at 50% 50%, rgba(217, 119, 87, 0.10) 0%, transparent 60%),
            var(--surface);
        border: 1px solid var(--border);
        overflow: hidden;
        text-align: center;
        animation: fadeUp 700ms ease;
    }
    @keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
    .home-hero::before {
        content: ""; position: absolute; inset: 0;
        background-image: radial-gradient(rgba(28,24,21,0.06) 1px, transparent 1px);
        background-size: 24px 24px;
        opacity: 0.5;
        pointer-events: none;
    }
    .hero-eyebrow {
        position: relative; z-index: 1;
        display: inline-flex; align-items: center; gap: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px; letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--brand);
        background: var(--bg-2);
        padding: 8px 16px;
        border: 1px solid var(--border-strong);
        border-radius: 999px;
        margin-bottom: 24px;
    }
    .hero-eyebrow .live-dot { background: var(--accent); box-shadow: none; }
    .hero-headline {
        position: relative; z-index: 1;
        font-family: 'Fraunces', serif;
        font-size: 76px !important;
        line-height: 0.98;
        letter-spacing: -0.038em;
        color: var(--text);
        margin: 0 auto 22px auto;
        max-width: 980px;
        font-weight: 500;
    }
    .hero-headline em { font-style: italic; color: var(--brand); font-weight: 500; }
    .hero-headline .accent { color: var(--accent); font-style: italic; }
    .hero-sub {
        position: relative; z-index: 1;
        font-size: 19px;
        line-height: 1.55;
        color: var(--text-2);
        max-width: 700px;
        margin: 0 auto 38px auto;
    }
    .hero-cta-row { position: relative; z-index: 1; display: flex; gap: 12px; justify-content: center; }
    .cta-primary {
        background: var(--brand);
        color: var(--bg);
        padding: 14px 26px;
        border-radius: 999px;
        font-weight: 500;
        font-size: 14.5px;
        border: 1px solid var(--brand-2);
        display: inline-flex; align-items: center; gap: 8px;
        transition: transform 200ms ease, box-shadow 200ms ease;
        box-shadow: var(--shadow);
    }
    .cta-primary:hover { transform: translateY(-2px); box-shadow: var(--shadow-lg); }
    .cta-secondary {
        background: var(--surface);
        color: var(--text);
        padding: 14px 26px;
        border-radius: 999px;
        font-weight: 500;
        font-size: 14.5px;
        border: 1px solid var(--border-strong);
        display: inline-flex; align-items: center; gap: 8px;
        transition: transform 200ms ease, border-color 200ms ease;
    }
    .cta-secondary:hover { transform: translateY(-2px); border-color: var(--brand); }

    .stats-strip {
        margin: 36px auto 0 auto;
        max-width: 1100px;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 22px 32px;
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 32px;
        position: relative;
        z-index: 1;
    }
    .stats-strip .stat-num {
        font-family: 'Fraunces', serif;
        font-size: 36px;
        font-weight: 500;
        letter-spacing: -0.02em;
        color: var(--text);
        line-height: 1;
    }
    .stats-strip .stat-num .accent { color: var(--accent); }
    .stats-strip .stat-lab {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: var(--text-3);
        margin-top: 8px;
    }

    .home-section { margin-top: 64px; }
    .home-section .eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px; letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 12px;
    }
    .home-section h2 {
        font-family: 'Fraunces', serif;
        font-size: 44px !important;
        font-weight: 500;
        letter-spacing: -0.025em;
        line-height: 1.1;
        margin-bottom: 14px;
        max-width: 820px;
    }
    .home-section .sec-tag {
        font-size: 17px;
        color: var(--text-2);
        max-width: 680px;
        margin-bottom: 36px;
    }

    .pipeline {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        overflow: hidden;
    }
    .pipeline-step {
        padding: 22px 22px;
        position: relative;
        border-right: 1px solid var(--border);
        transition: background 200ms ease;
    }
    .pipeline-step:hover { background: var(--bg-2); }
    .pipeline-step:last-child { border-right: none; }
    .pipeline-step .num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px; letter-spacing: 0.12em;
        color: var(--text-3); margin-bottom: 10px;
    }
    .pipeline-step .nm {
        font-family: 'Fraunces', serif;
        font-size: 19px; font-weight: 500;
        color: var(--text);
        margin-bottom: 8px;
    }
    .pipeline-step .desc {
        font-size: 13px; color: var(--text-2);
        line-height: 1.5;
    }

    .demo-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 18px;
    }
    .demo-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 28px 26px;
        position: relative;
        overflow: hidden;
        transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
        min-height: 260px;
        display: flex; flex-direction: column;
    }
    .demo-card::after {
        content: ""; position: absolute;
        top: -40px; right: -40px;
        width: 140px; height: 140px;
        border-radius: 50%;
        background: radial-gradient(circle, var(--accent-soft) 0%, transparent 70%);
        opacity: 0.7; pointer-events: none;
    }
    .demo-card:hover { transform: translateY(-3px); box-shadow: var(--shadow); border-color: var(--border-strong); }
    .demo-card .ic {
        width: 38px; height: 38px;
        background: var(--brand);
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        margin-bottom: 18px;
    }
    .demo-card .num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase;
        color: var(--accent); margin-bottom: 6px;
    }
    .demo-card h3 {
        font-family: 'Fraunces', serif;
        font-size: 22px !important; font-weight: 500;
        margin: 0 0 10px 0;
    }
    .demo-card p { font-size: 14px; color: var(--text-2); line-height: 1.55; margin: 0; flex: 1; }
    .demo-card .stat-line {
        margin-top: 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px; color: var(--brand);
        padding-top: 14px;
        border-top: 1px solid var(--border);
    }

    .tech-strip {
        display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
    }
    .tech-chip {
        background: var(--surface);
        border: 1px solid var(--border);
        padding: 8px 16px;
        border-radius: 999px;
        font-size: 12.5px;
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-2);
        transition: all 180ms ease;
    }
    .tech-chip:hover { border-color: var(--brand); color: var(--text); }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--surface);
        padding: 5px;
        border-radius: 12px;
        border: 1px solid var(--border);
        box-shadow: var(--shadow-sm);
        width: fit-content;
        margin: 0 auto 28px auto;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 0 22px;
        border-radius: 8px;
        font-weight: 500;
        font-size: 13.5px;
        color: var(--text-2);
        transition: all 180ms ease;
    }
    .stTabs [data-baseweb="tab"]:hover { color: var(--text); background: var(--bg-2); }
    .stTabs [aria-selected="true"] {
        background: var(--brand) !important;
        color: var(--bg) !important;
        box-shadow: 0 1px 3px rgba(45, 74, 58, 0.25);
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }

    .section-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--accent);
        margin: 0 0 6px 0;
    }
    .section-title {
        font-family: 'Fraunces', serif;
        font-size: 32px !important;
        font-weight: 500;
        letter-spacing: -0.022em;
        margin: 0;
    }
    .section-tag { font-size: 15px; color: var(--text-2); margin: 8px 0 18px 0; max-width: 800px; line-height: 1.55; }

    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 20px 22px;
        box-shadow: var(--shadow-sm);
        transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
    }
    .card.hoverable:hover { transform: translateY(-2px); box-shadow: var(--shadow); border-color: var(--border-strong); }
    .card-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px; letter-spacing: 0.12em;
        text-transform: uppercase; color: var(--text-3);
        margin-bottom: 6px;
    }
    .card-title { font-family: 'Fraunces', serif; font-size: 19px; font-weight: 500; color: var(--text); margin: 0; }
    .card-meta  { font-size: 13px; color: var(--text-2); margin-top: 4px; }

    .kpi {
        position: relative;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 20px 22px;
        overflow: hidden;
        transition: transform 220ms ease, box-shadow 220ms ease;
    }
    .kpi:hover { transform: translateY(-2px); box-shadow: var(--shadow); }
    .kpi.brand {
        background:
            radial-gradient(circle at 100% 0%, rgba(217, 119, 87, 0.25) 0%, transparent 50%),
            linear-gradient(135deg, var(--brand-2) 0%, var(--brand) 100%);
        color: var(--bg); border: none;
    }
    .kpi.brand .kpi-label { color: rgba(247, 241, 227, 0.65); }
    .kpi.brand .kpi-delta.good { color: #c4e0b8; }
    .kpi.brand .kpi-delta.bad  { color: #f3c8b8; }
    .kpi-label { color: var(--text-3); font-size: 11px; font-weight: 500;
                 text-transform: uppercase; letter-spacing: 0.10em; margin-bottom: 10px; }
    .kpi-value { font-family: 'Fraunces', serif; font-size: 32px; font-weight: 500;
                 letter-spacing: -0.025em; line-height: 1; }
    .kpi-delta { font-size: 12px; margin-top: 8px; display: inline-flex; align-items: center; gap: 4px; }
    .kpi-delta.good { color: var(--good); }
    .kpi-delta.bad  { color: var(--bad); }
    .kpi-spark { position: absolute; right: 16px; top: 16px; height: 32px; width: 96px; opacity: 0.85; }

    .trust-badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 5px 11px; border-radius: 999px;
        font-size: 12px; font-weight: 600;
        font-variant-numeric: tabular-nums;
    }
    .trust-badge::before { content: ""; width: 6px; height: 6px; border-radius: 50%; }
    .trust-high { background: var(--good-soft); color: var(--good); }
    .trust-high::before { background: var(--good); }
    .trust-mid  { background: var(--warn-soft); color: var(--warn); }
    .trust-mid::before  { background: var(--warn); }
    .trust-low  { background: var(--bad-soft);  color: var(--bad); }
    .trust-low::before  { background: var(--bad); }

    .citation {
        background: var(--bg-2);
        border-left: 3px solid var(--brand);
        padding: 11px 14px;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px; line-height: 1.55;
        color: var(--text);
        margin-bottom: 8px;
    }
    .citation.bad { border-left-color: var(--bad); background: var(--bad-soft); }
    .citation b { color: var(--text); font-weight: 600; }
    .cit-meta { color: var(--text-3); font-size: 11px; margin-top: 4px; }

    .agent-step {
        display: flex; gap: 12px; align-items: flex-start;
        padding: 11px 14px;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        margin-bottom: 6px;
        font-size: 13px;
        animation: slideIn 280ms ease both;
    }
    @keyframes slideIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
    .step-num {
        flex: 0 0 26px; height: 26px;
        background: var(--brand);
        color: var(--bg);
        border-radius: 7px;
        display: flex; align-items: center; justify-content: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px; font-weight: 600;
    }
    .step-name { font-weight: 600; color: var(--text); font-family: 'Inter', sans-serif; }
    .step-detail { color: var(--text-2); font-size: 12px; margin-top: 2px; }
    .step-output {
        font-family: 'JetBrains Mono', monospace;
        color: var(--brand-3);
        font-size: 11px; margin-top: 4px;
        background: var(--bg-2);
        padding: 4px 8px; border-radius: 5px;
        display: inline-block;
    }

    .chip {
        display: inline-flex; align-items: center; gap: 4px;
        padding: 3px 10px; border-radius: 999px;
        background: var(--bg-2);
        border: 1px solid var(--border);
        font-size: 11px; color: var(--text-2);
        margin: 2px 4px 2px 0;
        font-family: 'JetBrains Mono', monospace;
    }
    .chip.accent { background: var(--accent-soft); color: var(--accent); border-color: #ecc5b3; }
    .chip.bad    { background: var(--bad-soft); color: var(--bad); border-color: #ecc1b3; }

    .empty {
        background: var(--surface);
        border: 1px dashed var(--border-strong);
        border-radius: var(--radius);
        padding: 60px 32px;
        text-align: center;
    }
    .empty-title { font-family: 'Fraunces', serif; font-size: 22px; color: var(--text); margin-bottom: 6px; }
    .empty-tag   { font-size: 14px; color: var(--text-2); }

    .stButton > button {
        background: var(--surface);
        border: 1px solid var(--border);
        color: var(--text);
        font-size: 12.5px;
        font-weight: 500;
        border-radius: 12px;
        padding: 11px 14px;
        text-align: left;
        transition: all 180ms ease;
        line-height: 1.4;
        white-space: normal;
        height: auto !important;
        box-shadow: var(--shadow-sm);
    }
    .stButton > button:hover {
        border-color: var(--brand);
        background: var(--surface);
        color: var(--text);
        transform: translateY(-1px);
        box-shadow: var(--shadow);
    }
    .stButton > button:focus { outline: none !important; box-shadow: 0 0 0 3px rgba(45,74,58,0.15); }

    [data-testid="stChatMessage"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 12px 14px !important;
        box-shadow: var(--shadow-sm);
    }
    div[data-testid="stChatInput"] {
        border-radius: 14px !important;
        border: 1px solid var(--border-strong) !important;
        background: var(--surface) !important;
        box-shadow: var(--shadow-sm);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: var(--radius);
        overflow: hidden;
    }

    [data-baseweb="select"] > div, .stSelectbox > div > div {
        border-radius: 12px !important;
        border-color: var(--border-strong) !important;
        background: var(--surface) !important;
    }
    .stSlider > div > div > div { background: var(--brand) !important; }
    .stSlider [role="slider"] {
        background: var(--brand) !important;
        border: 3px solid var(--surface) !important;
        box-shadow: 0 2px 8px rgba(45,74,58,0.18) !important;
    }
    [data-baseweb="checkbox"] [role="checkbox"][aria-checked="true"] { background: var(--brand) !important; border-color: var(--brand) !important; }

    [data-testid="stPlotlyChart"] {
        border-radius: var(--radius);
        overflow: hidden;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border);
        background: var(--surface);
    }

    .footnote {
        margin-top: 18px;
        padding: 14px 18px;
        background: var(--surface);
        border-radius: var(--radius);
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: var(--text-2);
        border: 1px solid var(--border);
        line-height: 1.6;
    }
    .footnote strong { color: var(--text); font-weight: 600; }

    .global-footer {
        margin-top: 48px;
        padding: 32px 32px;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        display: grid;
        grid-template-columns: 2fr 1fr 1fr 1fr;
        gap: 32px;
    }
    .global-footer .col-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase;
        color: var(--text-3); margin-bottom: 14px;
    }
    .global-footer .footer-brand {
        display: flex; align-items: center; gap: 12px; margin-bottom: 12px;
    }
    .global-footer .footer-brand-name {
        font-family: 'Fraunces', serif; font-size: 22px; font-weight: 500;
        color: var(--text);
    }
    .global-footer .footer-tag { font-size: 13px; color: var(--text-2); max-width: 320px; line-height: 1.55; }
    .global-footer ul { list-style: none; padding: 0; margin: 0; }
    .global-footer li {
        font-size: 13px; color: var(--text-2);
        padding: 4px 0;
    }
    .global-footer .copy {
        grid-column: 1 / -1;
        margin-top: 16px;
        padding-top: 16px;
        border-top: 1px solid var(--border);
        font-size: 12px; color: var(--text-3);
        display: flex; justify-content: space-between; align-items: center;
        font-family: 'JetBrains Mono', monospace;
    }

    .fac-row {
        display: grid;
        grid-template-columns: 1fr auto auto;
        gap: 16px;
        padding: 14px 16px;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        margin-bottom: 8px;
        align-items: center;
        transition: transform 200ms ease, border-color 200ms ease;
    }
    .fac-row:hover { transform: translateY(-1px); border-color: var(--border-strong); }
    .fac-row .fac-name { font-family: 'Fraunces', serif; font-weight: 500; font-size: 17px; color: var(--text); }
    .fac-row .fac-meta { color: var(--text-2); font-size: 12px; margin-top: 2px; }
    .fac-row .fac-dist { font-family: 'Fraunces', serif; font-weight: 500; font-size: 19px; color: var(--text); }
    .fac-row .fac-unit { color: var(--text-3); font-size: 11px; }

    [data-testid="stExpander"] {
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        background: var(--surface) !important;
        margin-top: 8px;
    }
    .streamlit-expanderHeader, [data-testid="stExpander"] summary {
        font-family: 'Inter', sans-serif !important;
        font-size: 13px !important;
        color: var(--text-2) !important;
    }

    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 6px; }
    ::-webkit-scrollbar-thumb:hover { background: #c4b899; }

    [data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
    [data-testid="stSidebar"] .stMarkdown { font-size: 13px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Data shapes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Citation:
    claim: str
    source_field: str
    source_row: int
    extraction_confidence: float
    validator_status: str
    supporting_evidence: list[str] = field(default_factory=list)


@dataclass
class Facility:
    facility_id: str
    name: str
    city: str
    state: str
    pin: str
    lat: float
    lon: float
    facility_type: str
    specialties: list[str]
    capabilities: list[str]
    equipment: list[str]
    trust_score: float            # normalized to 0–1
    trust_score_raw: int          # 0–100 from gold table
    trust_components: dict[str, float]
    contradictions: list[str]
    citations: list[Citation]


# ──────────────────────────────────────────────────────────────────────────────
# Real backend loaders (replace old mock layer)
# ──────────────────────────────────────────────────────────────────────────────

_FACILITY_LIMIT = 500


def _coerce_array(val) -> list[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val if x is not None]
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if x is not None]
        except (ValueError, TypeError):
            pass
        quoted = re.findall(r"""['"]([^'"]+)['"]""", s)
        if quoted:
            return quoted
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if not inner:
                return []
            return [x.strip() for x in re.split(r"\s*[,|]\s*", inner) if x.strip()]
        return [s]
    return [str(val)]


def _coerce_flags(val) -> list[dict]:
    """flags column is array<struct{rule_id, severity, message, evidence, citation_column}>."""
    if val is None:
        return []
    raw = val
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            raw = json.loads(s)
        except (ValueError, TypeError):
            return []
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw:
        if isinstance(item, dict):
            out.append({
                "rule_id":         item.get("rule_id", ""),
                "severity":        item.get("severity", ""),
                "message":         item.get("message", ""),
                "evidence":        item.get("evidence", ""),
                "citation_column": item.get("citation_column", ""),
            })
    return out


def _build_citations(row_idx: int, capabilities: list[str], equipment: list[str],
                     flags: list[dict]) -> list[Citation]:
    cites: list[Citation] = []
    flag_msgs = {f.get("message", "").lower() for f in flags}

    for cap in capabilities[:5]:
        # heuristic: capability is "failed" if any flag message references it
        bad = any(cap.lower()[:24] in m for m in flag_msgs if m)
        cites.append(Citation(
            claim=cap,
            source_field="capability",
            source_row=row_idx,
            extraction_confidence=0.95 if equipment else 0.65,
            validator_status="failed" if bad else "passed",
            supporting_evidence=equipment[:2] if equipment else [],
        ))
    for fl in flags[:5]:
        cites.append(Citation(
            claim=fl.get("message") or fl.get("rule_id") or "validator flag",
            source_field=fl.get("citation_column") or "flag",
            source_row=row_idx,
            extraction_confidence=1.0,
            validator_status="failed",
            supporting_evidence=[fl.get("evidence")] if fl.get("evidence") else [],
        ))
    return cites


def _trust_components(capabilities: list[str], equipment: list[str], flags: list[dict],
                      row: dict) -> dict[str, float]:
    claim_density = min(1.0, (len(capabilities) + len(equipment)) / 20.0)
    evidence_agreement = 0.95 if equipment else 0.30
    contradiction_penalty = max(0.0, 1.0 - 0.18 * len(flags))

    high_severity_flags = sum(1 for f in flags if str(f.get("severity")).upper() == "HIGH")
    location_coherence = max(0.30, 0.95 - 0.20 * high_severity_flags)

    fillable = ["name", "state", "city", "pin", "facility_type",
                "latitude", "longitude", "capacity", "number_doctors",
                "year_established", "description"]
    filled = sum(1 for k in fillable if row.get(k) not in (None, "", []))
    structured_fillrate = round(filled / len(fillable), 2)

    return {
        "claim_density":         round(claim_density, 2),
        "evidence_agreement":    round(evidence_agreement, 2),
        "contradiction_penalty": round(contradiction_penalty, 2),
        "location_coherence":    round(location_coherence, 2),
        "structured_fillrate":   structured_fillrate,
    }


@st.cache_data(ttl=600, show_spinner=False)
def load_facilities() -> list[Facility]:
    sql = f"""
    SELECT facility_id, name, facility_type, state, city, pin,
           latitude, longitude, specialties, equipment, capability,
           capacity, number_doctors, year_established, description,
           trust_score, flags
    FROM {NS}.gold_facilities
    WHERE latitude IS NOT NULL AND longitude IS NOT NULL
    ORDER BY trust_score DESC
    LIMIT {_FACILITY_LIMIT}
    """
    df = run_sql(sql)
    if df.empty:
        return []

    out: list[Facility] = []
    for i, r in df.iterrows():
        row = r.to_dict()
        specialties = _coerce_array(row.get("specialties"))
        equipment   = _coerce_array(row.get("equipment"))
        capabilities = _coerce_array(row.get("capability"))
        flags = _coerce_flags(row.get("flags"))

        try:
            ts_raw = int(float(row.get("trust_score") or 0))
        except (ValueError, TypeError):
            ts_raw = 0
        ts_raw = max(0, min(100, ts_raw))
        trust = round(ts_raw / 100.0, 2)

        out.append(Facility(
            facility_id=str(row.get("facility_id") or f"VF{i:05d}"),
            name=str(row.get("name") or "—"),
            city=str(row.get("city") or "—"),
            state=str(row.get("state") or "—"),
            pin=str(row.get("pin") or ""),
            lat=float(row.get("latitude") or 0.0),
            lon=float(row.get("longitude") or 0.0),
            facility_type=str(row.get("facility_type") or "facility"),
            specialties=specialties,
            capabilities=capabilities,
            equipment=equipment,
            trust_score=trust,
            trust_score_raw=ts_raw,
            trust_components=_trust_components(capabilities, equipment, flags, row),
            contradictions=[f.get("message", "") for f in flags if f.get("message")],
            citations=_build_citations(i, capabilities, equipment, flags),
        ))
    return out


# Maps the desert-tab specialty UI options to keywords that exist in the
# gold_facilities specialties / capability / equipment arrays.
_DESERT_KEYWORDS = {
    "Dialysis": ["dialysis", "nephrology"],
    "Oncology": ["oncology", "chemotherapy", "radiation"],
    "Trauma":   ["trauma", "emergency"],
    "ICU":      ["icu", "critical", "ventilator"],
    "Cardiac":  ["cardio", "cardiac", "angio"],
}


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@st.cache_data(ttl=600, show_spinner=False)
def load_district_desert_data(specialty: str) -> pd.DataFrame:
    keywords = _DESERT_KEYWORDS.get(specialty, [specialty.lower()])
    spec_clause = " OR ".join([
        f"exists(specialties, x -> contains(lower(x), '{kw}')) "
        f"OR exists(capability, x -> contains(lower(x), '{kw}')) "
        f"OR exists(equipment, x -> contains(lower(x), '{kw}'))"
        for kw in keywords
    ])
    sql = f"""
    SELECT state, city,
           COUNT(*) AS n_facilities,
           SUM(CASE WHEN {spec_clause} THEN 1 ELSE 0 END) AS n_specialty,
           AVG(latitude)  AS lat,
           AVG(longitude) AS lon
    FROM {NS}.gold_facilities
    WHERE state IS NOT NULL AND city IS NOT NULL
      AND latitude IS NOT NULL AND longitude IS NOT NULL
    GROUP BY state, city
    """
    df = run_sql(sql)
    if df.empty:
        return df

    df["n_facilities"] = pd.to_numeric(df["n_facilities"], errors="coerce").fillna(0).astype(int)
    df["n_specialty"]  = pd.to_numeric(df["n_specialty"],  errors="coerce").fillna(0).astype(int)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

    df["coverage_pct"] = (df["n_specialty"] / df["n_facilities"].clip(lower=1) * 100).round(2)
    ci = df.apply(lambda r: _wilson_ci(int(r["n_specialty"]), int(r["n_facilities"])), axis=1)
    df["ci_low"]  = (ci.apply(lambda x: x[0]) * 100).round(2)
    df["ci_high"] = (ci.apply(lambda x: x[1]) * 100).round(2)
    df["deficit_score"] = ((100 - df["coverage_pct"]) / 100 * df["n_facilities"]).round(2)
    df = df.rename(columns={"city": "district"})
    return df.sort_values("deficit_score", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=600, show_spinner=False)
def load_overview_metrics() -> dict:
    sql = f"""
    SELECT COUNT(*) AS n_facilities,
           AVG(trust_score)/100.0 AS mean_trust,
           SUM(CASE WHEN size(flags) > 0 THEN 1 ELSE 0 END) AS n_with_flags,
           SUM(size(flags)) AS n_flags_total
    FROM {NS}.gold_facilities
    """
    df = run_sql(sql)
    if df.empty:
        return {"n_facilities": 0, "mean_trust": 0.0, "n_with_flags": 0, "n_flags_total": 0}
    r = df.iloc[0]
    return {
        "n_facilities":  int(float(r.get("n_facilities") or 0)),
        "mean_trust":    float(r.get("mean_trust") or 0.0),
        "n_with_flags":  int(float(r.get("n_with_flags") or 0)),
        "n_flags_total": int(float(r.get("n_flags_total") or 0)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# UI helpers
# ──────────────────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def trust_class(s): return "trust-high" if s >= 0.70 else ("trust-mid" if s >= 0.45 else "trust-low")
def trust_label(s): return "High" if s >= 0.70 else ("Moderate" if s >= 0.45 else "Low")
def trust_badge_html(s): return f'<span class="trust-badge {trust_class(s)}">Trust {s:.2f} · {trust_label(s)}</span>'


def section_header(eyebrow, title, tag=""):
    st.markdown(
        f'<p class="section-eyebrow">{eyebrow}</p>'
        f'<h2 class="section-title">{title}</h2>'
        + (f'<p class="section-tag">{tag}</p>' if tag else ""),
        unsafe_allow_html=True,
    )


def topnav():
    st.markdown(
        f"""
        <div class="topnav">
            <div class="topnav-left">
                <div class="topnav-logo">{LOGO_LIGHT_BG}</div>
                <div class="topnav-name">PramanaCare<span class="dot">.ai</span></div>
            </div>
            <div class="topnav-right">
                <span class="topnav-link">Product</span>
                <span class="topnav-link">Methodology</span>
                <span class="topnav-link">Demo</span>
                <span class="topnav-pill"><span class="live-dot"></span> Vector index live</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_tile(label, value, delta=None, tone=None, brand=False, sparkline=None):
    spark_html = ""
    if sparkline:
        n = len(sparkline)
        lo, hi = min(sparkline), max(sparkline)
        rng_v = max(hi - lo, 1e-9)
        pts = " ".join(f"{96*i/(n-1):.1f},{32-32*(v-lo)/rng_v:.1f}" for i, v in enumerate(sparkline))
        stroke = "rgba(247,241,227,0.55)" if brand else "var(--brand-3)"
        spark_html = (
            f'<svg class="kpi-spark" viewBox="0 0 96 32" preserveAspectRatio="none">'
            f'<polyline fill="none" stroke="{stroke}" stroke-width="1.6" points="{pts}"/></svg>'
        )
    delta_html = f'<div class="kpi-delta {tone or ""}">{delta}</div>' if delta else ""
    cls = "kpi brand" if brand else "kpi"
    return (f'<div class="{cls}">{spark_html}'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>{delta_html}</div>')


def render_kpi_row(items):
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        col.markdown(kpi_tile(**item), unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Agent — local ranker + real model streaming
# ──────────────────────────────────────────────────────────────────────────────

QUERY_PRESETS = {
    "I need dialysis near 110001 (Delhi)":
        ("dialysis", 28.6139, 77.2090),
    "Find a hospital that can perform an emergency appendectomy in rural Bihar":
        ("generalSurgery", 25.5941, 85.1376),
    "Where can a stage-3 lung cancer patient get radiation in Andhra Pradesh?":
        ("radiationOncology", 17.6868, 83.2185),
    "Closest 24x7 trauma centre to Aurangabad":
        ("trauma", 19.8762, 75.3433),
    "Cardiac angioplasty within 50 km of Jaipur":
        ("interventionalCardiology", 26.9124, 75.7873),
    "Maternity care in the Sundarbans delta":
        ("obstetrics", 22.3208, 88.6635),
}


_AGENT_SPECIALTY_KEYWORDS = {
    "dialysis": ["dialysis", "nephrology", "hemodialysis"],
    "generalsurgery": ["generalsurgery", "general surgery", "surgery", "appendectomy", "laparoscopic"],
    "radiationoncology": ["radiationoncology", "radiation", "radiotherapy", "oncology", "cancer"],
    "trauma": ["trauma", "emergency", "criticalcare", "critical care"],
    "interventionalcardiology": ["interventionalcardiology", "cardiology", "cardiac", "angioplasty", "cath"],
    "obstetrics": ["obstetrics", "gynecology", "maternity", "delivery", "pregnancy"],
    "generalmedicine": ["generalmedicine", "general medicine", "familymedicine", "family medicine"],
}


def _specialty_keywords(specialty: str) -> list[str]:
    key = re.sub(r"[^a-z0-9]+", "", specialty.lower())
    return _AGENT_SPECIALTY_KEYWORDS.get(key, [specialty.lower()])


def agent_plan(query, lat, lon, specialty_hint):
    return [
        {"name": "Planner",   "detail": "Decompose query into (specialty, geography, urgency, trust threshold).",
         "output": json.dumps({"specialty": specialty_hint, "anchor": [lat, lon],
                               "urgency": "high" if "emergency" in query.lower() else "routine",
                               "min_trust": 0.45})},
        {"name": "Vector Search", "detail": "Embed query → retrieve top-50 canonical claim matches from Mosaic Vector Search.",
         "output": "candidates retrieved · grounded in canonical claim graph"},
        {"name": "Graph Traversal", "detail": "Walk kg_edges HAS_CAPABILITY → PERFORMS_PROCEDURE → HAS_EQUIPMENT.",
         "output": "evidence chains assembled across capability/equipment edges"},
        {"name": "Geo Filter", "detail": "Haversine cutoff + state-coherence filter against PIN-state mapping.",
         "output": "filtered to candidates within reach of the anchor"},
        {"name": "Ranker", "detail": "Score = 0.45·trust + 0.35·specialty_match − 0.20·distance_norm.",
         "output": "top-K facilities surfaced"},
        {"name": "Validator", "detail": "Run rule-based contradiction checks against returned facilities.",
         "output": "validator status attached to every recommendation"},
        {"name": "Synthesizer", "detail": "Compose answer with row-level citations and MLflow trace IDs.",
         "output": "answer composed · grounded in real gold tables"},
    ]


def agent_pick(facilities: list[Facility], specialty: str, lat: float, lon: float, k: int = 5):
    if not facilities:
        return []
    keywords = _specialty_keywords(specialty)
    scored = []
    for f in facilities:
        spec_match = 0.0
        if any(any(kw in s.lower() for kw in keywords) for s in f.specialties):
            spec_match = 1.0
        elif any(any(kw in c.lower() for kw in keywords) for c in f.capabilities):
            spec_match = 0.7
        elif any(any(kw in e.lower() for kw in keywords) for e in f.equipment):
            spec_match = 0.5
        if spec_match == 0.0:
            continue
        d = haversine_km(lat, lon, f.lat, f.lon)
        if d > 1500:
            continue
        type_bonus = 0.05 if f.facility_type.lower() == "hospital" else 0.0
        score = 0.45 * f.trust_score + 0.35 * spec_match + type_bonus - 0.20 * min(1.0, d / 1500.0)
        scored.append((f, d, score))
    scored.sort(key=lambda x: x[2], reverse=True)
    return [(f, d) for f, d, _ in scored[:k]]


def stream_chat_response(messages: list[dict]):
    """Yield text chunks from the real model serving endpoint."""
    client = _openai_client()
    if client is None or not ENDPOINT:
        yield ("_(Model serving endpoint not configured. Set the `SERVING_ENDPOINT_NAME` "
               "environment variable to enable live answers.)_")
        return
    try:
        resp = client.chat.completions.create(
            model=ENDPOINT, messages=messages, stream=True, timeout=180,
        )
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta:
                yield chunk.choices[0].delta.content or ""
    except Exception as e:                                   # noqa: BLE001
        yield f"\n\n_Model call failed: {e}_"


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""<div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
        <div style="width:32px;height:32px;">{LOGO_LIGHT_BG}</div>
        <div style="font-family:'Fraunces',serif;font-weight:600;font-size:18px;">PramanaCare<span style="color:#c4623a;">.ai</span></div>
        </div>""", unsafe_allow_html=True)
    st.markdown("Confidence-calibrated healthcare intelligence for India.")
    st.divider()
    st.markdown("### Stack")
    st.markdown(
        "- Mosaic AI Vector Search  \n- Agent Bricks (planner + validator)  \n"
        "- Unity Catalog governance  \n- MLflow 3 Tracing  \n- NetworkX in-memory KG"
    )
    st.divider()
    st.markdown("### Backend")
    st.caption(f"Catalog: `{CATALOG}.{SCHEMA}`")
    st.caption(f"Endpoint: `{ENDPOINT or '—'}`")
    st.caption(f"Warehouse: `{WAREHOUSE_ID or '—'}`")
    st.divider()
    st.caption(f"Build: live · {datetime.now():%Y-%m-%d %H:%M}")


topnav()

tab_home, tab_finder, tab_desert, tab_audit = st.tabs([
    "Home", "Patient Finder", "Medical Desert Map", "Trust & Audit",
])

# ──────────────────────────────────────────────────────────────────────────────
# Tab 0 — Home / Landing
# ──────────────────────────────────────────────────────────────────────────────

with tab_home:
    metrics = load_overview_metrics()
    n_facilities  = metrics["n_facilities"]
    mean_trust    = metrics["mean_trust"]
    n_with_flags  = metrics["n_with_flags"]
    n_flags_total = metrics["n_flags_total"]

    st.markdown(
        f"""
        <div class="home-hero">
            <div class="hero-eyebrow"><span class="live-dot"></span> Built on Databricks · MIT × Databricks for Good</div>
            <h1 class="hero-headline">
                Healthcare you can <em>trust.</em><br>
                Down to the <span class="accent">claim.</span>
            </h1>
            <p class="hero-sub">
                PramanaCare turns messy Indian medical-facility records into a
                confidence-calibrated knowledge graph — so a patient in rural Bihar can find the
                nearest hospital that <em>actually</em> performs the surgery they need, and a planner
                can see the deserts before the headlines do.
            </p>
            <div class="hero-cta-row">
                <span class="cta-primary">Open the live demo →</span>
                <span class="cta-secondary">How it works</span>
            </div>
            <div class="stats-strip">
                <div>
                    <div class="stat-num">{n_facilities:,}</div>
                    <div class="stat-lab">facilities indexed</div>
                </div>
                <div>
                    <div class="stat-num">{n_flags_total:,}</div>
                    <div class="stat-lab">validator flags</div>
                </div>
                <div>
                    <div class="stat-num"><span class="accent">{n_with_flags:,}</span></div>
                    <div class="stat-lab">facilities with contradictions</div>
                </div>
                <div>
                    <div class="stat-num">{mean_trust:.2f}</div>
                    <div class="stat-lab">mean trust score</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="home-section">
            <div class="eyebrow">The problem</div>
            <h2>Most facility data lies — quietly.</h2>
            <p class="sec-tag">
                A clinic claims it can perform advanced surgery. It lists no anaesthesiologist.
                A hospital lists itself in Uttar Pradesh — its actual address is in Maharashtra.
                A trauma centre ticks every box on its registration form. It owns no defibrillator.
                Multiplied across 1.4 billion people, these small lies become triage delays, missed
                treatments, and avoidable deaths. PramanaCare is a verifier-first stack designed to
                surface them at scale.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="home-section">
            <div class="eyebrow">How it works</div>
            <h2>Five layers · one source of truth.</h2>
            <p class="sec-tag">
                Every layer is a Delta table in Unity Catalog. Every step writes an MLflow trace.
                The LLM is a scalpel, not a hammer — it touches only the ambiguous residual that
                embeddings can't canonicalize on their own.
            </p>
            <div class="pipeline">
                <div class="pipeline-step">
                    <div class="num">01 · BRONZE</div>
                    <div class="nm">Land</div>
                    <div class="desc">Raw records → Delta with PIN preservation, lat/lon validation, dtype casting.</div>
                </div>
                <div class="pipeline-step">
                    <div class="num">02 · SILVER</div>
                    <div class="nm">Canonicalize</div>
                    <div class="desc">JSON arrays explode into a long claims table. Embedding clusters fold raw equipment strings into canonical IDs. LLM tags only the residual.</div>
                </div>
                <div class="pipeline-step">
                    <div class="num">03 · GOLD</div>
                    <div class="nm">Graph</div>
                    <div class="desc">kg_nodes + kg_edges Delta tables. Facility ↔ specialty ↔ equipment ↔ procedure, with extraction confidence on every edge.</div>
                </div>
                <div class="pipeline-step">
                    <div class="num">04 · AGENT</div>
                    <div class="nm">Reason</div>
                    <div class="desc">Planner → Vector Search → Graph traversal → Geo filter → Ranker → Validator → Synthesizer. Self-corrects on contradiction.</div>
                </div>
                <div class="pipeline-step">
                    <div class="num">05 · APP</div>
                    <div class="nm">Surface</div>
                    <div class="desc">Patient Finder, Desert Map, Trust Audit. Every aggregate ships with a 95% confidence interval. Every recommendation cites the rows.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="home-section">
            <div class="eyebrow">What it shows</div>
            <h2>Three demos. Three different kinds of truth.</h2>
            <p class="sec-tag">Click any tab above — the system is wired to the live gold tables.</p>
            <div class="demo-grid">
                <div class="demo-card">
                    <div class="ic">""" + LOGO_LIGHT_BG + """</div>
                    <div class="num">Demo 01</div>
                    <h3>The Dialysis Desert</h3>
                    <p>Pick a specialty and watch entire districts light up red where coverage drops to zero. PramanaCare puts a 95% Wilson interval around the rate.</p>
                    <div class="stat-line">city-level coverage · 95% CI · sortable deficit score</div>
                </div>
                <div class="demo-card">
                    <div class="ic">""" + LOGO_LIGHT_BG + """</div>
                    <div class="num">Demo 02</div>
                    <h3>Trust Scorer in Action</h3>
                    <p>Find every facility that claims advanced procedures while listing zero equipment. The validator flags it; the trust score drops; the chat surfaces it as a citation.</p>
                    <div class="stat-line">""" + f"{n_flags_total:,}" + """ contradictions · validator self-correction loop</div>
                </div>
                <div class="demo-card">
                    <div class="ic">""" + LOGO_LIGHT_BG + """</div>
                    <div class="num">Demo 03</div>
                    <h3>Data-Quality Audit</h3>
                    <p>Facilities whose listed state contradicts their coordinates. PramanaCare surfaces these geographic incoherencies as a first-class signal, not a footnote.</p>
                    <div class="stat-line">Location coherence · PIN ↔ district ↔ state</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="home-section">
            <div class="eyebrow">The stack</div>
            <h2>Built on Databricks, end to end.</h2>
            <p class="sec-tag">No lock-in to anything that won't run on Free Edition. Every table governed by Unity Catalog. Every agent step traced in MLflow 3.</p>
            <div class="tech-strip">
                <span class="tech-chip">Databricks Free Edition</span>
                <span class="tech-chip">Unity Catalog</span>
                <span class="tech-chip">Mosaic AI Vector Search</span>
                <span class="tech-chip">Agent Bricks</span>
                <span class="tech-chip">MLflow 3 Tracing</span>
                <span class="tech-chip">Delta Lake</span>
                <span class="tech-chip">NetworkX</span>
                <span class="tech-chip">Streamlit</span>
                <span class="tech-chip">Plotly · Carto Positron</span>
                <span class="tech-chip">Census of India 2011</span>
                <span class="tech-chip">IHME GBD India</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Tab 1 — Patient Finder
# ──────────────────────────────────────────────────────────────────────────────

with tab_finder:
    facilities = load_facilities()

    overview = load_overview_metrics()
    render_kpi_row([
        {"label": "Facilities indexed", "value": f"{overview['n_facilities']:,}",
         "delta": "live from gold_facilities", "tone": None, "brand": True,
         "sparkline": [0.2, 0.4, 0.45, 0.6, 0.7, 0.85, 1.0]},
        {"label": "Validator flags", "value": f"{overview['n_flags_total']:,}",
         "delta": "across the catalog", "tone": "bad",
         "sparkline": [0.3, 0.5, 0.6, 0.55, 0.7, 0.8, 0.92]},
        {"label": "Facilities with contradictions", "value": f"{overview['n_with_flags']:,}",
         "delta": "validator self-correction", "tone": "bad",
         "sparkline": [0.1, 0.2, 0.25, 0.5, 0.6, 0.75, 0.9]},
        {"label": "Mean trust score", "value": f"{overview['mean_trust']:.2f}",
         "delta": "0–1 normalized", "tone": "good",
         "sparkline": [0.55, 0.6, 0.62, 0.65, 0.68, 0.7, 0.71]},
    ])

    st.markdown(" ")
    section_header(
        "01 · Patient Finder",
        "Find the right facility under real-world constraints",
        "Ask in natural language. The agent decomposes the query, retrieves from the knowledge graph, "
        "validates against contradictions, and returns ranked facilities with row-level citations."
    )

    if "messages" not in st.session_state: st.session_state.messages = []
    if "last_results" not in st.session_state: st.session_state.last_results = None

    chat_col, map_col = st.columns([0.42, 0.58], gap="large")

    with chat_col:
        st.markdown("<p class='section-eyebrow'>Quick start</p>", unsafe_allow_html=True)
        preset_cols = st.columns(2)
        for i, preset in enumerate(QUERY_PRESETS.keys()):
            if preset_cols[i % 2].button(preset, key=f"preset_{i}", use_container_width=True):
                st.session_state.pending_query = preset

        st.markdown("<p class='section-eyebrow' style='margin-top:14px;'>Conversation</p>", unsafe_allow_html=True)
        msg_container = st.container(height=380, border=False)
        with msg_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"], unsafe_allow_html=True)

        prompt = st.chat_input("Ask about facilities, capabilities, locations…")
        if "pending_query" in st.session_state:
            prompt = st.session_state.pop("pending_query")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            specialty, lat, lon = QUERY_PRESETS.get(prompt, ("generalMedicine", 20.5937, 78.9629))

            with msg_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    plan_box = st.empty()
                    steps_html = ""
                    for i, step in enumerate(agent_plan(prompt, lat, lon, specialty), start=1):
                        steps_html += (
                            f'<div class="agent-step">'
                            f'<div class="step-num">{i:02d}</div>'
                            f'<div style="flex:1;">'
                            f'<div class="step-name">{step["name"]}</div>'
                            f'<div class="step-detail">{step["detail"]}</div>'
                            f'<div class="step-output">{step["output"]}</div>'
                            f'</div></div>'
                        )
                        plan_box.markdown(steps_html, unsafe_allow_html=True)
                        time.sleep(0.12)

                    answer_box = st.empty()
                    streamed = ""
                    chat_history = [m for m in st.session_state.messages
                                    if m["role"] in ("user", "assistant")
                                    and isinstance(m.get("content"), str)
                                    and not m["content"].lstrip().startswith("<")]
                    for chunk in stream_chat_response(chat_history):
                        streamed += chunk
                        answer_box.markdown(streamed)

                    picks = agent_pick(facilities, specialty, lat, lon, k=5)
                    summary = (
                        f"\n\n**{len(picks)}** facilities matching `{specialty}` within reach of "
                        f"({lat:.3f}, {lon:.3f}) — ranked by trust × specialty match × inverse distance."
                    )
                    st.markdown(summary)

            assistant_payload = (
                steps_html
                + (f"<div style='margin-top:8px'>{streamed}</div>" if streamed else "")
                + f"<div style='margin-top:8px'>{summary}</div>"
            )
            st.session_state.messages.append({"role": "assistant", "content": assistant_payload})
            st.session_state.last_results = (specialty, lat, lon, picks)

    with map_col:
        st.markdown("<p class='section-eyebrow'>Recommended facilities</p>", unsafe_allow_html=True)
        if st.session_state.last_results is None:
            st.markdown(
                '<div class="empty">'
                '<p class="empty-title">No query yet</p>'
                '<p class="empty-tag">Click a quick-start prompt or type your own to see the agent reason '
                'over the knowledge graph and return ranked facilities with citations.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            picks = []
        else:
            specialty, anchor_lat, anchor_lon, picks = st.session_state.last_results

            if not picks:
                st.markdown(
                    '<div class="empty">'
                    '<p class="empty-title">No matching facilities</p>'
                    '<p class="empty-tag">No facility within 1,500 km matched the requested specialty. '
                    'Try a different query or broaden the geography.</p>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                map_df = pd.DataFrame([
                    {"name": f.name, "city": f.city, "state": f.state,
                     "lat": f.lat, "lon": f.lon, "trust": f.trust_score,
                     "distance_km": round(d, 1), "tier": trust_label(f.trust_score)}
                    for f, d in picks
                ])
                anchor_df = pd.DataFrame([{"name": "Patient location", "lat": anchor_lat, "lon": anchor_lon,
                                           "trust": 1.0, "tier": "Anchor", "city": "—", "state": "—",
                                           "distance_km": 0.0}])
                plot_df = pd.concat([map_df, anchor_df], ignore_index=True)

                fig = px.scatter_mapbox(
                    plot_df, lat="lat", lon="lon", color="tier",
                    color_discrete_map={"High": "#2d4a3a", "Moderate": "#b8843c",
                                        "Low": "#a8412a", "Anchor": "#c4623a"},
                    hover_name="name",
                    hover_data={"city": True, "state": True, "trust": True, "distance_km": True,
                                "lat": False, "lon": False, "tier": False},
                    size=[18] * len(plot_df), size_max=18,
                    zoom=4.4, height=380,
                )
                fig.update_layout(
                    mapbox_style="carto-positron",
                    margin={"l": 0, "r": 0, "t": 0, "b": 0},
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                                font=dict(size=11)),
                    paper_bgcolor="white",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("<p class='section-eyebrow' style='margin-top:14px;'>Citations &amp; validator status</p>",
                            unsafe_allow_html=True)
                for f, d in picks:
                    st.markdown(
                        f'<div class="fac-row">'
                        f'<div><div class="fac-name">{f.name}</div>'
                        f'<div class="fac-meta">{f.facility_type.title()} · {f.city}, {f.state} · PIN {f.pin}</div></div>'
                        f'<div style="text-align:right;"><div class="fac-dist">{d:.1f}<span class="fac-unit"> km</span></div>'
                        f'<div class="fac-unit">haversine</div></div>'
                        f'<div>{trust_badge_html(f.trust_score)}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    with st.expander("Claim-level evidence  ·  validator findings"):
                        if not f.citations:
                            st.markdown("<div class='footnote'>No claim-level rows attached to this facility.</div>",
                                        unsafe_allow_html=True)
                        for cit in f.citations:
                            bad = cit.validator_status == "failed"
                            evidence = (
                                "<br><span style='color:var(--text-3);'>supports:</span> " +
                                " ".join(f"<span class='chip'>{e}</span>" for e in cit.supporting_evidence)
                            ) if cit.supporting_evidence else ""
                            st.markdown(
                                f"<div class='citation {'bad' if bad else ''}'>"
                                f"<b>{cit.claim}</b>{evidence}"
                                f"<div class='cit-meta'>"
                                f"source: {cit.source_field}[row {cit.source_row}] · "
                                f"extraction_conf={cit.extraction_confidence} · "
                                f"validator=<b>{cit.validator_status}</b>"
                                f"</div></div>",
                                unsafe_allow_html=True,
                            )
                        if f.contradictions:
                            st.markdown(
                                "<div style='margin-top:10px;'>"
                                "<span class='chip bad'>VALIDATOR FLAGS</span><br>"
                                + "".join(f"<div class='citation bad' style='margin-top:6px;'>{c}</div>"
                                          for c in f.contradictions) +
                                "</div>",
                                unsafe_allow_html=True,
                            )

# ──────────────────────────────────────────────────────────────────────────────
# Tab 2 — Medical Desert Map
# ──────────────────────────────────────────────────────────────────────────────

with tab_desert:
    section_header(
        "02 · Medical Desert Map",
        "Where the gaps are",
        "City-level facility coverage for high-acuity needs. Bubble size = total facilities, "
        "colour = % of those facilities offering the selected specialty. 95% confidence intervals "
        "are Wilson scores on the specialty-share proportion."
    )

    ctrl_a, ctrl_b, ctrl_c = st.columns([0.30, 0.30, 0.40])
    specialty = ctrl_a.selectbox("Specialty", list(_DESERT_KEYWORDS.keys()), index=0)
    min_fac = ctrl_b.slider("Min facilities in city", 1, 200, 5, step=1)
    show_table = ctrl_c.toggle("Show CI plot alongside table", value=True)

    df = load_district_desert_data(specialty)
    if df.empty:
        st.info("No data available — verify WAREHOUSE_ID and that gold_facilities is populated.")
    else:
        df_view = df[df["n_facilities"] >= min_fac].copy()
        df_view["bubble_size"] = df_view["n_facilities"].clip(lower=1)

        n_districts = len(df_view)
        total_fac     = int(df_view["n_facilities"].sum())
        total_spec    = int(df_view["n_specialty"].sum())
        agg_coverage  = (total_spec / total_fac * 100) if total_fac > 0 else 0.0
        worst         = df_view.iloc[0] if len(df_view) else None

        render_kpi_row([
            {"label": "Cities in view",   "value": f"{n_districts}",
             "delta": f"≥{min_fac} facilities", "tone": None},
            {"label": "Aggregate coverage", "value": f"{agg_coverage:.1f}%",
             "delta": f"{total_spec:,} of {total_fac:,} facilities", "tone": "good"},
            {"label": "Worst deficit",    "value": worst['district'] if worst is not None else "—",
             "delta": f"score {worst['deficit_score']:.2f}" if worst is not None else "",
             "tone": "bad"},
            {"label": "Total facilities", "value": f"{total_fac:,}",
             "delta": "in selected cities", "tone": None, "brand": True,
             "sparkline": [0.6, 0.55, 0.65, 0.7, 0.75, 0.82, 0.9]},
        ])

        st.markdown(" ")

        if df_view.empty:
            st.info("No cities match the current filter — lower the minimum facility threshold.")
        else:
            max_cov = float(df_view["coverage_pct"].max() or 1.0)
            fig = px.scatter_mapbox(
                df_view, lat="lat", lon="lon", size="bubble_size",
                color="coverage_pct",
                color_continuous_scale=[(0, "#a8412a"), (0.5, "#b8843c"), (1.0, "#2d4a3a")],
                range_color=(0, max_cov),
                hover_name="district",
                hover_data={"state": True, "n_facilities": True, "n_specialty": True,
                            "coverage_pct": True, "ci_low": True, "ci_high": True,
                            "deficit_score": True, "lat": False, "lon": False, "bubble_size": False},
                size_max=46, zoom=3.7, height=540,
            )
            fig.update_layout(
                mapbox_style="carto-positron",
                margin={"l": 0, "r": 0, "t": 0, "b": 0},
                coloraxis_colorbar=dict(title=f"{specialty} share %", thickness=10, len=0.5,
                                        x=0.99, xanchor="right"),
                paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(" ")
            cA, cB = st.columns([0.55, 0.45], gap="large") if show_table else (st.container(), None)

            with cA:
                st.markdown(
                    f"<p class='section-eyebrow'>Worst 10 — {specialty.lower()} deficit</p>",
                    unsafe_allow_html=True)
                bottom = df_view.head(10)[
                    ["district", "state", "n_facilities", "n_specialty",
                     "coverage_pct", "ci_low", "ci_high", "deficit_score"]
                ]
                st.dataframe(
                    bottom,
                    column_config={
                        "district":      st.column_config.TextColumn("city", width="medium"),
                        "state":         st.column_config.TextColumn("state"),
                        "n_facilities":  st.column_config.NumberColumn("facilities"),
                        "n_specialty":   st.column_config.NumberColumn("with specialty"),
                        "coverage_pct":  st.column_config.NumberColumn("coverage %", format="%.2f"),
                        "ci_low":        st.column_config.NumberColumn("CI low %",  format="%.2f"),
                        "ci_high":       st.column_config.NumberColumn("CI high %", format="%.2f"),
                        "deficit_score": st.column_config.ProgressColumn(
                            "deficit score", min_value=0.0,
                            max_value=float(df_view["deficit_score"].max() or 1.0),
                            format="%.2f",
                        ),
                    },
                    use_container_width=True, hide_index=True, height=420,
                )

            if show_table and cB is not None:
                with cB:
                    st.markdown("<p class='section-eyebrow'>95% Wilson CI · top 10</p>", unsafe_allow_html=True)
                    agg = df_view.head(10).copy()
                    ci_fig = go.Figure()
                    ci_fig.add_trace(go.Scatter(
                        x=agg["coverage_pct"], y=agg["district"], mode="markers",
                        marker=dict(color="#2d4a3a", size=11),
                        error_x=dict(
                            type="data", symmetric=False,
                            array=agg["ci_high"] - agg["coverage_pct"],
                            arrayminus=agg["coverage_pct"] - agg["ci_low"],
                            color="#8f8475", thickness=1.2, width=4,
                        ),
                        hovertemplate="%{y}<br>%{x:.2f}%% coverage<extra></extra>",
                    ))
                    ci_fig.update_layout(
                        height=420, plot_bgcolor="white", paper_bgcolor="white",
                        xaxis_title=f"{specialty} share % (95% CI)",
                        yaxis=dict(autorange="reversed"),
                        margin={"l": 8, "r": 8, "t": 8, "b": 8},
                        font=dict(family="Inter", size=12, color="#1c1815"),
                    )
                    ci_fig.update_xaxes(showgrid=True, gridcolor="#f0e9d6", zeroline=False)
                    ci_fig.update_yaxes(showgrid=False)
                    st.plotly_chart(ci_fig, use_container_width=True)

    st.markdown(
        '<div class="footnote">'
        '<strong>Methodology</strong> · Per-city specialty share = facilities offering the selected '
        'capability ÷ total facilities in that city. The 95% interval is a Wilson score on that '
        'binomial proportion (more honest than normal approximation at the tails). Deficit score = '
        '(1 − share) × n_facilities, surfacing dense cities with weak coverage. Data source: '
        f'<code>{NS}.gold_facilities</code>.'
        '</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3 — Trust Audit
# ──────────────────────────────────────────────────────────────────────────────

with tab_audit:
    st.header("We audited our own data")
    st.caption(
        "The agent doesn't just trust the input — it scores every row against 8 rules and "
        "exposes the breakdown. Coordinate issues are surfaced as R3 flags when a point falls "
        "outside India or outside the claimed state."
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
    coord_flags = run_sql(f"""
        SELECT COUNT(*) AS n_total,
               SUM(CASE WHEN exists(flags, f -> f.rule_id = 'R3') THEN 1 ELSE 0 END) AS n_coord_flags
        FROM {NS}.gold_facilities
    """)
    rc = {r["severity"]: int(r["n"]) for _, r in rule_counts.iterrows()} if not rule_counts.empty else {}
    if coord_flags.empty:
        n_total, n_coord = 0, 0
    else:
        n_total = int(float(coord_flags.iloc[0].get("n_total") or 0))
        n_coord = int(float(coord_flags.iloc[0].get("n_coord_flags") or 0))
    coord_pct = (100.0 * n_coord / n_total) if n_total else 0.0

    m1.metric("'farmacy' typo entries", int(farmacy.iloc[0]["n"]) if not farmacy.empty else 0)
    m2.metric("HIGH-severity contradictions", rc.get("HIGH", 0))
    m3.metric(
        "Coordinate/state flags", f"{coord_pct:.2f}%",
        f"{n_coord:,} facilities flagged by R3",
        delta_color="inverse",
    )
