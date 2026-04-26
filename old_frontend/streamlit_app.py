"""
PramanaCare.ai — Confidence-calibrated healthcare intelligence for India.

Streamlit UI shell. All data is mocked; same function signatures will be
wired to Delta tables and the agent layer on day 5.

Run:
    pip install -r app/requirements.txt
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
# Logo (inline SVG — three nodes in a triangle, the three Pramanas, with one
# highlighted node and connecting graph edges; doubles as a network motif)
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
  <!-- triangle edges -->
  <path d="M14 34 L34 34 L24 14 Z" fill="none" stroke="#f3e8c8" stroke-width="1.6" stroke-linejoin="round" opacity="0.55"/>
  <!-- inner edges to centroid -->
  <path d="M24 27 L14 34 M24 27 L34 34 M24 27 L24 14" stroke="#f3e8c8" stroke-width="1" opacity="0.35"/>
  <!-- nodes -->
  <circle cx="14" cy="34" r="3.2" fill="#f7f1e3"/>
  <circle cx="34" cy="34" r="3.2" fill="#f7f1e3"/>
  <circle cx="24" cy="14" r="3.2" fill="#f7f1e3"/>
  <!-- centroid (highlighted) -->
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
# CSS — cream/sage/terracotta system
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

    /* ───── TYPOGRAPHY ───── */
    h1, h2, h3, h4 {
        font-family: 'Fraunces', 'Inter', serif !important;
        font-weight: 500 !important;
        letter-spacing: -0.025em !important;
        color: var(--text);
    }
    h1 { font-size: 56px !important; line-height: 1.04; }
    h2 { font-size: 32px !important; line-height: 1.15; margin: 0; }
    h3 { font-size: 20px !important; line-height: 1.3; margin: 0; }

    /* ───── TOP NAV ───── */
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

    /* ───── HERO ───── */
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

    /* ───── STATS STRIP ───── */
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

    /* ───── HOW IT WORKS ───── */
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

    /* ───── DEMO CARDS ───── */
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

    /* ───── TECH CHIP STRIP ───── */
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

    /* ───── TABS ───── */
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

    /* ───── SECTION HEADER ───── */
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

    /* ───── CARDS ───── */
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

    /* ───── KPI BENTO ───── */
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

    /* ───── TRUST BADGES ───── */
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

    /* ───── CITATION ───── */
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

    /* ───── AGENT STEPS ───── */
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

    /* ───── CHIPS ───── */
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

    /* ───── EMPTY ───── */
    .empty {
        background: var(--surface);
        border: 1px dashed var(--border-strong);
        border-radius: var(--radius);
        padding: 60px 32px;
        text-align: center;
    }
    .empty-title { font-family: 'Fraunces', serif; font-size: 22px; color: var(--text); margin-bottom: 6px; }
    .empty-tag   { font-size: 14px; color: var(--text-2); }

    /* ───── BUTTONS ───── */
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

    /* ───── CHAT ───── */
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

    /* ───── DATAFRAME ───── */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: var(--radius);
        overflow: hidden;
    }

    /* ───── INPUTS ───── */
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
    /* toggle */
    [data-baseweb="checkbox"] [role="checkbox"][aria-checked="true"] { background: var(--brand) !important; border-color: var(--brand) !important; }

    /* ───── PLOT ───── */
    [data-testid="stPlotlyChart"] {
        border-radius: var(--radius);
        overflow: hidden;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border);
        background: var(--surface);
    }

    /* ───── FOOTNOTE ───── */
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

    /* ───── GLOBAL FOOTER ───── */
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

    /* ───── FACILITY ROW ───── */
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

    /* ───── EXPANDER ───── */
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

    /* ───── SCROLLBAR ───── */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 6px; }
    ::-webkit-scrollbar-thumb:hover { background: #c4b899; }

    /* ───── SIDEBAR ───── */
    [data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
    [data-testid="stSidebar"] .stMarkdown { font-size: 13px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Mock data
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
    facility_id: str; name: str; city: str; state: str; pin: str
    lat: float; lon: float; facility_type: str
    specialties: list[str]; capabilities: list[str]; equipment: list[str]
    trust_score: float; trust_components: dict[str, float]
    contradictions: list[str]; citations: list[Citation]


def _mock_facilities() -> list[Facility]:
    rng = random.Random(42)
    seeds = [
        ("Sanjeevani Multi-Speciality Hospital", "Mumbai", "Maharashtra", "400050", 19.0760, 72.8777, "hospital",
         ["cardiology", "oncology", "nephrology", "generalSurgery"], False),
        ("Apollo Health City", "Hyderabad", "Telangana", "500033", 17.4239, 78.4738, "hospital",
         ["cardiology", "neurology", "transplantSurgery", "oncology"], False),
        ("Lifeline Critical Care", "Patna", "Bihar", "800001", 25.5941, 85.1376, "hospital",
         ["criticalCare", "trauma", "nephrology"], False),
        ("Gomti Nagar Dialysis Centre", "Lucknow", "Uttar Pradesh", "226010", 26.8467, 80.9462, "clinic",
         ["nephrology", "dialysis"], False),
        ("Sant Tukaram Rural Hospital", "Aurangabad", "Uttar Pradesh", "431001", 19.8762, 75.3433, "hospital",
         ["generalMedicine", "obstetrics"], True),
        ("Coastal Cancer Institute", "Visakhapatnam", "Andhra Pradesh", "530002", 17.6868, 83.2185, "hospital",
         ["oncology", "radiationOncology", "palliativeCare"], False),
        ("Kerala Backwaters Clinic", "Alappuzha", "Kerala", "688001", 9.4981, 76.3388, "clinic",
         ["familyMedicine", "ayurveda"], False),
        ("Rajasthan Heart Foundation", "Jaipur", "Rajasthan", "302017", 26.9124, 75.7873, "hospital",
         ["cardiology", "interventionalCardiology"], False),
        ("Sundarbans Mobile Health Unit", "Canning", "West Bengal", "743329", 22.3208, 88.6635, "clinic",
         ["familyMedicine", "obstetrics", "pediatrics"], False),
        ("Aravalli District Hospital", "Udaipur", "Rajasthan", "313001", 24.5854, 73.7125, "hospital",
         ["generalSurgery", "orthopedics", "trauma"], False),
        ("MetroLab Diagnostics", "Pune", "Maharashtra", "411014", 18.5204, 73.8567, "clinic",
         ["pathology", "radiology"], False),
        ("Vidarbha Trauma Centre", "Nagpur", "Maharashtra", "440010", 21.1458, 79.0882, "hospital",
         ["trauma", "neurosurgery", "criticalCare"], False),
        ("Brahmaputra Riverside Clinic", "Dibrugarh", "Assam", "786001", 27.4728, 94.9120, "clinic",
         ["familyMedicine", "tropicalMedicine"], False),
        ("Saraswati Maternity Home", "Indore", "Madhya Pradesh", "452001", 22.7196, 75.8577, "hospital",
         ["obstetrics", "neonatology"], False),
        ("Konkan Coastal Hospital", "Ratnagiri", "Maharashtra", "415612", 16.9944, 73.3000, "hospital",
         ["familyMedicine", "generalSurgery"], False),
        ("Tamil Nadu Eye Bank", "Madurai", "Tamil Nadu", "625001", 9.9252, 78.1198, "clinic",
         ["ophthalmology"], False),
        ("Northeast Cancer Wing — GMC", "Guwahati", "Assam", "781032", 26.1445, 91.7362, "hospital",
         ["oncology", "hematology"], False),
        ("Bundelkhand Rural Health Post", "Jhansi", "Uttar Pradesh", "284001", 25.4484, 78.5685, "clinic",
         ["familyMedicine"], False),
        ("Goa Coastal Diagnostics", "Panaji", "Goa", "403001", 15.4909, 73.8278, "clinic",
         ["radiology", "pathology", "cardiology"], False),
        ("Chambal Trauma Response", "Gwalior", "Madhya Pradesh", "474001", 26.2183, 78.1828, "hospital",
         ["trauma", "orthopedics"], False),
        ("Kashi Vishwanath Multi-Speciality", "Varanasi", "Uttar Pradesh", "221002", 25.3176, 82.9739, "hospital",
         ["cardiology", "oncology", "generalSurgery"], False),
        ("Mahanadi Dialysis & Kidney Centre", "Cuttack", "Odisha", "753001", 20.4625, 85.8828, "clinic",
         ["nephrology", "dialysis"], False),
        ("Punjab Heartline Cardiac Care", "Ludhiana", "Punjab", "141001", 30.9010, 75.8573, "hospital",
         ["cardiology", "cardiothoracicSurgery"], False),
        ("Karnataka Bone & Joint Centre", "Mysuru", "Karnataka", "570020", 12.2958, 76.6394, "hospital",
         ["orthopedics", "rheumatology"], False),
        ("Saharanpur Family Medicine Hub", "Saharanpur", "Uttar Pradesh", "247001", 29.9680, 77.5552, "clinic",
         ["familyMedicine"], False),
    ]

    catalog = {
        "cardiology":               (["Cardiac monitor", "ECG machine", "Echo Doppler"], ["Cardiologist on-call 24x7", "Performs angioplasty"]),
        "interventionalCardiology": (["Cath lab", "C-arm imaging"], ["Performs angioplasty", "Stent placement available"]),
        "cardiothoracicSurgery":    (["Heart-lung machine", "Operating microscope"], ["Performs CABG", "Valve replacement surgery"]),
        "oncology":                 (["Linear accelerator", "Chemotherapy infusion suite"], ["Day-care chemotherapy", "Tumor board weekly"]),
        "radiationOncology":        (["Linear accelerator", "Brachytherapy unit", "CT simulator"], ["External beam radiation therapy"]),
        "palliativeCare":           (["Syringe driver pumps"], ["Home-based palliative care", "Pain management clinic"]),
        "hematology":               (["Cell counter", "Flow cytometer"], ["Bone marrow biopsy", "Anemia clinic"]),
        "nephrology":               (["Hemodialysis machine x4", "Reverse osmosis water plant"], ["Performs hemodialysis sessions", "CKD outpatient clinic"]),
        "dialysis":                 (["Hemodialysis machine x4", "Peritoneal dialysis kits", "Reverse osmosis water plant"], ["Performs hemodialysis sessions", "Night-shift dialysis available"]),
        "transplantSurgery":        (["Operating microscope", "ICU recovery suite"], ["Renal transplant program", "Liver transplant — referral pending"]),
        "neurology":                (["EEG machine", "MRI scanner"], ["Stroke unit", "Epilepsy clinic"]),
        "neurosurgery":             (["Operating microscope", "Neuronavigation system"], ["Performs craniotomy", "Spinal surgery"]),
        "trauma":                   (["X-ray machine", "Defibrillator", "Crash cart"], ["24x7 emergency", "Polytrauma resuscitation"]),
        "criticalCare":             (["Ventilator x2", "ICU bed monitoring", "Defibrillator"], ["6-bed ICU", "Intensivist on-call"]),
        "generalSurgery":           (["Operating room — laminar flow", "Anaesthesia machine", "Autoclave"], ["Performs appendectomy", "Cholecystectomy", "Hernia repair"]),
        "orthopedics":              (["C-arm imaging", "Bone drill set"], ["Joint replacement", "Fracture clinic"]),
        "rheumatology":             ([], ["Arthritis day clinic"]),
        "obstetrics":               (["Ultrasound — obstetric", "Foetal monitor", "Delivery table"], ["24x7 delivery services", "C-section capable"]),
        "neonatology":              (["Neonatal warmer x2", "CPAP machine"], ["Level 2 neonatal care"]),
        "pediatrics":               (["Pediatric resuscitation kit"], ["Routine immunization", "Pediatric OPD"]),
        "ophthalmology":            (["Slit lamp", "Phaco machine"], ["Cataract surgery", "Glaucoma clinic"]),
        "familyMedicine":           ([], ["General OPD", "Routine immunization"]),
        "generalMedicine":          (["BP monitor", "Glucometer"], ["General OPD"]),
        "tropicalMedicine":         ([], ["Malaria diagnostics", "Dengue management"]),
        "ayurveda":                 ([], ["Panchakarma therapy"]),
        "pathology":                (["Cell counter", "Microscope — binocular"], ["Routine bloodwork", "Histopathology reporting"]),
        "radiology":                (["X-ray machine", "Ultrasound scanner"], ["Radiology reporting <24h"]),
    }

    facilities: list[Facility] = []
    for i, (name, city, state, pin, lat, lon, ftype, spec_pool, quality_issue) in enumerate(seeds):
        equipment, capabilities = [], []
        for s in spec_pool:
            eq, caps = catalog.get(s, ([], []))
            equipment.extend(eq); capabilities.extend(caps)

        deliberately_advanced_no_eq = (i % 7 == 3)
        if deliberately_advanced_no_eq:
            capabilities.append("Advanced surgery available"); equipment = []

        contradictions = []
        if deliberately_advanced_no_eq:
            contradictions.append("Claims 'Advanced surgery' but equipment array is empty")
        if quality_issue:
            contradictions.append(f"Geographic inconsistency: {city} is in Maharashtra, listed as {state}")
        if "trauma" in spec_pool and equipment and "Defibrillator" not in equipment:
            contradictions.append("Claims trauma capability without defibrillator on equipment list")

        claim_density = min(1.0, (len(capabilities) + len(equipment)) / 20)
        evidence_agreement = 0.95 if equipment else 0.30
        contradiction_penalty = 1.0 - 0.18 * len(contradictions)
        location_coherence = 0.45 if quality_issue else 0.95
        structured_fillrate = rng.uniform(0.4, 0.85)

        trust = max(0.05, min(1.0,
            0.25 * claim_density + 0.30 * evidence_agreement +
            0.20 * contradiction_penalty + 0.15 * location_coherence +
            0.10 * structured_fillrate))

        citations = [
            Citation(
                claim=cap, source_field="capability",
                source_row=1000 + i * 17 + j,
                extraction_confidence=round(rng.uniform(0.78, 0.99), 2),
                validator_status="failed" if (deliberately_advanced_no_eq and "Advanced surgery" in cap) else "passed",
                supporting_evidence=[e for e in equipment[:2]] if equipment else [],
            )
            for j, cap in enumerate(capabilities[:5])
        ]

        facilities.append(Facility(
            facility_id=f"VF{i:05d}", name=name, city=city, state=state, pin=pin,
            lat=lat, lon=lon, facility_type=ftype,
            specialties=spec_pool, capabilities=capabilities, equipment=equipment,
            trust_score=round(trust, 2),
            trust_components={
                "claim_density": round(claim_density, 2),
                "evidence_agreement": round(evidence_agreement, 2),
                "contradiction_penalty": round(contradiction_penalty, 2),
                "location_coherence": round(location_coherence, 2),
                "structured_fillrate": round(structured_fillrate, 2),
            },
            contradictions=contradictions, citations=citations,
        ))
    return facilities


@st.cache_data(show_spinner=False)
def load_facilities() -> list[Facility]:
    return _mock_facilities()


@st.cache_data(show_spinner=False)
def load_district_desert_data(specialty: str) -> pd.DataFrame:
    rng = random.Random(hash(specialty) & 0xFFFFFFFF)
    cities = [
        ("Mumbai", "Maharashtra", 19.0760, 72.8777, 12_478_000),
        ("Delhi", "Delhi", 28.7041, 77.1025, 16_787_941),
        ("Bangalore", "Karnataka", 12.9716, 77.5946, 8_443_675),
        ("Chennai", "Tamil Nadu", 13.0827, 80.2707, 4_646_732),
        ("Kolkata", "West Bengal", 22.5726, 88.3639, 4_496_694),
        ("Hyderabad", "Telangana", 17.3850, 78.4867, 6_809_970),
        ("Pune", "Maharashtra", 18.5204, 73.8567, 3_115_431),
        ("Ahmedabad", "Gujarat", 23.0225, 72.5714, 5_577_940),
        ("Jaipur", "Rajasthan", 26.9124, 75.7873, 3_073_350),
        ("Lucknow", "Uttar Pradesh", 26.8467, 80.9462, 2_815_601),
        ("Patna", "Bihar", 25.5941, 85.1376, 1_684_222),
        ("Bhopal", "Madhya Pradesh", 23.2599, 77.4126, 1_798_218),
        ("Visakhapatnam", "Andhra Pradesh", 17.6868, 83.2185, 1_730_320),
        ("Nagpur", "Maharashtra", 21.1458, 79.0882, 2_405_421),
        ("Indore", "Madhya Pradesh", 22.7196, 75.8577, 1_960_631),
        ("Kanpur", "Uttar Pradesh", 26.4499, 80.3319, 2_767_031),
        ("Coimbatore", "Tamil Nadu", 11.0168, 76.9558, 1_050_721),
        ("Cuttack", "Odisha", 20.4625, 85.8828, 663_849),
        ("Gaya", "Bihar", 24.7914, 85.0002, 463_454),
        ("Muzaffarpur", "Bihar", 26.1209, 85.3647, 393_724),
        ("Gorakhpur", "Uttar Pradesh", 26.7606, 83.3732, 671_048),
        ("Aurangabad", "Maharashtra", 19.8762, 75.3433, 1_175_116),
        ("Ranchi", "Jharkhand", 23.3441, 85.3096, 1_073_440),
        ("Jamshedpur", "Jharkhand", 22.8046, 86.2029, 629_659),
        ("Jhansi", "Uttar Pradesh", 25.4484, 78.5685, 547_638),
        ("Bareilly", "Uttar Pradesh", 28.3670, 79.4304, 898_167),
        ("Aligarh", "Uttar Pradesh", 27.8974, 78.0880, 872_575),
        ("Saharanpur", "Uttar Pradesh", 29.9680, 77.5552, 703_345),
        ("Dibrugarh", "Assam", 27.4728, 94.9120, 154_019),
        ("Alappuzha", "Kerala", 9.4981, 76.3388, 174_164),
    ]
    rate_floor, rate_ceiling = {
        "Dialysis": (0.05, 1.8), "Oncology": (0.10, 4.5),
        "Trauma":   (0.20, 6.0), "ICU":      (0.30, 5.0),
        "Cardiac":  (0.15, 5.5),
    }[specialty]

    rows = []
    for city, state, lat, lon, pop in cities:
        urban_factor = 1.6 if pop > 2_000_000 else (1.0 if pop > 1_000_000 else 0.55)
        per100k = max(0.0, rng.uniform(rate_floor, rate_ceiling) * urban_factor)
        n_facilities = max(0, round(per100k * pop / 100_000))
        ci_half = max(0.05, per100k * rng.uniform(0.18, 0.32))
        rows.append({
            "district": city, "state": state, "lat": lat, "lon": lon,
            "population": pop, "facilities": n_facilities,
            "per_100k": round(per100k, 2),
            "ci_low": round(max(0, per100k - ci_half), 2),
            "ci_high": round(per100k + ci_half, 2),
            "deficit_score": round(max(0.0, rate_ceiling - per100k) * (pop / 1_000_000), 2),
        })
    return pd.DataFrame(rows).sort_values("deficit_score", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
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
# Mock agent
# ──────────────────────────────────────────────────────────────────────────────

QUERY_PRESETS = {
    "I need dialysis near 110001 (Delhi)":     ("dialysis", 28.6139, 77.2090),
    "Find a hospital that can perform an emergency appendectomy in rural Bihar":
                                                ("generalSurgery", 25.5941, 85.1376),
    "Where can a stage-3 lung cancer patient get radiation in Andhra Pradesh?":
                                                ("radiationOncology", 17.6868, 83.2185),
    "Closest 24x7 trauma centre to Aurangabad": ("trauma", 19.8762, 75.3433),
    "Cardiac angioplasty within 50 km of Jaipur":
                                                ("interventionalCardiology", 26.9124, 75.7873),
    "Maternity care in the Sundarbans delta":   ("obstetrics", 22.3208, 88.6635),
}


def mock_agent_plan(query, lat, lon, specialty_hint):
    return [
        {"name": "Planner",   "detail": "Decompose query into (specialty, geography, urgency, trust threshold).",
         "output": json.dumps({"specialty": specialty_hint, "anchor": [lat, lon],
                               "urgency": "high" if "emergency" in query.lower() else "routine",
                               "min_trust": 0.45})},
        {"name": "Vector Search", "detail": "Embed query → retrieve top-50 canonical claim matches from Mosaic Vector Search.",
         "output": "50 candidates · top match cosine 0.872"},
        {"name": "Graph Traversal", "detail": "Walk kg_edges HAS_CAPABILITY → PERFORMS_PROCEDURE → HAS_EQUIPMENT.",
         "output": "27 facilities support the requested capability with ≥1 supporting equipment edge"},
        {"name": "Geo Filter", "detail": "Haversine cutoff + state-coherence filter against PIN-state mapping.",
         "output": "Reduced 27 → 12 within 250 km radius"},
        {"name": "Ranker", "detail": "Score = 0.45·trust + 0.35·specialty_match − 0.20·distance_norm.",
         "output": "Ranked 12 candidates · surfaced top 5"},
        {"name": "Validator", "detail": "Run rule-based contradiction checks against returned facilities.",
         "output": "All 5 passed; 1 has equipment-claim mismatch flag attached"},
        {"name": "Synthesizer", "detail": "Compose answer with row-level citations and MLflow trace IDs.",
         "output": "Answer composed · 17 citations grounded · trace_id=tr_8f2a1c"},
    ]


def mock_agent_pick(facilities, specialty, lat, lon, k=5):
    scored = []
    for f in facilities:
        if specialty not in f.specialties and not any(specialty.lower() in c.lower() for c in f.capabilities):
            continue
        d = haversine_km(lat, lon, f.lat, f.lon)
        if d > 1500: continue
        score = 0.45 * f.trust_score + 0.35 * (1.0 if specialty in f.specialties else 0.5) - 0.20 * min(1.0, d/1500)
        scored.append((f, d, score))
    scored.sort(key=lambda x: x[2], reverse=True)
    return [(f, d) for f, d, _ in scored[:k]]


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
    st.caption(f"Build: dev preview · {datetime.now():%Y-%m-%d %H:%M}")


topnav()

tab_home, tab_finder, tab_desert, tab_audit = st.tabs([
    "Home", "Patient Finder", "Medical Desert Map", "Trust & Audit",
])

# ──────────────────────────────────────────────────────────────────────────────
# Tab 0 — Home / Landing
# ──────────────────────────────────────────────────────────────────────────────

with tab_home:
    st.markdown(
        f"""
        <div class="home-hero">
            <div class="hero-eyebrow"><span class="live-dot"></span> Built on Databricks · MIT × Databricks for Good</div>
            <h1 class="hero-headline">
                Healthcare you can <em>trust.</em><br>
                Down to the <span class="accent">claim.</span>
            </h1>
            <p class="hero-sub">
                PramanaCare turns 10,000 messy Indian medical-facility records into a
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
                    <div class="stat-num">10,000</div>
                    <div class="stat-lab">facilities indexed</div>
                </div>
                <div>
                    <div class="stat-num">147,832</div>
                    <div class="stat-lab">claims canonicalized</div>
                </div>
                <div>
                    <div class="stat-num"><span class="accent">1,284</span></div>
                    <div class="stat-lab">contradictions surfaced</div>
                </div>
                <div>
                    <div class="stat-num">0.71</div>
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
                    <div class="desc">10,000 raw records → Delta with PIN preservation, lat/lon validation, dtype casting.</div>
                </div>
                <div class="pipeline-step">
                    <div class="num">02 · SILVER</div>
                    <div class="nm">Canonicalize</div>
                    <div class="desc">JSON arrays explode into a long claims table. Embedding clusters fold 3,665 equipment strings into ~400 canonical IDs. LLM tags only the residual.</div>
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
                    <div class="desc">Patient Finder, Desert Map, Trust Audit. Every aggregate ships with a 95% bootstrap CI. Every recommendation cites the rows.</div>
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
            <p class="sec-tag">Click any tab above — the system is live with mocked data on this build, ready to swap to the Delta gold tables on day 5.</p>
            <div class="demo-grid">
                <div class="demo-card">
                    <div class="ic">""" + LOGO_LIGHT_BG + """</div>
                    <div class="num">Demo 01</div>
                    <h3>The Dialysis Desert</h3>
                    <p>Only 202 of 10,000 facilities mention dialysis. Weight by district population and CKD burden, and entire Indian states show up as red. PramanaCare puts a 95% CI around it.</p>
                    <div class="stat-line">202 / 10,000 facilities · 17 districts · pop 89M underserved</div>
                </div>
                <div class="demo-card">
                    <div class="ic">""" + LOGO_LIGHT_BG + """</div>
                    <div class="num">Demo 02</div>
                    <h3>Trust Scorer in Action</h3>
                    <p>Find every facility that claims "Advanced Surgery" while listing zero equipment. The validator agent flags it; the trust score drops; the chat surfaces it as a citation.</p>
                    <div class="stat-line">1,284 contradictions · 6 categories · validator self-correction loop</div>
                </div>
                <div class="demo-card">
                    <div class="ic">""" + LOGO_LIGHT_BG + """</div>
                    <div class="num">Demo 03</div>
                    <h3>Data-Quality Audit</h3>
                    <p>Aurangabad is in Maharashtra — yet our row 2 lists it under Uttar Pradesh. PramanaCare surfaces these geographic incoherencies as a first-class signal, not a footnote.</p>
                    <div class="stat-line">Location coherence · PIN ↔ district ↔ state · 213 mismatches found</div>
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

    render_kpi_row([
        {"label": "Facilities indexed", "value": "10,000",
         "delta": "across 194 regions", "tone": None, "brand": True,
         "sparkline": [0.2, 0.4, 0.45, 0.6, 0.7, 0.85, 1.0]},
        {"label": "Claims canonicalized", "value": "147,832",
         "delta": "+12.4% via embedding clusters", "tone": "good",
         "sparkline": [0.1, 0.2, 0.25, 0.5, 0.6, 0.75, 0.9]},
        {"label": "Contradictions surfaced", "value": "1,284",
         "delta": "validator self-correction", "tone": "bad",
         "sparkline": [0.3, 0.5, 0.6, 0.55, 0.7, 0.8, 0.92]},
        {"label": "Mean trust score", "value": "0.71",
         "delta": "± 0.04 (95% bootstrap)", "tone": "good",
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
                    for i, step in enumerate(mock_agent_plan(prompt, lat, lon, specialty), start=1):
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
                        time.sleep(0.16)

                    picks = mock_agent_pick(facilities, specialty, lat, lon, k=5)
                    summary = (
                        f"Returned **{len(picks)}** facilities matching `{specialty}` within reach of "
                        f"({lat:.3f}, {lon:.3f}). Ranked by trust × specialty match × inverse distance. "
                        f"Each result is grounded in row-level citations from the canonical claim graph."
                    )
                    st.markdown(summary)

            st.session_state.messages.append({
                "role": "assistant",
                "content": steps_html + f"<div style='margin-top:8px'>{summary}</div>",
            })
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
        "Per-district facility coverage for high-acuity needs, weighted by population. "
        "Bubble size = unmet-need score. 95% confidence intervals are bootstrap (1,000 resamples over facilities within district)."
    )

    ctrl_a, ctrl_b, ctrl_c = st.columns([0.30, 0.30, 0.40])
    specialty = ctrl_a.selectbox("Specialty", ["Dialysis", "Oncology", "Trauma", "ICU", "Cardiac"], index=0)
    min_pop = ctrl_b.slider("Min district population", 100_000, 5_000_000, 500_000, step=100_000)
    show_table = ctrl_c.toggle("Show CI plot alongside table", value=True)

    df = load_district_desert_data(specialty)
    df_view = df[df["population"] >= min_pop].copy()
    df_view["bubble_size"] = df_view["deficit_score"].clip(lower=0.4) + 0.3

    n_districts = len(df_view)
    total_pop = df_view["population"].sum()
    total_fac = df_view["facilities"].sum()
    nat_per100k = (total_fac / total_pop * 100_000) if total_pop > 0 else 0
    worst = df_view.iloc[0] if len(df_view) else None

    render_kpi_row([
        {"label": "Districts in view",   "value": f"{n_districts}",
         "delta": f"≥{min_pop:,} population", "tone": None},
        {"label": "Aggregate per 100k",  "value": f"{nat_per100k:.2f}",
         "delta": "weighted across districts", "tone": "good"},
        {"label": "Worst deficit",       "value": worst['district'] if worst is not None else "—",
         "delta": f"score {worst['deficit_score']:.2f}" if worst is not None else "",
         "tone": "bad"},
        {"label": "Population uncovered", "value": f"{total_pop/1e6:.1f}M",
         "delta": "in selected districts", "tone": None, "brand": True,
         "sparkline": [0.6, 0.55, 0.65, 0.7, 0.75, 0.82, 0.9]},
    ])

    st.markdown(" ")

    fig = px.scatter_mapbox(
        df_view, lat="lat", lon="lon", size="bubble_size",
        color="per_100k",
        color_continuous_scale=[(0, "#a8412a"), (0.5, "#b8843c"), (1.0, "#2d4a3a")],
        range_color=(0, df_view["per_100k"].max() if len(df_view) else 1),
        hover_name="district",
        hover_data={"state": True, "population": ":,", "facilities": True,
                    "per_100k": True, "ci_low": True, "ci_high": True,
                    "deficit_score": True, "lat": False, "lon": False, "bubble_size": False},
        size_max=46, zoom=3.7, height=540,
    )
    fig.update_layout(
        mapbox_style="carto-positron",
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        coloraxis_colorbar=dict(title=f"{specialty} per 100k", thickness=10, len=0.5,
                                x=0.99, xanchor="right"),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(" ")
    cA, cB = st.columns([0.55, 0.45], gap="large") if show_table else (st.container(), None)

    with cA:
        st.markdown(f"<p class='section-eyebrow'>Bottom 10 — population-weighted {specialty.lower()} deficit</p>",
                    unsafe_allow_html=True)
        bottom = df_view.head(10)[
            ["district", "state", "population", "facilities", "per_100k", "ci_low", "ci_high", "deficit_score"]
        ]
        st.dataframe(
            bottom,
            column_config={
                "district":   st.column_config.TextColumn("district", width="medium"),
                "state":      st.column_config.TextColumn("state"),
                "population": st.column_config.NumberColumn("population", format="%d"),
                "facilities": st.column_config.NumberColumn("facilities"),
                "per_100k":   st.column_config.NumberColumn("per 100k", format="%.2f"),
                "ci_low":     st.column_config.NumberColumn("CI low",  format="%.2f"),
                "ci_high":    st.column_config.NumberColumn("CI high", format="%.2f"),
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
            st.markdown("<p class='section-eyebrow'>95% bootstrap CI · top 10</p>", unsafe_allow_html=True)
            agg = df_view.head(10).copy()
            ci_fig = go.Figure()
            ci_fig.add_trace(go.Scatter(
                x=agg["per_100k"], y=agg["district"], mode="markers",
                marker=dict(color="#2d4a3a", size=11),
                error_x=dict(
                    type="data", symmetric=False,
                    array=agg["ci_high"] - agg["per_100k"],
                    arrayminus=agg["per_100k"] - agg["ci_low"],
                    color="#8f8475", thickness=1.2, width=4,
                ),
                hovertemplate="%{y}<br>%{x:.2f} per 100k<extra></extra>",
            ))
            ci_fig.update_layout(
                height=420, plot_bgcolor="white", paper_bgcolor="white",
                xaxis_title=f"{specialty} per 100k (95% CI)",
                yaxis=dict(autorange="reversed"),
                margin={"l": 8, "r": 8, "t": 8, "b": 8},
                font=dict(family="Inter", size=12, color="#1c1815"),
            )
            ci_fig.update_xaxes(showgrid=True, gridcolor="#f0e9d6", zeroline=False)
            ci_fig.update_yaxes(showgrid=False)
            st.plotly_chart(ci_fig, use_container_width=True)

    st.markdown(
        '<div class="footnote">'
        '<strong>Methodology</strong> · Per-district rates are mocked for the UI shell. Real calculation: '
        'count distinct facility_ids per district whose canonical claim graph contains the selected capability '
        'node, divide by district population from Census of India 2011, bootstrap (n=1,000) over facility '
        'resampling within district to get the 95% CI. Deficit score = (national p90 rate − local rate) × '
        'district population / 1e6.'
        '</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3 — Trust Audit
# ──────────────────────────────────────────────────────────────────────────────

with tab_audit:
    facilities = load_facilities()
    audit_df = pd.DataFrame([{
        "facility_id": f.facility_id, "name": f.name, "type": f.facility_type,
        "city": f.city, "state": f.state, "pin": f.pin,
        "trust_score": f.trust_score, "tier": trust_label(f.trust_score),
        "n_capabilities": len(f.capabilities), "n_equipment": len(f.equipment),
        "n_contradictions": len(f.contradictions),
        **{k: v for k, v in f.trust_components.items()},
    } for f in facilities])

    section_header(
        "03 · Trust & Data-Quality Audit",
        "Surface what the data is hiding",
        "Trust is computed from graph topology — claim density, evidence agreement, contradiction flags, "
        "location coherence, and structured-data fill-rate. Below: aggregate signals, then per-facility drill-down."
    )

    total = len(audit_df)
    avg_trust = audit_df["trust_score"].mean()
    n_contrad = (audit_df["n_contradictions"] > 0).sum()
    n_empty_eq = (audit_df["n_equipment"] == 0).sum()

    render_kpi_row([
        {"label": "Facilities audited", "value": f"{total:,}",
         "delta": "of 10,000 in full dataset", "tone": None, "brand": True,
         "sparkline": [0.2, 0.3, 0.45, 0.6, 0.75, 0.88, 1.0]},
        {"label": "Mean trust score", "value": f"{avg_trust:.2f}",
         "delta": "weighted by claim density", "tone": "good",
         "sparkline": [0.55, 0.6, 0.65, 0.68, 0.7, 0.72, 0.75]},
        {"label": "With contradictions", "value": f"{n_contrad}",
         "delta": f"{100*n_contrad/total:.0f}% of audited", "tone": "bad",
         "sparkline": [0.1, 0.15, 0.2, 0.3, 0.4, 0.55, 0.7]},
        {"label": "Empty equipment list", "value": f"{n_empty_eq}",
         "delta": "candidate trust-scorer hits", "tone": "bad"},
    ])

    st.markdown(" ")
    f1, f2, f3, f4 = st.columns([0.25, 0.25, 0.25, 0.25])
    state_filter = f1.selectbox("State", ["All"] + sorted(audit_df["state"].unique().tolist()))
    type_filter  = f2.selectbox("Facility type", ["All"] + sorted(audit_df["type"].unique().tolist()))
    only_contr   = f3.toggle("Only contradictions", value=False)
    sort_by      = f4.selectbox("Sort by", ["trust_score (asc)", "trust_score (desc)",
                                            "n_contradictions (desc)", "n_capabilities (desc)"])

    view = audit_df.copy()
    if state_filter != "All": view = view[view["state"] == state_filter]
    if type_filter  != "All": view = view[view["type"]  == type_filter]
    if only_contr:            view = view[view["n_contradictions"] > 0]

    sort_map = {
        "trust_score (asc)":         ("trust_score", True),
        "trust_score (desc)":        ("trust_score", False),
        "n_contradictions (desc)":   ("n_contradictions", False),
        "n_capabilities (desc)":     ("n_capabilities", False),
    }
    col, asc = sort_map[sort_by]
    view = view.sort_values(col, ascending=asc).reset_index(drop=True)

    left, right = st.columns([0.58, 0.42], gap="large")

    with left:
        st.markdown(f"<p class='section-eyebrow'>Audit table — {len(view)} facilities</p>",
                    unsafe_allow_html=True)
        st.dataframe(
            view[["facility_id", "name", "city", "state", "type",
                  "trust_score", "n_contradictions", "n_capabilities", "n_equipment"]],
            column_config={
                "facility_id": st.column_config.TextColumn("id", width="small"),
                "name":        st.column_config.TextColumn("facility", width="large"),
                "city":        st.column_config.TextColumn("city"),
                "state":       st.column_config.TextColumn("state"),
                "type":        st.column_config.TextColumn("type", width="small"),
                "trust_score": st.column_config.ProgressColumn(
                    "trust", min_value=0.0, max_value=1.0, format="%.2f", width="medium",
                ),
                "n_contradictions": st.column_config.NumberColumn("flags", width="small"),
                "n_capabilities":   st.column_config.NumberColumn("caps", width="small"),
                "n_equipment":      st.column_config.NumberColumn("eqp", width="small"),
            },
            use_container_width=True, hide_index=True, height=520,
        )

    with right:
        st.markdown("<p class='section-eyebrow'>Inspect a facility</p>", unsafe_allow_html=True)
        if not view.empty:
            pick_id = st.selectbox(
                "Facility", view["facility_id"].tolist(),
                format_func=lambda x: f"{x} — {view[view.facility_id == x]['name'].iloc[0]}",
                label_visibility="collapsed",
            )
            f = next(fac for fac in facilities if fac.facility_id == pick_id)
            st.markdown(
                f"<div class='card hoverable'>"
                f"<div class='card-eyebrow'>{f.facility_id} · {f.facility_type}</div>"
                f"<div class='card-title'>{f.name}</div>"
                f"<div class='card-meta'>{f.city}, {f.state} · PIN {f.pin} · ({f.lat:.3f}, {f.lon:.3f})</div>"
                f"<div style='margin-top:10px;'>{trust_badge_html(f.trust_score)}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            tc = f.trust_components
            comp_fig = go.Figure(go.Bar(
                x=[tc["claim_density"], tc["evidence_agreement"], tc["contradiction_penalty"],
                   tc["location_coherence"], tc["structured_fillrate"]],
                y=["Claim density", "Evidence agreement", "Contradiction penalty",
                   "Location coherence", "Structured fill-rate"],
                orientation="h",
                marker=dict(
                    color=[tc["claim_density"], tc["evidence_agreement"], tc["contradiction_penalty"],
                           tc["location_coherence"], tc["structured_fillrate"]],
                    colorscale=[(0, "#a8412a"), (0.5, "#b8843c"), (1.0, "#2d4a3a")],
                    cmin=0, cmax=1, line=dict(width=0),
                ),
                text=[f"{v:.2f}" for v in [tc["claim_density"], tc["evidence_agreement"],
                       tc["contradiction_penalty"], tc["location_coherence"], tc["structured_fillrate"]]],
                textposition="outside",
                hovertemplate="%{y}: %{x:.2f}<extra></extra>",
            ))
            comp_fig.update_layout(
                height=240,
                xaxis=dict(range=[0, 1.15], showgrid=True, gridcolor="#f0e9d6", zeroline=False, showticklabels=False),
                yaxis=dict(autorange="reversed"),
                margin={"l": 8, "r": 8, "t": 8, "b": 8},
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Inter", size=12, color="#1c1815"),
                showlegend=False,
            )
            st.markdown("<p class='section-eyebrow' style='margin-top:14px;'>Trust components</p>",
                        unsafe_allow_html=True)
            st.plotly_chart(comp_fig, use_container_width=True)

            if f.contradictions:
                st.markdown("<p class='section-eyebrow' style='margin-top:14px;'>Validator findings</p>",
                            unsafe_allow_html=True)
                for c in f.contradictions:
                    st.markdown(f"<div class='citation bad'>{c}</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div class='footnote'><strong>Clean.</strong> No validator contradictions on this facility.</div>",
                    unsafe_allow_html=True,
                )

            with st.expander("Top 5 claims with citations"):
                for cit in f.citations:
                    bad = cit.validator_status == "failed"
                    st.markdown(
                        f"<div class='citation {'bad' if bad else ''}'>"
                        f"<b>{cit.claim}</b>"
                        f"<div class='cit-meta'>"
                        f"source: {cit.source_field}[row {cit.source_row}] · "
                        f"conf={cit.extraction_confidence} · validator=<b>{cit.validator_status}</b>"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )

    st.markdown(
        '<div class="footnote">'
        '<strong>Trust formula</strong> · 0.25·claim_density + 0.30·evidence_agreement + '
        '0.20·contradiction_penalty + 0.15·location_coherence + 0.10·structured_fillrate. '
        'Each component is derived from kg_edges topology.'
        '</div>',
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div class="global-footer">
        <div>
            <div class="footer-brand">
                <div style="width:36px;height:36px;">{LOGO_LIGHT_BG}</div>
                <div class="footer-brand-name">PramanaCare<span style="color:#c4623a;">.ai</span></div>
            </div>
            <div class="footer-tag">
                Pramana — सिद्धि — proof, evidence, the means of valid knowledge.
                A confidence-calibrated healthcare knowledge graph for India.
            </div>
        </div>
        <div>
            <div class="col-title">Product</div>
            <ul>
                <li>Patient Finder</li>
                <li>Desert Map</li>
                <li>Trust Audit</li>
                <li>Methodology</li>
            </ul>
        </div>
        <div>
            <div class="col-title">Stack</div>
            <ul>
                <li>Databricks</li>
                <li>Mosaic AI Vector Search</li>
                <li>Agent Bricks</li>
                <li>MLflow 3</li>
            </ul>
        </div>
        <div>
            <div class="col-title">Built for</div>
            <ul>
                <li>Databricks for Good</li>
                <li>MIT Club of Northern California</li>
                <li>MIT Club of Germany</li>
            </ul>
        </div>
        <div class="copy">
            <span><span class="live-dot" style="background:#2d4a3a;"></span> workspace.hack_nation.gold · trace_id=tr_8f2a1c</span>
            <span>© 2026 PramanaCare · {datetime.now():%Y-%m-%d %H:%M:%S}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
