
import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# =========================
# PILIH MODE TEMA
# =========================
mode = st.sidebar.selectbox(
    "🌈 Pilih Mode Tampilan",
    ["🌞 Light Mode", "🌙 Dark Mode"]
)

if mode == "🌙 Dark Mode":
    BG_COLOR = "#0f172a"
    TEXT_COLOR = "#e5e7eb"
    PRIMARY = "#22d3ee"
    GRID_COLOR = "#334155"
else:
    BG_COLOR = "#ffffff"
    TEXT_COLOR = "#1f2933"
    PRIMARY = "#2563eb"
    GRID_COLOR = "#e5e7eb"

# =========================
# CSS DINAMIS
# =========================
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BG_COLOR};
        color: {TEXT_COLOR};
    }}

    h1, h2, h3 {{
        color: {PRIMARY};
    }}

    .stButton > button {{
        background-color: {PRIMARY};
        color: white;
        border-radius: 10px;
        font-weight: bold;
    }}

    .stButton > button:hover {{
        opacity: 0.85;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# JUD
