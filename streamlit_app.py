import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# =========================
# PILIH MODE TEMA
# =========================
mode = st.sidebar.selectbox(
    "🎨 Pilih Mode Tampilan",
    ["🌞 Light Mode", "🌙 Dark Mode"]
)

if mode == "🌙 Dark Mode":
    BG_COLOR = "#0f172a"
    CARD_COLOR = "#020617"
    TEXT_COLOR = "#e5e7eb"
    PRIMARY = "#22d3ee"
    GRID_COLOR = "#334155"
    SHADOW = "none"
else:
    BG_COLOR = "#f8fafc"
    CARD_COLOR = "#ffffff"
    TEXT_COLOR = "#1f2937"
    PRIMARY = "#2563eb"
    GRID_COLOR = "#cbd5e1"
    SHADOW = "0 4px 10px rgba(0,0,0,0.08)"

# =========================
# CSS DINAMIS (RAPI & JELAS)
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
        font-weight: 700;
    }}

    /* Card Container */
    .card {{
        background-color: {CARD_COLOR};
        padding: 1.5rem;
        border-radius: 14px;
        box-shadow: {SHADOW};
        margin-bottom: 1.5rem;
    }}

    /* Input label */
    label {{
        font-weight: 600;
        color: {TEXT_COLOR};
    }}

    /* Button */
    .stButton > button {{
        background-color: {PRIMARY};
        color: white;
        border-radius: 10px;
        font-weight: bold;
        padding: 0.6rem 1.2rem;
    }}

    .stButton > button:hover {{
        opacity: 0.9;
    }}

    /* Dataframe */
    .stDataFrame {{
        border-radius: 12px;
        box-shadow: {SHADOW};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# JUDUL
# =========================
st.title("✨ Newton-Raphson SPNL ✨")
st.markdown("### 🚀 Sistem Persamaan Non Linier (2 Variabel)")

# =========================
# INPUT (CARD)
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("📌 Input Persamaan")
f1_input = st.text_input("f1(x, y)", "x**2 + y**2 - 4")
f2_input = st.text_input("f2(x, y)", "x - y - 1")

st.subheader("🔢 Parameter Awal")
col1, col2 = st.columns(2)
with col1:
    x0 = st.number_input("x₀", value=1.0)
with col2:
    y0 = st.number_input("y₀", value=1.0)

maks_iterasi = st.number_input("Maksimum Iterasi", value=10)
toleransi = st.number_input("Toleransi Error", value=0.0001)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# PROSES
# =========================
if st.button("🚀 Hitung Sekarang"):
    try:
        x, y = sp.symbols('x y')
        f1 = sp.sympify(f1_input)
        f2 = sp.sympify(f2_input)

        J = sp.Matrix([
            [sp.diff(f1, x), sp.diff(f1, y)],
            [sp.diff(f2, x), sp.diff(f2, y)]
        ])
        F = sp.Matrix([f1, f2])

        xn = np.array([x0, y0], dtype=float)
        data_iterasi, error_list = [], []

        for i in range(1, maks_iterasi + 1):
            J_val = np.array(J.subs({x: xn[0], y: xn[1]}), dtype=float)
            F_val = np.array(F.subs({x: xn[0], y: xn[1]}), dtype=float).flatten()

            delta = np.linalg.solve(J_val, -F_val)
            xn_new = xn + delta

            error = np.linalg.norm(delta)
            error_list.append(error)
            data_iterasi.append([i, xn_new[0], xn_new[1], error])

            xn = xn_new
            if error < toleransi:
                break

        # =========================
        # OUTPUT (CARD)
        # =========================
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("📋 Tabel Iterasi")
        df = pd.DataFrame(data_iterasi, columns=["Iterasi", "x", "y", "Error"])
        st.dataframe(df)

        st.success(
            f"🎯 Solusi Konvergen\n\n"
            f"x = {xn[0]:.6f}\n"
            f"y = {xn[1]:.6f}"
        )

        st.subheader("📊 Grafik Konvergensi Error")
        fig, ax = plt.subplots()
        it = range(1, len(error_list) + 1)

        ax.plot(it, error_list, color=PRIMARY, linewidth=2)
        ax.scatter(it, error_list, color="#f59e0b", s=60)
        ax.scatter(it[-1], error_list[-1], color="#ef4444", s=120)

        ax.set_xlabel("Iterasi")
        ax.set_ylabel("Error")
        ax.grid(True, linestyle="--", alpha=0.6, color=GRID_COLOR)

        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error("❌ Terjadi kesalahan dalam perhitungan")
        st.write(e)
