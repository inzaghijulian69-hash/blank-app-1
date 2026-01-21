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
    GRADIENT = """
        linear-gradient(-45deg,
        #020617,
        #1e1b4b,
        #0f172a,
        #164e63)
    """
    TEXT_MAIN = "#e5e7eb"
    LABEL_COLOR = "#38bdf8"
    TITLE_COLOR = "#22d3ee"
    SUBTITLE_COLOR = "#a5b4fc"
    PRIMARY = "#22d3ee"
    GRID_COLOR = "#475569"
else:
    GRADIENT = """
        linear-gradient(-45deg,
        #dbeafe,
        #ede9fe,
        #ffffff,
        #e0f2fe)
    """
    TEXT_MAIN = "#000000"
    LABEL_COLOR = "#1d4ed8"
    TITLE_COLOR = "#1e3a8a"
    SUBTITLE_COLOR = "#7c3aed"
    PRIMARY = "#2563eb"
    GRID_COLOR = "#cbd5e1"

# =========================
# CSS GRADIENT ANIMASI
# =========================
st.markdown(
    f"""
    <style>
    .stApp {{
        background: {GRADIENT};
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: {TEXT_MAIN};
    }}

    @keyframes gradientBG {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    h1 {{
        color: {TITLE_COLOR};
        font-weight: 900;
        text-align: center;
        letter-spacing: 1px;
    }}

    h2, h3 {{
        color: {SUBTITLE_COLOR};
        font-weight: 700;
    }}

    label {{
        color: {LABEL_COLOR} !important;
        font-weight: 600;
    }}

    input {{
        color: #000000 !important;
        font-weight: 500;
    }}

    .stButton > button {{
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white;
        border-radius: 16px;
        font-weight: bold;
        padding: 0.7rem 1.6rem;
        box-shadow: 0 8px 18px rgba(0,0,0,0.2);
        transition: 0.3s ease;
    }}

    .stButton > button:hover {{
        transform: scale(1.05);
        box-shadow: 0 10px 24px rgba(0,0,0,0.25);
    }}

    .stDataFrame td, .stDataFrame th {{
        color: {TEXT_MAIN};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# JUDUL
# =========================
st.title("✨ Newton–Raphson Method ✨")
st.markdown(
    "<h3 style='text-align:center;'>🔢 Sistem Persamaan Non Linier (SPNL)</h3>",
    unsafe_allow_html=True
)

# =========================
# INPUT
# =========================
st.subheader("📌 Input Persamaan")
f1_input = st.text_input("🔹 f1(x, y)", "x**2 + y**2 - 4")
f2_input = st.text_input("🔹 f2(x, y)", "x - y - 1")

st.subheader("⚙️ Parameter Awal")
col1, col2 = st.columns(2)
with col1:
    x0 = st.number_input("🔸 Nilai awal x₀", value=1.0)
with col2:
    y0 = st.number_input("🔸 Nilai awal y₀", value=1.0)

maks_iterasi = st.number_input("🔁 Maksimum Iterasi", value=10)
toleransi = st.number_input("🎯 Toleransi Error", value=0.0001)

# =========================
# PROSES
# =========================
if st.button("🚀 Hitung Solusi"):
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
        # OUTPUT
        # =========================
        st.subheader("📋 Tabel Iterasi")
        df = pd.DataFrame(data_iterasi, columns=["Iterasi", "x", "y", "Error"])
        st.dataframe(df)

        st.success(
            f"🎉 Solusi Konvergen!\n\n"
            f"✔ x = {xn[0]:.6f}\n"
            f"✔ y = {xn[1]:.6f}"
        )

        st.subheader("📊 Grafik Konvergensi Error")
        fig, ax = plt.subplots()

        it = range(1, len(error_list) + 1)
        ax.plot(it, error_list, color=PRIMARY, linewidth=2)
        ax.scatter(it, error_list, color="#22c55e", s=70)
        ax.scatter(it[-1], error_list[-1], color="#ef4444", s=140)

        ax.set_xlabel("Iterasi")
        ax.set_ylabel("Error")
        ax.grid(True, linestyle="--", alpha=0.6, color=GRID_COLOR)

        st.pyplot(fig)

    except Exception as e:
        st.error("❌ Terjadi kesalahan dalam perhitungan")
        st.write(e)
