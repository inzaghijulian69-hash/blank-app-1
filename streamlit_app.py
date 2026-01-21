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
# JUDUL
# =========================
st.title("✨ Newton-Raphson SPNL ✨")
st.markdown("### 🚀 Sistem Persamaan Non Linier (2 Variabel)")

# =========================
# INPUT PERSAMAAN
# =========================
f1_input = st.text_input("📌 Masukkan f1(x, y)", "x**2 + y**2 - 4")
f2_input = st.text_input("📌 Masukkan f2(x, y)", "x - y - 1")

# =========================
# INPUT PARAMETER
# =========================
col1, col2 = st.columns(2)
with col1:
    x0 = st.number_input("🔢 Tebakan awal x₀", value=1.0)
with col2:
    y0 = st.number_input("🔢 Tebakan awal y₀", value=1.0)

maks_iterasi = st.number_input("🔁 Maksimum Iterasi", value=10)
toleransi = st.number_input("🎯 Toleransi Error", value=0.0001)

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

        data_iterasi = []
        error_list = []

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
        # TABEL ITERASI
        # =========================
        st.subheader("📋 Tabel Iterasi Newton-Raphson")
        df = pd.DataFrame(
            data_iterasi,
            columns=["Iterasi", "x", "y", "Error"]
        )
        st.dataframe(df)

        # =========================
        # HASIL
        # =========================
        st.success(
            f"🎯 Solusi Konvergen!\n\n"
            f"x = {xn[0]:.6f}\n"
            f"y = {xn[1]:.6f}"
        )

        # =========================
        # GRAFIK KONVERGENSI
        # =========================
        st.subheader("📊 Grafik Konvergensi Error")

        fig, ax = plt.subplots()
        it = range(1, len(error_list) + 1)

        ax.plot(it, error_list, color=PRIMARY, linewidth=2, label="Error")
        ax.scatter(it, error_list, color="#facc15", s=60, label="Iterasi")
        ax.scatter(it[-1], error_list[-1], color="#ef4444", s=120, label="Konvergen ❤️")

        ax.set_xlabel("Iterasi")
        ax.set_ylabel("Error")
        ax.set_title("📈 Konvergensi Newton-Raphson")

        ax.grid(True, linestyle="--", alpha=0.6, color=GRID_COLOR)
        ax.legend()

        st.pyplot(fig)

    except Exception as e:
        st.error("❌ Terjadi kesalahan dalam perhitungan")
        st.write(e)

