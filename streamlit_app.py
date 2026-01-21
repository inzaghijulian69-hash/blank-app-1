import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# =========================
# CSS CUSTOM
# =========================
st.markdown("""
<style>
h1, h2, h3 {
    color: #2563EB;
}

.stButton > button {
    background-color: #2563EB;
    color: white;
    border-radius: 8px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #1D4ED8;
}

</style>
""", unsafe_allow_html=True)

# =========================
# JUDUL
# =========================
st.title("Newton-Raphson SPNL")
st.markdown("### Sistem Persamaan Non Linier (2 Variabel)")

# =========================
# INPUT
# =========================
f1_input = st.text_input("Masukkan f1(x, y)", "x**2 + y**2 - 4")
f2_input = st.text_input("Masukkan f2(x, y)", "x - y - 1")

col1, col2 = st.columns(2)
with col1:
    x0 = st.number_input("Tebakan awal x₀", value=1.0)
with col2:
    y0 = st.number_input("Tebakan awal y₀", value=1.0)

maks_iterasi = st.number_input("Maksimum Iterasi", value=10)
toleransi = st.number_input("Toleransi Error", value=0.0001)

# =========================
# PROSES
# =========================
if st.button("Hitung"):
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
        df = pd.DataFrame(
            data_iterasi,
            columns=["Iterasi", "x", "y", "Error"]
        )
        st.subheader("Tabel Iterasi")
        st.dataframe(df)

        st.success(f"Solusi Konvergen: x = {xn[0]:.6f}, y = {xn[1]:.6f}")

        # =========================
        # GRAFIK VARIATIF
        # =========================
        st.subheader("Grafik Konvergensi Error")

        fig, ax = plt.subplots()

        iterations = range(1, len(error_list) + 1)

        # Garis utama
        ax.plot(
            iterations,
            error_list,
            linestyle='-',
            linewidth=2,
            color='#2563EB',
            label='Error'
        )

        # Titik iterasi (warna berbeda)
        ax.scatter(
            iterations,
            error_list,
            color='#F59E0B',
            s=60,
            zorder=5,
            label='Iterasi'
        )

        # Highlight iterasi terakhir
        ax.scatter(
            iterations[-1],
            error_list[-1],
            color='#DC2626',
            s=100,
            label='Konvergen'
        )

        ax.set_xlabel("Iterasi")
        ax.set_ylabel("Error")
        ax.set_title("Konvergensi Metode Newton-Raphson")

        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

        st.pyplot(fig)

    except Exception as e:
        st.error("Terjadi kesalahan dalam perhitungan")
        st.write(e)
