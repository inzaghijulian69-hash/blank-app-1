import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# =========================
# Judul Aplikasi
# =========================
st.title("Newton-Raphson SPNL")
st.markdown("### Sistem Persamaan Non Linier (2 Variabel)")

# =========================
# Input Persamaan
# =========================
f1_input = st.text_input("Masukkan f1(x, y)", "x**2 + y**2 - 4")
f2_input = st.text_input("Masukkan f2(x, y)", "x - y - 1")

# =========================
# Input Parameter
# =========================
col1, col2 = st.columns(2)
with col1:
    x0 = st.number_input("Tebakan awal x₀", value=1.0)
with col2:
    y0 = st.number_input("Tebakan awal y₀", value=1.0)

maks_iterasi = st.number_input("Maksimum Iterasi", value=10)
toleransi = st.number_input("Toleransi Error", value=0.0001)

# =========================
# Tombol Hitung
# =========================
if st.button("Hitung"):
    try:
        # =========================
        # Definisi simbol
        # =========================
        x, y = sp.symbols('x y')

        f1 = sp.sympify(f1_input)
        f2 = sp.sympify(f2_input)

        # =========================
        # Jacobian
        # =========================
        J = sp.Matrix([
            [sp.diff(f1, x), sp.diff(f1, y)],
            [sp.diff(f2, x), sp.diff(f2, y)]
        ])

        F = sp.Matrix([f1, f2])

        # =========================
        # Iterasi Newton-Raphson
        # =========================
        xn = np.array([x0, y0], dtype=float)

        data_iterasi = []
        error_list = []

        for i in range(1, maks_iterasi + 1):
            J_val = np.array(J.subs({x: xn[0], y: xn[1]}), dtype=float)
            F_val = np.array(F.subs({x: xn[0], y: xn[1]}), dtype=float).astype(float).flatten()

            delta = np.linalg.solve(J_val, -F_val)
            xn_new = xn + delta

            error = np.linalg.norm(delta)
            error_list.append(error)

            data_iterasi.append([i, xn_new[0], xn_new[1], error])

            xn = xn_new

            if error < toleransi:
                break

        # =========================
        # Tabel Iterasi
        # =========================
        df = pd.DataFrame(
            data_iterasi,
            columns=["Iterasi", "x", "y", "Error"]
        )

        st.subheader("Tabel Iterasi Newton-Raphson")
        st.dataframe(df)

        # =========================
        # Solusi Akhir
        # =========================
        st.success(f"Solusi Konvergen: x = {xn[0]:.6f}, y = {xn[1]:.6f}")

        # =========================
        # Grafik Konvergensi
        # =========================
        st.subheader("Grafik Konvergensi Error")

        fig, ax = plt.subplots()
        ax.plot(error_list, marker='o')
        ax.set_xlabel("Iterasi")
        ax.set_ylabel("Error")
        ax.set_title("Grafik Konvergensi Newton-Raphson")
        ax.grid(True)

        st.pyplot(fig)

    except Exception as e:
        st.error("Terjadi kesalahan pada perhitungan")
        st.write(e)
