import streamlit as st
import matplotlib.pyplot as plt

st.title("Aplikasi SPNL Newton-Raphson")

f1 = st.text_input("f1(x, y)")
f2 = st.text_input("f2(x, y)")

if st.button("Hitung"):
    error = [1, 0.3, 0.1, 0.01, 0.001]

    fig, ax = plt.subplots()
    ax.plot(error)
    ax.set_xlabel("Iterasi")
    ax.set_ylabel("Error")
    st.pyplot(fig)
