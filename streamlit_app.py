import streamlit as st

st.set_page_config(page_title="Newton Raphson SPNL", layout="wide")

st.sidebar.title("Pengaturan")
iterasi = st.sidebar.slider("Maks Iterasi", 1, 50, 10)
toleransi = st.sidebar.number_input("Toleransi", value=0.0001)

st.title("Aplikasi Newton-Raphson")
st.markdown("### Sistem Persamaan Non Linier")

col1, col2 = st.columns(2)

with col1:
    f1 = st.text_input("f1(x, y)")
    x0 = st.number_input("x0")

with col2:
    f2 = st.text_input("f2(x, y)")
    y0 = st.number_input("y0")

if st.button("Proses"):
    st.info("Menjalankan perhitungan...")
