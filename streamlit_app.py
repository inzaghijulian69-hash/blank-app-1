import streamlit as st
import pandas as pd

st.title("Newton-Raphson SPNL")

f1 = st.text_input("f1(x, y)")
f2 = st.text_input("f2(x, y)")

if st.button("Hitung"):
    data = {
        "Iterasi": [1, 2, 3],
        "x": [1.0, 1.2, 1.25],
        "y": [2.0, 2.1, 2.12],
        "Error": [0.5, 0.1, 0.01]
    }
    df = pd.DataFrame(data)
    st.table(df)
