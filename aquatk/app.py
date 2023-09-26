import streamlit as st

st.title("AquaTk Runner")

st.sidebar.title("Options")

dir1 = st.sidebar.button("Select directory 1")
dir2 = st.sidebar.button("Select directory 2")

metrics = st.sidebar.multiselect("Select metrics", ["KL Divergence", "MSE", "MAE", "FAD", "KD"])
embeddings = st.sidebar.multiselect("Select neural embeddings", ["PANNs", "OpenL3", "VGGish", "JukeMIR"])


