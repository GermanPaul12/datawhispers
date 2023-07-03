from datawhispers import advancedProg as ap
import streamlit as st

st.title("Welcome to the 💻 Advanced Programming Page")

st.write("---")

st.subheader("Use the submodule yourself with:")
st.code("pip install datawhispers", language="bash")
st.code("from datawhispers import advancedProg as ap")

st.write("---")

st.subheader("Show MNIST NUmber from csv-file")

uploaded_file = st.file_uploader(label="upload your csv-file")
if uploaded_file is not None:
    st.write(ap.show_mnist_from_file(uploaded_file))
    