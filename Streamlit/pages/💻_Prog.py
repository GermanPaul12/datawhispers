from datawhispers import advancedProg as ap
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.title("Welcome to the 💻 Advanced Programming Page")

st.write("---")

st.subheader("Use the submodule yourself with:")
st.code("""
    # Command prompt
    pip install datawhispers""", language="bash")
st.code("""
    # Python 3
    from datawhispers import advancedProg as ap""")

st.write("---")

st.subheader("Show MNIST Numbers from csv-file")

with st.expander("MNIST"):
    try:
        uploaded_file = st.file_uploader("Choose a CSV file")
        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            data = uploaded_file.getvalue().decode('utf-8').splitlines()   
            label=""
            for i in range(len(data)):
                if "," in data[i]:   
                    arr = np.array([int(num) for num in data[i].split(",")])
                    if arr.shape == (785,): label,arr = arr[0],arr[1:].reshape((28, 28))
                    if arr.shape == (784,): arr = arr.reshape((28, 28))
                    st.write(f"MNIST Number {label}") 
                    st.image(arr, width=100)   
                else:    
                    arr = np.array([int(num) for num in data[i].split(";")])
                    if arr.shape == (785,): label,arr = arr[0],arr[1:].reshape((28, 28))
                    if arr.shape == (784,): arr = arr.reshape((28, 28))
                    image = Image.fromarray(arr)
                    st.write(f"MNIST Number {label}") 
                    st.image(arr, width=100)               
    
    except Exception as e:
        st.write("Sorry try the func 'show_mnist_from_array' because your file does not seem to work with this method") 
        st.error(e)


    