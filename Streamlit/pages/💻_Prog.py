from datawhispers import advancedProg as ap
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from io import BytesIO,BufferedReader

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

with st.expander("Show MNIST"):
    try:
        uploaded_file = st.file_uploader("Choose a CSV file")
        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            data = uploaded_file.getvalue().decode('utf-8').splitlines()   
            label=""
            image_files = []
            for i in range(len(data)):
                if "," in data[i]:   
                    arr = np.array([int(num) for num in data[i].split(",")])
                    if arr.shape == (785,): label,arr = arr[0],arr[1:].reshape((28, 28))
                    if arr.shape == (784,): arr = arr.reshape((28, 28))
                    st.write(f"MNIST Number {label}") 
                    st.image(arr, width=100)   
                    ret, img_enco = cv2.imencode(".png", arr)  #numpy.ndarray
                    srt_enco = img_enco.tobytes()  #bytes
                    img_BytesIO = BytesIO(srt_enco) #_io.BytesIO
                    img_BufferedReader = BufferedReader(img_BytesIO) #_io.BufferedReader
                    img_for_list = Image.open(img_BufferedReader)
                    image_files.append(img_for_list)
                else:    
                    arr = np.array([int(num) for num in data[i].split(";")])
                    if arr.shape == (785,): label,arr = arr[0],arr[1:].reshape((28, 28))
                    if arr.shape == (784,): arr = arr.reshape((28, 28))
                    st.write(f"MNIST Number {label}") 
                    st.image(arr, width=100)       
                    ret, img_enco = cv2.imencode(".png", arr)  #numpy.ndarray
                    srt_enco = img_enco.tobytes()  #bytes
                    img_BytesIO = BytesIO(srt_enco) #_io.BytesIO
                    img_BufferedReader = BufferedReader(img_BytesIO) #_io.BufferedReader
                    image_files.append(img_BufferedReader)      
            with st.container():              
                col1,col2 = st.columns(2)
                with col1:
                    st.warning("This is optional. Only use if you know what you're doing.")
                    filename = st.text_input("Input path where to save. \nDefault is cwd.", "")
                with col2:
                    st.info("""Download your images here
                             """, icon="😎")
                    st.write("")
                    st.write("")
                    st.write("")
                    if st.button("Download all images", ):
                        for index in range(len(image_files)): 
                            image_files[index].save(filename + f"mnist_outut_{index}.png")      
            
    except Exception as e:
        st.write("Sorry try the module datawhsipers.advancedProg because your file does not seem to work with this method") 
        st.error(e)

st.subheader("Add two MNIST Numbers")

with st.expander("Add MNIST"):
    try:
        uploaded_df = st.file_uploader("Choose a CSV file", key="df_add_mnist")
        if uploaded_df is not None:
            bytes_data = uploaded_df.getvalue()
            data = uploaded_df.getvalue().decode('utf-8').splitlines() 
            nums_dict,nums_list={},[i for i in range(10)]
            for i in range(len(data)):
                if ";" in data[i]:
                    line = [int(num) for num in data[i].split(";")]
                    label,arr=line[0],line[1:]
                    if list(nums_dict.keys()) == nums_list: 
                        break
                    if label in nums_dict:
                        continue
                    else:
                        nums_dict[label]= arr
                else:
                    line = [int(num) for num in data[i].split(",")]
                    label,arr=line[0],line[1:]  
                    if list(nums_dict.keys()) == nums_list: 
                        break
                    if label in nums_dict:
                        continue
                    else:
                        nums_dict[label]= arr
            with st.container():
                col1,col2 = st.columns(2)            
                with col1:
                    choice1 = st.selectbox("Choose first num", np.array([i for i in range(10)]), index=0)                            
                with col2:
                    choice2 = st.selectbox("Choose second num", np.array([i for i in range(10)]), index=1)      
                if st.button('Add nums'):
                    #st.write(f"{choice1} + {choice2} from {nums_dict}")
                    arr = np.array(nums_dict[choice1]) + np.array(nums_dict[choice2])
                    if arr.shape == (784,): arr = arr.reshape((28, 28))
                    st.write(f"MNIST Number {choice1} + {choice2}") 
                    st.image(arr, width=200, clamp=True)  
                    with st.container():
                        col1,col2 = st.columns(2)  
                        with col1:
                            filename = st.text_input("Enter your desired filename", "add_mnist")
                        with col2:    
                            ret, img_enco = cv2.imencode(".png", arr)  #numpy.ndarray
                            srt_enco = img_enco.tobytes()  #bytes
                            img_BytesIO = BytesIO(srt_enco) #_io.BytesIO
                            img_BufferedReader = BufferedReader(img_BytesIO) #_io.BufferedReader
                            #st.image(img)
                            st.download_button("Download image", img_BufferedReader, file_name=filename + ".png")    
                else:
                    st.write('Select your numbers and press button')
    except Exception as e:
        st.write("Sorry try the module datawhsipers.advancedProg because your file does not seem to work with this method") 
        st.error(e)