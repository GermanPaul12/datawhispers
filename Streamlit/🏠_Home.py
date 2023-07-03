import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import os
import requests
import io

# PageConfig
st.set_page_config(page_title='Homepage',page_icon='🏠')
st.sidebar.success('Select a page above ⬆')

# ---- HEADER SECTION ----
with st.container():
    st.title('DataWhispers - Module')
    st.write('You want to see the code? ➡ Check out the [Github Repository](https://github.com/GermanPaul12/datawhispers) 💡')

# ---- MAIN SECTION ----

with st.container():
    st.write('---')
    st.header('Target Goal')
    st.write('The purpose of this project is to help students solve tasks in advanced programming and data visualization at DHBW Mannheim. This project is designed to solve specific exams at DHBW Mannheim and is probably not appropriate for general users.')
    st.write("If you're interested how the code works you should click on [![Documentation Status](https://readthedocs.org/projects/datawhispers/badge/?version=latest)](https://datawhispers.readthedocs.io/en/latest/?badge=latest)")
    
    
    

   
# ---- Credits ----
with st.container():
    st.write('---')
    
    st.header('Contributers')
    #col1,col2,col3,col4 = st.columns(4)
    col1,col2 = st.columns(2)
    
    
    with col1:
        GP_image = Image.open("assets/GP_Github.png")
        st.image(GP_image,use_column_width=True, caption='German Paul')
    with col2:
        MG_image = Image.open("assets/MG_Github.png")
        st.image(MG_image,use_column_width=True, caption='Michael Greif')
        
        
# https://raw.githubusercontent.com/GermanPaul12/Streamlit-and-Voila-Website-Fortgeschrittene-Programmierung/blob/master/assets/img/GP_Github.png        