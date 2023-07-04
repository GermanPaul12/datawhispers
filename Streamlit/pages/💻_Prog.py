from datawhispers import advancedProg as ap
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from io import BytesIO,BufferedReader, StringIO

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

st.subheader("Make Regression")

with st.expander("Make Regressions"):
    try:
        st.session_state["data_dfs"] = ["Df not uploaded yet", "Df2 not uploaded yet"]   
        if st.checkbox("Upload df number 1"):
            upload_df = st.file_uploader("Choose a CSV file", key="df_make_reg")
            if upload_df is not None:
                st.session_state.df1 = []            
                name = upload_df.name
                bytes_data = upload_df.getvalue()
                data = upload_df.getvalue().decode('utf-8').splitlines()     
                sep=","
                st.write(data[0][0])
                for index in range(len(data)):
                    if "#" in data[index]:
                        data = data[1:]   
                        continue
                    if ";" in data[index]:
                        sep=";"
                    data = [data[i].split(sep) for i in range(len(data))]
                    cols,arr=data[0],data[1:]
                    #data  
                    st.session_state.df1 = pd.DataFrame(arr, columns=cols)   
                    st.session_state.df1 = st.session_state.df1.apply(pd.to_numeric, errors='ignore')    
                    st.session_state["data_dfs"][0] = name
                    break
                if st.checkbox(f"Show the Dataframe of {st.session_state['data_dfs'][0]}", key="chkbox 1"):    
                    st.session_state.df1         
                    
                if st.checkbox("Upload df number 2"):
                    upload_df2 = st.file_uploader("Choose a CSV file", key="df_make_reg2")
                    if upload_df2 is not None: 
                        st.session_state.df2=[]          
                        name = upload_df2.name
                        bytes_data = upload_df2.getvalue()
                        data = upload_df2.getvalue().decode('utf-8').splitlines()     
                        sep=","
                        for index in range(len(data)):
                            if "#" in data[index]:
                                data = data[1:]   
                                continue
                            if ";" in data[index]:
                                sep=";"
                            data = [data[i].split(sep) for i in range(len(data))]
                            cols,arr=data[0],data[1:]
                            #data  
                            st.session_state.df2 = pd.DataFrame(arr, columns=cols)      
                            st.session_state.df2 = st.session_state.df2.apply(pd.to_numeric, errors='ignore')  
                            st.session_state["data_dfs"][1] = name
                            break
                        if st.checkbox(f"Show the Dataframe of {st.session_state['data_dfs'][1]}", key="chkbox 2"):    
                            st.session_state.df2               

                with st.container():
                    st.markdown("### Your Data")
                    col1,col2= st.columns(2)  
                    with col1:
                        st.markdown("### Your x-value:")
                        selected_df_1 = st.selectbox("Select your df", [st.session_state.data_dfs[i] for i in range(len(st.session_state.data_dfs))], index=0, key="selectbox_df_1")
                        chosen_df_1=st.session_state.df1 if selected_df_1 == st.session_state.data_dfs[0] else st.session_state.df2
                        selected_column_1 = st.selectbox("Select your column", [column for column in chosen_df_1.columns], index=0, key="selectbox_col_1")
                    with col2:
                        st.markdown("### Your y-value:")
                        selected_df_2 = st.selectbox("Select your df", [st.session_state.data_dfs[i] for i in range(len(st.session_state.data_dfs))], index=0, key="selectbox_df_2")
                        chosen_df_2=st.session_state.df1 if selected_df_2 == st.session_state.data_dfs[0] else st.session_state.df2
                        selected_column_2 = st.selectbox("Select your column", [column for column in chosen_df_2.columns], index=1, key="selectbox_col_2")    
                    st.markdown("### Your Regression")
                    col1,col2= st.columns(2) 
                    with col1: 
                        ansatz = st.selectbox("Ansatz", ["linReg", "polReg", "trigReg", "expReg"])    
                        
                        if ansatz == "polReg": 
                            with col2:
                                deg = st.selectbox("Degree", [i for i in range(2, 15)])
                                model = ap.Trend(chosen_df_1[selected_column_1], chosen_df_2[selected_column_2], ansatz=ansatz, deg=int(deg)) 
                                deg   
                        else:
                            pass
                            model = ap.Trend(chosen_df_1[selected_column_1], chosen_df_2[selected_column_2], ansatz=ansatz)

                        #buffer = StringIO()    
                        #chosen_df_1.info(buf=buffer)
                        #s = buffer.getvalue()
                        #st.text(s)    
                        if st.checkbox("Change advanced Plot settings"):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1: 
                                st.write("x-ticks")
                            with col2:    
                                xticks_start = st.text_input("start (input integer)", "", key="text_xticks_start", )
                            with col3:    
                                xticks_end = st.text_input("end (input integer)", "", key="text_xticks_end", )    
                            with col4:    
                                xticks = st.text_input("number of ticks", 0, key="xticks", )        
                            col1, col2, col3, col4 = st.columns(4)
                            with col1: 
                                st.write("y-ticks")
                            with col2:    
                                yticks_start = st.text_input("start (input integer)", "", key="text_yticks_start", )
                            with col3:    
                                yticks_end = st.text_input("end (input integer)", "", key="text_yticks_end", )    
                            with col4:    
                                yticks = st.text_input("number of ticks", 0, key="yticks", )
                            col1, col2 = st.columns(2)
                            with col1: 
                                st.write("set x-lim")
                            with col2:    
                                xlim_left,xlim_right = st.text_input("Left x-limit", "", key="xlim_left", ),st.text_input("Right x-limit", "", key="xlim_right", ) 
                            col1, col2 = st.columns(2)
                            with col1: 
                                st.write("set y-lim")
                            with col2:    
                                ylim_left,ylim_right = st.text_input("Lower y-limit", "", key="ylim_left", ),st.text_input("Upper y-limit", "", key="ylim_right", )                       
                            col1, col2, col3 = st.columns(3)      
                            with col1:
                                xlabel = st.text_input("Input a xlabel", "x", key="xlabel")
                            with col2:    
                                ylabel = st.text_input("Input a ylabel", "y", key="ylabel")    
                            with col3:    
                                title = st.text_input("Input a title", "", key="title")        
                            col1, col2 = st.columns(2)
                            with col1:
                                sc_color = st.text_input("Input a color for the scatterplot", "lightblue", key="scatter_color")
                            with col2:
                                pt_color = st.text_input("Input a color for the reg plot", "black", key="plot_color")   
                            fig, ax = plt.subplots()                   
                            ax.plot(model.x,ap.predict(model.ansatz, model.coef, model.x),color=pt_color);
                            ax.scatter(model.x,model.y, color=sc_color);
                            ax.set_xlabel(xlabel) 
                            ax.set_ylabel(ylabel)
                            ax.set_title(title)
                            if xticks_start and xticks_end: ax.set_xticks(np.linspace(float(xticks_start), float(xticks_end), int(xticks)));
                            else: ax.set_xticks(ax.get_xticks())
                            if yticks_start and yticks_end: ax.set_yticks(np.linspace(float(yticks_start), float(yticks_end), int(yticks)));
                            else: ax.set_yticks(ax.get_yticks())
                            if xlim_left and xlim_right: ax.set_xlim(float(xlim_left), float(xlim_right))
                            else: ax.set_xlim(ax.get_xlim())
                            if ylim_left and ylim_right: ax.set_ylim(float(ylim_left), float(ylim_right))
                            else: ax.set_ylim(ax.get_ylim())
                            #plt.savefig(f"{name}");  
                            st.pyplot(fig)
                        else:
                            fig, ax = plt.subplots()                   
                            ax.plot(model.x,ap.predict(model.ansatz, model.coef, model.x),color="black");
                            ax.scatter(model.x,model.y, color="lightblue");  
                            st.pyplot(fig)
                            #plt.savefig(f"{name}");
                                
                            
                        st.write(f"r2-value: {model.r2}, coefs: {model.coef}")  
                        filename = st.text_input("Enter your desired filename", "reg_plot", key="filename_reg")
                        filename=filename+".png"
                        fig.savefig(filename)
                        with open(filename, "rb") as img:
                            btn = st.download_button(
                                label="Download graph",
                                data=img,
                                file_name=filename,
                                mime="image/png"
    )
                                
                        
        
    except Exception as e:
        st.write("Sorry try the module datawhsipers.advancedProg because your file does not seem to work with this method") 
        st.error(e)        