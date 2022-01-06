
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 11:44:01 2022

@author: akhil
"""

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow
from PIL import Image
from tensorflow import keras
from itertools import groupby


ml_model = keras.models.load_model("./model.h5")

st.title("handwritten equation solver")

st.markdown("predict the answer using images of equation")

st.sidebar.header('Input your equation image. (only png and jpg images allowed)')

image = st.file_uploader(label="equation_image", type=['jpg',"png"], accept_multiple_files=False)

def math_expression_generator(arr):
    
    op = {
              10,   # = "/"
              11,   # = "+"
              12,   # = "-"
              13    # = "*"
                  }   
    
    m_exp = []
    temp = []
        
    for item in arr:
        if item not in op:
            temp.append(item)
        else:
            m_exp.append(temp)
            m_exp.append(item)
            temp = []
    if temp:
        m_exp.append(temp)
        
    i = 0
    num = 0
    for item in m_exp:
        if type(item) == list:
            if not item:
                m_exp[i] = ""
                i = i + 1
            else:
                num_len = len(item)
                for digit in item:
                    num_len = num_len - 1
                    num = num + ((10 ** num_len) * digit)
                m_exp[i] = str(num)
                num = 0
                i = i + 1
        else:
            m_exp[i] = str(item)
            m_exp[i] = m_exp[i].replace("10","/")
            m_exp[i] = m_exp[i].replace("11","+")
            m_exp[i] = m_exp[i].replace("12","-")
            m_exp[i] = m_exp[i].replace("13","*")
            
            i = i + 1
    
    
    separator = ' '
    m_exp_str = separator.join(m_exp)
    
    return (m_exp_str)

final_equation = ""

if image is not None:
    image = Image.open(image).convert("L")
    w = image.size[0]
    h = image.size[1]
    r = w / h # aspect ratio
    new_w = int(r * 28)
    new_h = 28
    new_image = image.resize((new_w, new_h))
    new_image_arr = np.array(new_image)
    new_inv_image_arr = 255 - new_image_arr
    final_image_arr = new_inv_image_arr / 255.0

    m = final_image_arr.any(0)
    out = [final_image_arr[:,[*g]] for k, g in groupby(np.arange(len(m)), lambda x: m[x] != 0) if k]

    num_of_elements = len(out)
    elements_list = []

    for x in range(0, num_of_elements):
        img = out[x]
        width = img.shape[1]
        filler = (final_image_arr.shape[0] - width) / 2
    
        if filler.is_integer() == False:    #odd number of filler columns
           filler_l = int(filler)
           filler_r = int(filler) + 1
        else:                               #even number of filler columns
           filler_l = int(filler)
           filler_r = int(filler)
    
        arr_l = np.zeros((final_image_arr.shape[0], filler_l)) #left fillers
        arr_r = np.zeros((final_image_arr.shape[0], filler_r)) #right fillers
    
    #concatinating the left and right fillers
        help_ = np.concatenate((arr_l, img), axis= 1)
        element_arr = np.concatenate((help_, arr_r), axis= 1)
    
        element_arr.resize(28, 28, 1) #resize array 2d to 3d

    #storing all elements in a list
        elements_list.append(element_arr)


        elements_array = np.array(elements_list)

        elements_array = elements_array.reshape(-1, 28, 28, 1)
        
        elements_pred = ml_model.predict(elements_array)
        predictions = np.argmax(elements_pred, axis=1)
        
        m_exp_str = math_expression_generator(predictions)

        while True:
              try:
                 with st.spinner('in progress'):
                      answer = eval(m_exp_str)    #evaluating the answer
                      answer = round(answer, 2)
                      equation  = m_exp_str + " = " + str(answer)
                      final_equation = equation
                      print(equation)   #printing the equation
                 break

              except SyntaxError:
                     print("Invalid predicted expression!!")
                     print("Following is the predicted expression:")
                     print(m_exp_str)
                     break

if final_equation ==  "":
    st.subheader("No Input received")
else:
    st.subheader("Answer is: " + final_equation)
    

st.write("github repository: https://github.com/arnav-sys/equation-solver" )
