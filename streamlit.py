# -*- coding: utf-8 -*-
"""
Created on Sat March 7 2023

@author: sourav jha
"""

from tensorflow.keras.preprocessing.image import load_img, img_to_array

import numpy as np
import pickle
import streamlit as st 

pickle_in = open("Pickle_RL_Model.pkl","rb")
classifier=pickle.load(pickle_in)



def main():
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Garbage classification app </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    res = "click on predict"
    
    image = st.file_uploader("Choose a file")
    if image is not None:
        img = load_img(image, target_size=(224,224))
        img = img_to_array(img)
        img = img / 255
        img = np.expand_dims(img,axis=0)
    if st.button("Predict"):
        answer = classifier.predict(img)
        if answer[0][0] > 0.5:
            res = "The image belongs to Recycle waste category"
        else:
            res = "The image belongs to Organic waste category "
    st.success('The output is {}'.format(res))

if __name__=='__main__':
    main()
    
    
    