# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 16:01:13 2022

@author: Pierre
"""

import streamlit as st
from streamlit_utils import load_data, fit_a_model
import seaborn as sns
import matplotlib.pyplot as plt

page = st.sidebar.radio("",
               ('Introduction', 'Exploration', 'Modelling', 'Conclusion'))

if page == 'Introduction':
    st.title("Titanic Project")
    
    img = plt.imread("titanic.jpg")
    
    st.image(img)
    
    st.markdown("This is a sample project to showcase the ease of use of streamlit. I hope you enjoyed id.")
    
if page == 'Exploration':
    X_train, X_test, y_train, y_test = load_data()
    
    option = st.selectbox(
     'Select a column to plot',
     X_train.columns)
    
    sns.displot(X_train[option])
    
    fig = plt.gcf()
    
    st.pyplot(fig)
    
    


if page == 'Modelling':


    X_train, X_test, y_train, y_test = load_data()
    
    model_name = st.radio("Which model would you like to try",
                          ('Logistic Regression', 'KNN', 'Decision Tree'))
    
    
    
    
    model = fit_a_model(model_name, X_train, y_train)
    
    st.write(model.score(X_test, y_test))




