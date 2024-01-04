#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install streamlit


# In[2]:


#!pip install --upgrade altair


# In[3]:


#!pip install pandas


# In[4]:
!pip install scikit-learn


# import all libraries needed
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import warnings
# To ignore all warnings
warnings.filterwarnings("ignore")

from absenteeism_module import *


# In[5]:


def Input_Output():
    data = st.file_uploader('ipload file', type={'csv', 'txt'})
    if data is not None:
        df = pd.read_csv(data)
        model = absenteeism_model('model')
        model.load_and_clean_data('Absenteeism_new_data.csv')
    result = ''
    if st.button('Click here to Predict'):
        result = model.predicted_outputs()
        st.balloons()
    st.success('The output is as follows: ')
    st.write(result)

if __name__ =='__main__':
    Input_Output()


# In[ ]:





# In[ ]:




