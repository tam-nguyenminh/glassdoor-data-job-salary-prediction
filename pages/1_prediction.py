import streamlit as st
import pandas as pd 
import numpy as np 
import json, pickle
from sklearn.base import BaseEstimator, TransformerMixin

# st.title('Salary Prediction :moneybag:')
st.header(':blue[Predict Data Job Salary] :moneybag:', divider='rainbow')
st.markdown('The model has a mean error of about  20-25% compared to actual average salary.')

def import_feature_value(filename):
    with open(filename, 'r') as f:
        dict = json.load(f)
        mylist = list(dict.values())[0]
        list_name = list(dict.items())[0][0]    
        return  list_name, mylist

industry, list_industry = import_feature_value('static/dict_Industry_.json')
sector, list_sector = import_feature_value('static/dict_Sector_.json')
job_state, list_job_state = import_feature_value('static/dict_job_state_.json')
size, list_size = import_feature_value('static/dict_Size_.json')
ownership, list_ownership = import_feature_value('static/dict_Type of ownership_.json')

def get_feature_list():
    features = ['Industry', 'Sector', 'job_state', 'Type of ownership', 'Size',
       'python_yn', 'spark', 'aws', 'excel']
    return features
    
def skill_toggle_yesno(skillname):
    toggle_on = st.toggle(skillname)
    skill=0
    if toggle_on:
        skill = 1
    # print(f'{toggle_on} {skill}')
    return skill

def load_model():
    with open('model_salary_glassdoor_lr.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def predict_():
    model = load_model()

class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, colname, listvalue):
        self.colname = colname
        self.listvalue = listvalue
        self.mapping = {value: idx for idx, value in enumerate(listvalue)}
    def fit(self, X, y=None):
        return self 
    def transform(self, X):
        if isinstance(X, pd.Series):
            # If X is a Series, handle as a single column DataFrame
            X = X.to_frame()
        if self.colname not in X.columns:
            raise ValueError(f"Column {self.colname} does not exist in the DataFrame")
        X[self.colname] = X[self.colname].map(self.mapping).astype(int)
        if isinstance(X, pd.Series):
            # If input was originally a Series, return a Series
            return X[self.colname]
        return X

def get_new_data(newdata_list):
       # Create a DataFrame with the columns
    features = get_feature_list()
    df_new = pd.DataFrame(columns=features)
    df_new.loc[0] = newdata_list
    return df_new

def show_predict_page():


    if 'visibility' not in st.session_state:
        st.session_state.visibility = 'visibility'
        st.session_state.disabled = False
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Input Entreprise Info.')
        industry_slt = st.selectbox(f'Select industry', list_industry)
        sector_slt = st.selectbox(f'Select sector', list_sector)
        state_slt = st.selectbox(f'Select job state', list_job_state)
        ownership_slt = st.selectbox(f'Select ownership', list_ownership)
        size_slt = st.selectbox(f'Select size', list_size)
        

    with col2:
        st.subheader('Input Skills.')
        python = skill_toggle_yesno('python')
        AWS = skill_toggle_yesno('AWS')
        Spark = skill_toggle_yesno('Spark')
        Excel = skill_toggle_yesno('Excel')

    # get newdata
    newdata_list = [industry_slt, sector_slt, state_slt, ownership_slt, size_slt, python, AWS, Spark, Excel]
    print(newdata_list)

    df_new = get_new_data(newdata_list)

    st.markdown(f'\n')
    # PREDICTION
    model = load_model()
    predicted_val = round(model.predict(df_new)[0],3)

    st.subheader(f'Estimated salary (thousands USD): ')
    st.header(f':dollar: :red[${predicted_val}]')

    st.divider()


show_predict_page()