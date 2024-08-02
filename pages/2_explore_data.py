import streamlit as st
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

import altair as alt

df = pd.read_csv('salary_glassdoor_data_cleaned_v1.csv')
feature_obj = df.select_dtypes(include='object').columns

def generate_chart(df, x, y ):
    """Generate a sample chart."""
    plt.figure(figsize=(15,10))

    chart = alt.Chart(df).mark_circle(size=100).encode(x=x, y=y, color=alt.Color(y, legend=None)).properties(width=600, height=600).interactive()
    # fig, ax = plt.subplots()
    # ax.plot(data)
    # ax.set_title(title)
    # chart
    return chart

def create_tab_with_chart(df, feature_list):
    num_tabs = len(feature_obj)
    tab_titles = [f'{i}' for i in feature_obj]
    # tab_titles = [f'Tab {i}' for i in range(1, num_tabs + 1)]

    tabs = st.tabs(tab_titles)
    for i, tab in enumerate(tabs):
        
        with tab:
            chart = generate_chart(df, 'avg_salary', feature_obj[i])
            
            st.altair_chart(chart, theme = None, use_container_width=True)
            
        
def show_info_page():
    # st.title('Exploratory')
    st.header('Explore Data', divider='rainbow')
    st.subheader('About this dataset')
    st.markdown(f'This dataset contains job postings from Glassdoor.com from 2017 with the following features It can be used to analyze the current trends based on job positions, company size, etc.')
    st.subheader('Data Source:')
    st.markdown('https://www.kaggle.com/datasets/thedevastator/jobs-dataset-from-glassdoor')
    # st.markdown('shape = (742 rows, 28 columns)')

    # Create a histogram of avg_salary
    st.divider()

    st.subheader('Take a look')
    charthisto = alt.Chart(df[['avg_salary']]).mark_bar().encode(
        alt.X('avg_salary:Q', bin=True, title='Value'),
        alt.Y('count():Q', title='Frequency')
    ).properties(
        title='Distribution of average salary',
        width=600,
        height=400)
    st.altair_chart(charthisto, use_container_width=True)
    # Create a histogram with KDE
    # fig, ax = plt.subplots()
    # sns.histplot(df[['avg_salary']], kde=True, edgecolor='black', ax=ax)
    # ax.set_title('Histogram with KDE')
    # ax.set_xlabel('Value')
    # ax.set_ylabel('Frequency')

    # Display the plot in Streamlit
    # st.pyplot(fig)

    st.markdown('Average salary distributed by different attributes')
    create_tab_with_chart(df, feature_obj)

    st.markdown('Sneak at the data')
    st.table(df.head())
show_info_page()