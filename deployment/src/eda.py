import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import streamlit as st
import altair as alt

def run():
    # header
    st.title("Exploratory Data Analysis (EDA)")
    st.markdown('---')

    #show dataframe
    df = pd.read_csv('online_shoppers_intention.csv')
    st.dataframe(df.head(10))

    # Visualisasi distribusi target
    st.subheader('Target Distribution Analysis')
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='Revenue', ax=ax)
    ax.set_title('Target Distribution (Revenue)')
    st.pyplot(fig)
    st.markdown('Target distribution shows that most visitors do not make a purchase (Revenue = 0)')

    # analysis by month of visit
    st.subheader('Monthly Visit Analysis')
    month_counts = df['Month'].value_counts().sort_index()
    fig, ax = plt.subplots()
    sns.barplot(x=month_counts.index, y=month_counts.values, ax=ax)
    ax.set_title('Number of Visits per Month')
    st.pyplot(fig)
    st.markdown('The month with the highest number of visits is September, followed by October and November.')
    st.markdown('The months with the lowest number of visits are February and March.')
    st.markdown('These months may have an impact on visitor purchase behavior.')
    st.markdown('Further analysis is needed to understand the factors influencing purchases.')
    st.markdown('---')

    # analysis by visitor type
    st.subheader('Visitor Type Analysis')
    visitor_counts = df['VisitorType'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=visitor_counts.index, y=visitor_counts.values, ax=ax)
    ax.set_title('Number of Visitors by Visitor Type')
    st.pyplot(fig)
    st.markdown('The most common visitor type is Returning Visitor, followed by New Visitor and Other.')
    st.markdown('This indicates that the majority of visitors are those who have visited before.')
    st.markdown('Further analysis is needed to understand purchase behavior by visitor type.')
    st.markdown('---')

    st.subheader('Average Session Duration Analysis: Purchasers vs. Non-Purchasers')
    duration_cols = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration']
    df['average_duration'] = df[duration_cols].mean(axis=1)
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='Revenue', y='average_duration', ax=ax)
    ax.set_title('Average Session Duration: Purchasers vs. Non-Purchasers')
    st.pyplot(fig)
    st.markdown('This boxplot shows the distribution of average session duration for purchasers and non-purchasers.')
    st.markdown('Further analysis is needed to understand the relationship between session duration and purchase behavior.')


if __name__ == "__main__":
    run()