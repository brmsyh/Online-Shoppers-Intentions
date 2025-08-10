import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import streamlit as st
import altair as alt
import 

def run():
    # header
    st.title("Exploratory Data Analysis (EDA) Website jualbeli.co.id")
    st.image("jualbeli.co.id.jpg")
    st.markdown('---')
    
    # Visualisasi distribusi target
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='Revenue', ax=ax)
    ax.set_title('Distribusi Target (Revenue)')
    st.pyplot(fig)
    st.markdown('distribusi target menunjukkan bahwa sebagian besar pengunjung tidak melakukan pembelian (Revenue = 0)')

    # analisis berdasarkan bulan kunjungan
    month_counts = df['Month'].value_counts().sort_index()
    fig, ax = plt.subplots()
    sns.barplot(x=month_counts.index, y=month_counts.values, ax=ax)
    ax.set_title('Jumlah Kunjungan per Bulan')
    st.pyplot(fig)
    st.markdown('Bulan dengan jumlah kunjungan tertinggi adalah September, diikuti oleh Oktober dan November.')
    st.markdown('Bulan dengan jumlah kunjungan terendah adalah Februari dan Maret.')
    st.markdown('Bulan-bulan ini mungkin memiliki pengaruh terhadap perilaku pembelian pengunjung.')
    st.markdown('Perlu analisis lebih lanjut untuk memahami faktor-faktor yang mempengaruhi pembelian.')
    st.markdown('---')

    # analisis pengunjung berdasarkan jenis visitornya
    visitor_counts = df['VisitorType'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=visitor_counts.index, y=visitor_counts.values, ax=ax)
    ax.set_title('Jumlah Pengunjung berdasarkan Tipe Visitor')
    st.pyplot(fig)
    st.markdown('Tipe pengunjung yang paling banyak adalah Returning Visitor, diikuti oleh New Visitor dan Other.')
    st.markdown('Hal ini menunjukkan bahwa sebagian besar pengunjung adalah pengunjung yang sudah pernah berkunjung sebelumnya.')
    st.markdown('Perlu analisis lebih lanjut untuk memahami perilaku pembelian berdasarkan tipe pengunjung.')
    st.markdown('---')

    



if __name__ == "__main__":
    run()