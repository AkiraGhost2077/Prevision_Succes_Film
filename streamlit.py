import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

csv_url = "https://raw.githubusercontent.com/AkiraGhost2077/Prevision_Succes_Film/main/train.csv"

@st.cache_data
def load_data():
    return pd.read_csv(csv_url)

df = load_data()

# 🔥 Interface Streamlit
st.title("Projet de classification binaire Titanic")
st.sidebar.title("Sommaire")
pages = ["Exploration", "DataVizualization", "Modélisation"]
page = st.sidebar.radio("Aller vers", pages)

if page == pages[0]:
    st.write("### Introduction")
    st.write("✅ Fichier CSV chargé avec succès !")
    st.dataframe(df.head(10))  # Afficher les 10 premières lignes du DataFrame
    st.write("Dimensions du dataframe :", df.shape)
    st.dataframe(df.describe())
    if st.checkbox("Afficher les NA"):
        st.dataframe(df.isna().sum())
elif page == pages[1]:
    st.write("### DataVizualization")
    
    # Distribution de la variable cible "Survived"
    fig = plt.figure()
    sns.countplot(x='Survived', data=df)
    plt.title("Distribution de la variable 'Survived'")
    st.pyplot(fig)
    
    # Répartition du genre des passagers
    fig = plt.figure()
    sns.countplot(x='Sex', data=df)
    plt.title("Répartition du genre des passagers")
    st.pyplot(fig)
    
    # Répartition des classes des passagers
    fig = plt.figure()
    sns.countplot(x='Pclass', data=df)
    plt.title("Répartition des classes des passagers")
    st.pyplot(fig)
    
    # Distribution de l'âge des passagers
    fig = plt.figure()
    sns.histplot(x='Age', data=df, kde=True)  # Utilisation de histplot qui retourne une figure
    plt.title("Distribution de l'âge des passagers")
    st.pyplot(fig)
    
elif page == pages[2]:
    st.write("### Modélisation")
    # Ajoute ici le code de modélisation
