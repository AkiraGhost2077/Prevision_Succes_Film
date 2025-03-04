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
elif page == "DataVizualization":
    st.write("Bienvenue sur la page DataVizualization")
    # Ajoute ici tes graphiques et visualisations
elif page == "Modélisation":
    st.write("Bienvenue sur la page Modélisation")
    # Ajoute ici le code de modélisation
