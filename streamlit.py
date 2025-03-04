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

# Afficher le contenu selon la page sélectionnée
if page == "Exploration":
    st.write("✅ Fichier CSV chargé avec succès !")
    st.dataframe(df.head())  # Afficher les premières lignes du DataFrame
elif page == "DataVizualization":
    st.write("Bienvenue sur la page DataVizualization")
    # Ajoute ici tes graphiques et visualisations
elif page == "Modélisation":
    st.write("Bienvenue sur la page Modélisation")
    # Ajoute ici le code de modélisation
