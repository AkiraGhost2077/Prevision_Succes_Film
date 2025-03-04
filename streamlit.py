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
    sns.histplot(x='Age', data=df, kde=True)
    plt.title("Distribution de l'âge des passagers")
    st.pyplot(fig)
    
    # (d) Countplot de la variable cible en fonction du genre.
    fig = plt.figure()
    sns.countplot(x='Survived', hue='Sex', data=df)
    plt.title("Countplot de 'Survived' par genre")
    st.pyplot(fig)
    
    # (e) Plot de la variable cible en fonction des classes.
    cat_fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
    plt.title("Point plot de 'Survived' en fonction des classes")
    st.pyplot(cat_fig.fig)
    
    # (f) Plot de la variable cible en fonction des âges.
    lm_fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
    plt.title("Relation entre 'Age' et 'Survived' selon la classe")
    st.pyplot(lm_fig.fig)
    
elif page == pages[2]:
    st.write("### Modélisation")
    
    # (b) Supprimer les variables non pertinentes
    df_model = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    # (c) Définir la variable cible et les variables explicatives
    y = df_model['Survived']
    X_cat = df_model[['Pclass', 'Sex', 'Embarked']].copy()
    X_num = df_model[['Age', 'Fare', 'SibSp', 'Parch']].copy()
    
    # (d) Remplacer les valeurs manquantes
    for col in X_cat.columns:
        X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
    for col in X_num.columns:
        X_num[col] = X_num[col].fillna(X_num[col].median())
    
    # (e) Encoder les variables catégorielles
    X_cat_encoded = pd.get_dummies(X_cat, columns=X_cat.columns)
    
    # (f) Concaténer les variables explicatives encodées et numériques
    X = pd.concat([X_cat_encoded, X_num], axis=1)
    
    # (g) Séparer les données en ensembles d'entraînement et de test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    # (h) Standardiser les variables numériques
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
    X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])
    
    # (j) Utiliser une selectbox pour choisir le classifieur
    choix = ['Random Forest', 'SVC', 'Logistic Regression']
    option = st.selectbox('Choix du modèle', choix)
    st.write('Le modèle choisi est :', option)
    
    # (i) Créer une fonction de prédiction pour entraîner le classifieur choisi
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    
    def prediction(classifier):
        if classifier == 'Random Forest':
            clf = RandomForestClassifier()
        elif classifier == 'SVC':
            clf = SVC()
        elif classifier == 'Logistic Regression':
            clf = LogisticRegression()
        clf.fit(X_train, y_train)
        return clf
    
    clf = prediction(option)
    
    # Fonction pour retourner l'accuracy ou la matrice de confusion
    def scores(clf, choice):
        if choice == 'Accuracy':
            return clf.score(X_test, y_test)
        elif choice == 'Confusion matrix':
            return confusion_matrix(y_test, clf.predict(X_test))
    
    # Interface utilisateur pour choisir la métrique d'évaluation
    metric_choice = st.selectbox("Choisir une métrique", ['Accuracy', 'Confusion matrix'])
    result = scores(clf, metric_choice)
    st.write("Résultat :", result)
