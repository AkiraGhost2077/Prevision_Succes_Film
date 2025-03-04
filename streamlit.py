elif page == pages[2]:
    st.write("### Modélisation")
    
    # (b) Supprimer les variables non pertinentes
    df_model = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    # (c) Création de la variable cible et des variables explicatives
    y = df_model['Survived']
    X_cat = df_model[['Pclass', 'Sex', 'Embarked']]
    X_num = df_model[['Age', 'Fare', 'SibSp', 'Parch']]
    
    # (d) Remplacer les valeurs manquantes
    for col in X_cat.columns:
        X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
    for col in X_num.columns:
        X_num[col] = X_num[col].fillna(X_num[col].median())
    
    # (e) Encoder les variables catégorielles
    X_cat_encoded = pd.get_dummies(X_cat, columns=X_cat.columns)
    
    # (f) Concaténer les variables explicatives encodées et numériques
    X = pd.concat([X_cat_encoded, X_num], axis=1)
    
    # (g) Séparer les données en ensemble d'entraînement et de test
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
    
    # (i) Fonction de prédiction qui entraîne le classifieur choisi
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
    
    # Fonction d'évaluation renvoyant l'accuracy ou la matrice de confusion
    def scores(clf, choice):
        if choice == 'Accuracy':
            return clf.score(X_test, y_test)
        elif choice == 'Confusion matrix':
            return confusion_matrix(y_test, clf.predict(X_test))
    
    # Choix de la métrique d'évaluation via une selectbox
    metric_choice = st.selectbox("Choisir une métrique", ['Accuracy', 'Confusion matrix'])
    result = scores(clf, metric_choice)
    st.write("Résultat :", result)
