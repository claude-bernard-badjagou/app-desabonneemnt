# Installation puis importation des packages 
# Packages de manipulation de données
import pandas as pd
import pickle as pk
import joblib

# Packages de visualisation de données 
import seaborn as sns
import matplotlib.pyplot as plt # à remplacer par plotly express
import plotly.express as px

# Packages de machine learning
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report, confusion_matrix

# Package l'interface web
import streamlit as st

# PARTIE 1 : CHARGEMENT DE TOUTES LES FONCTIONS ET VARIABLES UTILES POUR LE PROCESSUS
# Chargement des fichiers
# Chemin vers les fichiers contenant les données d'entrainement, de test et de leur description
fichier_train = 'churn_data_train.txt'
fichier_test = 'churn_data_test.txt'
description = 'description.txt'

# Lecture les ensembles de données
# Je dois utiliser l'option 'sep' pour spécifier à pandas que les données sont séparées par 1 espace.
df_train = pd.read_csv(fichier_train, sep=r" ", engine='python')
df_test = pd.read_csv(fichier_test, sep=r" ", engine='python')

# Description des données dans chaque colonne
def description_donnees(description):
    with open(description, 'r', encoding='utf-8') as file:
        description = file.read()
    return description

# Toute bonne analyse commence par de l'observation.
def echantillon(df):
    df_echantillon = df.sample(3)
    return df_echantillon

# Dimensions du dataset
def dimension(df):
    return {df.shape[0]}, {df.shape[1]}

nb_lignes, nb_colonnes = dimension(df_train)

# Ordonner les index des données
def ordonne_index(df):
    df_ordonne = df.sort_index()
    return df_ordonne

# Identifier les colonnes numériques et celles catégorielles dans les deux ensembles de données.
col_numeriques = df_test.select_dtypes(include=["float64", "int64"]).columns
col_categorielles = df_test.select_dtypes(include=["object"]).columns

# **3. Analyse des anomalies et données manquantes**
# Vérificattion des valeurs manquantes dans les données
def donnees_manquantes(df):
    donnees_manquantes = df.isnull().sum()
    donnees_manquantes = donnees_manquantes[donnees_manquantes > 0].sort_values(ascending=False)
    st.write("\nLes colonnes avec le nombre de valeurs manquantes :")
    st.write(donnees_manquantes)


# Valeurs aberrantes dans les colonnes numériques de mes données
# Initialiser une liste pour stocker les colonnes avec valeurs aberrantes
colonnes_aberrantes = []

def valeurs_aberrantes(df, col_numeriques):
    # Définir le seuil pour identifier les valeurs aberrantes en fonction du score Z
    seuil_z = 3

    # Boucle à travers chaque colonne numérique
    for colonne in col_numeriques:
        # Calculer la moyenne et l'écart-type de la colonne actuelle
        moyenne = df[colonne].mean()
        ecart_type = df[colonne].std()

        # Calculer les scores Z pour la colonne actuelle
        scores_z = (df[colonne] - moyenne) / ecart_type

        # Identifier les valeurs aberrantes pour la colonne actuelle
        valeurs_aberrantes = df[abs(scores_z) > seuil_z]

        # Afficher les valeurs aberrantes pour la colonne actuelle
        if not valeurs_aberrantes.empty:
             colonnes_aberrantes.append(colonne)
             #print(f"Voici les valeurs aberrantes dans la colonne '{colonne}' avec zscore = 3 :")
             #for index, valeur in valeurs_aberrantes[colonne].items():
                #print(f"Ligne: {index}, Valeur: {valeur}")
    st.write(f"\nIl y a {len(colonnes_aberrantes)} colonnes avec des valeurs aberrantes : {colonnes_aberrantes}")


# Vérification s'il y a des lignes dupliquées
def lignes_dupliquees(df):
    lignes_dupliquees = df[df.duplicated()]
    st.write(f"Il y a {lignes_dupliquees.shape[0]} lignes dupliquées dans cet ensemble de données.")

# Colonnes catégorielles avec le nombre de valeurs uniques
def count_unique_values(df):
    # Créer un dictionnaire pour stocker les résultats
    unique_counts_dict = {}
    # Compter les valeurs uniques pour chaque colonne catégorielle
    for col in col_categorielles:
        unique_counts_dict[col] = df[col].nunique()
    # Convertir le dictionnaire en DataFrame et définir l'index
    unique_counts = pd.DataFrame.from_dict(unique_counts_dict, orient='index', columns=['Nombre Valeurs distinctes'])
    return unique_counts

def valeurs_uniques(df, col_categorielles):
    for col in col_categorielles:
        col_value_unique = df[col].unique()
        st.write(f"Il y a {len(col_value_unique)} valeurs uniques dans la colonne '{col}':\n{df[col].value_counts()}\n")

# **1. Transformation des valeurs manquantes**
# Remplacement des valeurs manquantes dans chaque colonne
def valeur_manquante_remplacee(df, col, valeur):
    df[col].fillna({col : valeur}, inplace=True)
    #st.write(f"\nLes valeurs manquantes dans '{col}' sont remplacées par {valeur}.")

# Remplacement des valeurs atypiques '-99' de la colonne 'Customer.Satisfaction' par la valeur 's0'
def remplacement_valeur_atypique(df, col, valeur_actuelle, valeur_future):
    df[col] = df[col].replace(valeur_actuelle, valeur_future)
    st.write(f"\nLes valeurs atypiques '{valeur_actuelle}' dans '{col}' sont remplacées par '{valeur_future}'.")


# Sélection des colonnes catégorielles avec 2 valeurs uniques avec une boucle for           
col_cat_deux_valeurs = []
col_cat_plus_deux_valeurs = []

def repartition_valeurs_uniques(df, col_categorielles):

    for col in col_categorielles:
        if df[col].nunique() == 2:
            col_cat_deux_valeurs.append(col)
            #st.write(f"\nLa colonne '{col}' a {df[col].nunique()} valeurs uniques et est ajoutée à 'col_cat_deux_valeurs'.")
        else:
            col_cat_plus_deux_valeurs.append(col)
            #st.write(f"\nLa colonne '{col}' a {df[col].nunique()} valeurs uniques et est ajoutée à 'col_cat_plus_deux_valeurs'.")

# Encodage de la colonne avec labelEncoder s'il est dans col_cat_deux_valeurs          
def encodage_deux_valeurs(df, col_cat_deux_valeurs):
    for col in col_cat_deux_valeurs:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        #st.write(f"La colonne '{col}' a été encodée avec LabelEncoder.")

# Normalisation des colonnes avec des outliers avec RobustScaler
def normalisation_outliers(df):
    robust_scaler = RobustScaler()
    df[colonnes_aberrantes] = robust_scaler.fit_transform(df[colonnes_aberrantes])

# Normalisation des colonnes sans outliers avec standardScaler
colonnes_non_aberrantes = [col for col in col_numeriques if col not in colonnes_aberrantes]
# Supprimer les colonnes 'Churn.Value', 'Longitude', 'Latitude' des colonnes non aberrantes
for col in ['Churn.Value', 'Longitude', 'Latitude']:
    if col in colonnes_non_aberrantes:
        colonnes_non_aberrantes.remove(col)

def normalisation_non_outliers(df):
    standard_scaler = StandardScaler()
    df[colonnes_non_aberrantes] = standard_scaler.fit_transform(df[colonnes_non_aberrantes])

# chargement des données d'entrainement pretraitées après avoir transformé les données
df_train_copy = pk.load(open("df_train_traitees.pkl", "rb"))
df_test_copy = pk.load(open("df_test_traitees.pkl", "rb"))

# **5. Selections des caractéristiques importantes**
# 🎯 Objectif : Préparer les données train pour entraîner et évaluer le modèle.
def selection_colonnes_train(df, colonnes_selectionnees):
    X_train = df[colonnes_selectionnees]
    y_train = df["Churn.Value"]
    return X_train, y_train

# Préparer les données test pour tester le modèle
def selection_colonnes_test(df, colonnes_selectionnees):
    X_test = df[colonnes_selectionnees]
    y_test = df["Churn.Value"]
    return X_test, y_test

# Fonction pour charger le modèle avec mise en cache
@st.cache_resource
def load_model():
    return joblib.load("xgboost_model.joblib")


# PARTIE 2 : CREATION DES PAGES ET UTILISATION DES FONCTIONS ET VARIABLES
# Page: Accueil
def page_accueil():
    st.title("Accueil")
    st.write("### Bienvenue sur l'application de prédiction du désabonnement")
    st.write("L'objectif de ce travail est de prédire le désabonnement des clients dans l'industrie des télécommunications.")
    st.image("prediction.png", caption="Image générée avec Microsoft designer pour illustrer le désabonnement en télécom")
    st.write("Dans le secteur des télécommunications, les clients peuvent choisir entre plusieurs fournisseurs de services et passer activement de l'un à l'autre.")
    st.error("Problème dans la télécom:")
    st.write("La fidélisation individualisée des clients est difficile car la plupart des entreprises ont un grand nombre de clients et ne peuvent pas se permettre de consacrer beaucoup de temps à chacun d'entre eux. Les coûts seraient trop élevés et l'emporteraient sur les recettes supplémentaires.")
    st.info("Rappel important :")
    st.write("Si une entreprise pouvait prévoir à l'avance quels clients sont susceptibles de la quitter, elle pourrait concentrer ses efforts de fidélisation uniquement sur ces clients « à haut risque ».")
    st.write("En s'attaquant au problème du désabonnement, les entreprises de télécom peuvent non seulement préserver leur position sur le marché, mais aussi se développer et prospérer. Plus il y a de clients dans leur réseau, plus le coût d'initiation est faible et plus les bénéfices sont importants. Par conséquent, l'objectif principal de l'entreprise pour réussir est de réduire l'attrition des clients et de mettre en œuvre une stratégie de fidélisation efficace.")
    st.success("Approche de solution :")
    st.write("Pour détecter les signes précurseurs d'un désabonnement potentiel, il faut d'abord développer une vision globale des clients et de leurs interactions sur de nombreux canaux, notamment l'utilisation du service, l'historique des problèmes rencontrés, les appels au service clientèle, pour n'en citer que ces quelques-uns.")
    st.write("Rendez-vous sur la page Informations pour en savoir plus sur les données collectées sur les clients")

# Page: Informations
def page_informations():
    st.title("Informations")
    st.write("### Informations sur les données")
    st.write("Cette page fournit des détails sur les sources, formats et descriptions des données utilisées.")
    st.info("LES DONNEES (REELLES OU FICTIVES) UTILISEES DANS CE PROJET SONT FOURNIES POUR EFFECTUER UN TEST")
    st.info("Aperçu d'un échantillon des données d'entrainement du modèle (df_train)")
    st.write(echantillon(df_train))
    st.info("Aperçu d'un échantillon des données pour tester le modèle (df_test)")
    st.write(echantillon(df_test))
    st.write("Voici une description détaillée et compréhensible de chacune des colonnes de mes deux ensembles de données, ainsi que leurs significations : ")
    st.write(description_donnees(description))

    st.info("Passons à l'exploration de ces données")
  

# Page: Exploration des données
def page_exploration_des_donnees():
    st.title("Exploration des données")
    st.write("### **1. Compréhension de la structure des données**")
    st.write("Dimensions des ensembles de données")
    st.write("Données d'entrainement : il y a", nb_lignes, "lignes et ", nb_colonnes, "colonnes dans chaque ensemble de données")
    st.success("Trie des index et affichage de chaque ensemble de données")
    st.write("Données d'entrainement :\n", ordonne_index(df_train))
    st.write("Données de test :\n", ordonne_index(df_test))
    st.info("\nIdentification des colonnes numériques et celles catégorielles\n")
    st.write(f"Nous avons {len(list(col_numeriques))} colonnes numériques : {list(col_numeriques)}")
    st.write(f"\nNous avons {len(list(col_categorielles))} colonnes catégorielles : {list(col_categorielles)}")
    st.write("### **2. Satistiques descriptives**")
    st.write("Satistiques descriptives des colonnes numériques :")
    statistique_descriptive_col_numeriques = df_train[col_numeriques].describe()
    st.write(statistique_descriptive_col_numeriques.T)
    st.write("Satistiques descriptives des colonnes categorielles :")
    statistique_descriptive_col_categorielles = df_train[col_categorielles].describe()
    st.write(statistique_descriptive_col_categorielles.T)
    st.write("### **3. Analyse des anomalies et données manquantes dans train et test**")
    st.write("Vérificattion des valeurs manquantes dans chaque colonne de mes données")
    st.write("Données manquantes dans le train :\n", )
    donnees_manquantes(df_train)
    st.write("Données manquantes dans le test :\n", )
    donnees_manquantes(df_test)
    st.warning("Valeurs aberrantes identifiées dans les données train à l'aide de z-score")
    valeurs_aberrantes(df_train, col_numeriques)
    st.warning("Valeurs aberrantes identifiées dans les données test à l'aide de z-score")
    valeurs_aberrantes(df_test, col_numeriques)
    st.error("Vérification s'il y a des lignes dupliquées dans le train :")
    lignes_dupliquees(df_train)
    st.error("Vérification s'il y a des lignes dupliquées dans le test :")
    lignes_dupliquees(df_test)
    st.info("Colonnes catégorielles avec le nombre de valeurs uniques dans train :")
    count_unique_values(df_train).T
    st.info("Colonnes catégorielles avec le nombre de valeurs uniques dans test :")
    count_unique_values(df_test).T
    st.write("Colonnes catégorielles avec le nombre de valeurs uniques dans l'ensemble de données")
    if st.checkbox("Afficher le nombre d'occurence de valeurs distinctes par colonne de données train"):
        valeurs_uniques(df_train, col_categorielles)
        st.write("Remarque : La valeur '-99' la plus fréquente de la colonne 'Customer.Satisfaction' n'est pas une chaîne de caractères semblable aux autres de la même colonne. Nous allons gérer son cas par la suite.")
    if st.checkbox("Afficher le nombre d'occurence de valeurs distinctes par colonne de données test"):
        valeurs_uniques(df_test, col_categorielles)
        st.write("Remarque : La valeur '-99' la plus fréquente de la colonne 'Customer.Satisfaction' n'est pas une chaîne de caractères semblable aux autres de la même colonne. Nous allons gérer son cas par la suite.")
        st.write("Rendez-vous sur la page transformation pour comprendre les modifications apportées aux données collectées")

# Page: Transformation des données
def page_transformation_des_donnees():
    st.title("Ingénieurie de fonctionnalités")
    st.write("### **1. Transformation des données manquantes et atypiques**")
    st.warning("Remplacement des valeurs manquantes dans chaque colonne des données train et test")
    valeur_manquante_remplacee(df_train, 'Offer', 'No Offer') # train
    valeur_manquante_remplacee(df_test, 'Offer', 'No Offer') # test
    st.success("Dans les deux ensembles de données, les données manquantes dans 'Offer' sont remplacées par 'No Offer', pas d'offre")
    valeur_manquante_remplacee(df_train, 'Internet.Type', 'Mobile Networks') # train
    valeur_manquante_remplacee(df_test, 'Internet.Type', 'Mobile Networks') # test
    st.success("Dans les deux ensembles de données, les données manquantes dans 'Internet.Type' sont remplacées par 'Mobile Networks', Réseau Mobile")
    st.write("Remplacement des valeurs atypiques dans les deux ensembles de données")
    remplacement_valeur_atypique(df_train, 'Customer.Satisfaction', '-99', 's0') # train
    remplacement_valeur_atypique(df_test, 'Customer.Satisfaction', '-99', 's0')  # test
    st.success("Les valeurs atypiques '-99' de la colonne 'Customer.Satisfaction' dans les deux ensembles de données sont remplacées par's0'")
    st.write("### **2. Encodage des données**")
    st.info("Pour éviter de créer une relation d'ordre trompeur dans nos données :")
    st.write("**Encodons les colonnes avec deux valeurs uniques avec le labelEncoder**")
    repartition_valeurs_uniques(df_train, col_categorielles)
    # Création d'une copie des données
    df_train_copy = df_train.copy()  
    df_test_copy = df_test.copy()
    encodage_deux_valeurs(df_train_copy, col_cat_deux_valeurs) # train      
    encodage_deux_valeurs(df_test_copy, col_cat_deux_valeurs) # test
    st.success("Les colonnes avec deux valeurs uniques ont été encodé avec labelEncoder")
    st.write("**Encodons les colonnes avec plus de deux valeurs uniques avec le One Hot Encoder**")
    df_train_copy = pd.get_dummies(df_train_copy, columns=col_cat_plus_deux_valeurs, dtype=int)
    df_test_copy = pd.get_dummies(df_test_copy, columns=col_cat_plus_deux_valeurs, dtype=int)
    st.success("Les colonnes avec plus de deux valeurs uniques ont été encodé avec One Hot Encoder")
    st.write("### **3. Normalisation des colonnes numériques**")
    st.write("Normalisation des valeurs des colonnes identifiées comme des outliers avec RobustScaler")
    st.write("Rappel, dans le train :")
    valeurs_aberrantes(df_train_copy, col_numeriques)
    st.write("Rappel, dans le test :")
    valeurs_aberrantes(df_test_copy, col_numeriques)
    normalisation_outliers(df_train_copy) # train
    normalisation_outliers(df_test_copy) # test
    st.success("Les valeurs des colonnes identifiées comme des outliers sont normalisées avec RobustScaler")
    st.write("Normalisation des valeurs des colonnes identifiées comme non outliers avec StandardScaler")
    normalisation_non_outliers(df_train_copy) # train
    normalisation_non_outliers(df_test_copy) # test
    st.success("Les valeurs des colonnes identifiées comme non outliers sont normalisées avec StandardScaler")
    # Enregistrer les données prétraitées
    df_train_copy.to_pickle("df_train_traitees.pkl")
    st.info("9. Les données d'entrainement prétraitées sont enregistrées dans **df_train_traitees** au format pickle afin de l'utiliser par la suite")
    df_test_copy.to_pickle("df_test_traitees.pkl")
    st.info("9. Les données de test prétraitées sont enregistrées dans **df_test_traitees** au format pickle afin de l'utiliser par la suite")
 
# Page: Visualisation des données
def page_visualisation_des_donnees():
    st.title("Visualisation des données")
    st.write("### **1. Visualisation des données d'entrainement identifiées comme anormales**")
    valeurs_aberrantes(df_train, col_numeriques)
    st.write("### **2. Regardons les correlations dans les données train**")
    # Matrice de correlation
    corr = df_train_copy.corr()
    # Trier les corrélations par ordre croissant
    corr_triee = corr['Churn.Value'].sort_values(ascending=True)
    # Sélectionner les colonnes dans l'ordre trié
    colonnes_triees = corr_triee.index.tolist()
    #st.pyplot.figure(figsize=(20, 15))
    # Créer le heatmap avec les colonnes triées
    # Créer une figure pour le heatmap
    fig, ax = plt.subplots(figsize=(10, 8))  # Taille ajustée pour une meilleure lisibilité
    # Créer le heatmap avec Seaborn
    sns.heatmap(
        corr.loc[colonnes_triees, ['Churn.Value']],
        annot=True,
        cmap='YlGnBu',
        ax=ax,
        cbar_kws={'label': 'Correlation'})

    # Ajouter un titre au heatmap
    ax.set_title('Heatmap des corrélations triées avec Churn.Value', fontsize=16)

    # Afficher le graphique dans Streamlit
    st.pyplot(fig)

# Page: Développement de modèles
def page_developpement_de_modeles():
    st.title("Développement de modèles")
    st.write("### **Développement du Modèle**")
    st.write("**Sélection des caractéristiques importantes selon le coefficient de corelation**")
    corr = df_train_copy.corr()
    # Création d'une liste vide pour stocker les colonnes sélectionnées pour entrainer le modèle
    colonnes_selectionnees = []
    # Seuil de corrélation : les colonnes autour de 0 (-0.03 à 0.03) ne seront pas sélectionnées
    st.write("Seules les colonnes dont le **coefficient** est autour de **0** (-0.03 à 0.03) ne seront pas sélectionnées")
    seuil = 0.03
    for index, row in corr[['Churn.Value']].iterrows():
        if abs(row['Churn.Value']) > seuil:
            colonnes_selectionnees.append(index)
    st.write("Voici les", len(colonnes_selectionnees), "colonnes importantes pour notre modèle :", colonnes_selectionnees)   
    # Repartition des données de train et celles de test
    X_train, y_train = selection_colonnes_train(df_train_copy, colonnes_selectionnees)
    X_test, y_test = selection_colonnes_test(df_test_copy, colonnes_selectionnees)
    st.info("### **Dernières vérifications**")
    st.write("**Voici un aperçu des données finales traitées pour l'entraînement du modèle :**")  
    st.write("Voici les catactéristiques :\n", X_train.head())
    st.write("Voici les exemples de la valeur cible :\n", y_train.head())
    st.write("**Voici un aperçu des données nouvelles traitées pour tester modèle :**")
    st.write("Voici les catactéristiques :\n", X_test.head())
    st.write("Voici la cible à présire :\n", y_test.head())
    st.success("Voici un aperçu de la complémentarité entre les données d'entrainement et celles de test : Explorez les indices de lignes des données du premier tableau et celui du deuxième également.")
    st.write(ordonne_index(X_train))
    st.write(ordonne_index(X_test))
    st.write("### Choix du modèle final pour la prédiction : xgboost")
    st.write("L'algorithme XGBoost idéal dans le domaine des télécommunications car : \n - Il gère parfaitement les données déséquilibrées. \n - Il est efficace sur les données manquantes oes services moins utilisés. \n - Il traite rapidement les grands volumes de données en capturant des relations complexes.")
    # Utilisation d'une validation croisée à 5 plis pour mieux généraliser les performances.
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Modèle XGBoost
    model_xgb = xgb.XGBClassifier(
        objective="binary:logistic",  # Pour un problème de classification binaire
        n_estimators=100,            # Nombre d'arbres
        max_depth=6,                 # Profondeur des arbres
        learning_rate=0.1,           # Taux d'apprentissage
        random_state=42              # Pour la reproductibilité
        )
    # Entrainement du modèle
    model_xgb.fit(X_train, y_train)
    st.success("Le modèle XGBoost a été entrainé avec success")
    # Optimisation des hyperparamètres**
    st.write("**Utilisation de GridSearch pour trouver la meilleure combinaison de paramètres.**")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
        }

    grid_search = GridSearchCV(
        estimator=xgb.XGBClassifier(objective="binary:logistic", random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring='accuracy'
        )

    grid_search.fit(X_train, y_train)
    st.write("Voici la combinaison optimale des paramètres (meilleurs paramètres) obtenus avec GridSearch :", grid_search.best_params_)

    st.info("**Réentraînement du modèle avec les meilleurs paramètres**")
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, "xgboost_model.joblib")
    st.success("Le modèle a été bien réentraîné et sauvegardé dans le fichier 'modele_xgboost.joblib")

# Page: Faire des Prédictions
def page_faire_des_predictions():
    st.title("Faire des Prédictions")
    st.write("### Prédictions sur de nouvelles données en temps réel")
    st.write("Testez le modèle avec de nouvelles données fournies par le fichier test.")
    # Chargement du modèle préentrainé fichier joblib du modèle
    xgboost_model = load_model()
    st.success("**Le modèle est prèt pour faire des prédictions**")
    # Création d'une liste vide pour stocker les colonnes sélectionnées pour tester le modèle
    corr = df_train_copy.corr()
    colonnes_selectionnees = []
    # Seuil de corrélation : les colonnes autour de 0 (-0.03 à 0.03) ne seront pas sélectionnées
    seuil = 0.03
    for index, row in corr[['Churn.Value']].iterrows():
        if abs(row['Churn.Value']) > seuil:
            colonnes_selectionnees.append(index)
    X_test, y_test = selection_colonnes_test(df_test_copy, colonnes_selectionnees)
    # Faire des prédictions sur les données test
    if st.button("Prédire"):
        y_pred = xgboost_model.predict(X_test) # Prédictions des classes
        y_pred_prob = xgboost_model.predict_proba(X_test)  # Probabilité associée à la prédiction de la classe
        # Combiner les résultats dans un DataFrame pour une meilleure lisibilité
        # Récupérer et ajouter l'index de df1 dans df2
        customerID = y_test.index.to_series().reset_index(drop=True)
        resultats = pd.DataFrame({
            "customerID": customerID,  # Index de la ligne
            #"Probabilite_Classe_0": y_pred_prob[:, 0],  # Probabilité pour la classe 0
            "churn probability": y_pred_prob[:, 1],   # Probabilité pour la classe 1
            "churn value": y_pred
            })
        resultats = resultats.set_index("customerID")
        st.write(f"Voici le résultat des {len(y_pred)} prédictions sur les données test avec la valeur de la probabilité par classe 1: \n", resultats)
        
        st.info("### **Evaluation du modèle**")
        st.write("**Les métrics suivants sont utilisés pour évaluer les performances du modèle**")
        st.info("""A savoir : **VP** : Vrais positifs (correctement prédits comme positifs). \n **VN** : Vrais négatifs (correctement prédits comme négatifs).
                   **FP** : Faux positifs (prédits positifs à tort). \n **FN** : Faux négatifs (ratés comme positifs).""")
        st.write(" - **Précision** = Parmi les positifs prédits, combien sont corrects ? \n _Focus sur la qualité des prédictions positives._")
        st.write(" - **Rappel** = Parmi les positifs réels, combien sont détectés ? \n _Focus sur la capacité de capturer les vrais positifs._")
        st.write(" - **Score F1** = Trouvons un équilibre entre précision et rappel. \n _Quand ni FP ni FN ne doivent dominer._")
        st.write(" - **Matrice de confusion** = Où est-ce que mon modèle se trompe ? \n _Un tableau clair pour décomposer les erreurs._")
        st.write(f"Précision du modèle :\n {accuracy_score(y_test, y_pred)}")
        st.write(f"Rappel du modèle :\n {recall_score(y_test, y_pred)}")
        st.write(f"Score F1 du modèle :\n {f1_score(y_test, y_pred)}")
        st.write(f"Matrice de confusion du modèle :")
        st.write(confusion_matrix(y_test, y_pred))
        st.success("Le modèle affiche une performance extrême (100 pour 100) de taux de réussite pour toutes les métriques sur les données nouvelles prédites.") 
        

# Page: Documentation du projet
def page_documentation_du_projet():
    st.title("Documentation du projet")
    st.write("### Guide de l'utilisateur")
    st.info("Pour un projet en production, le guide devrait être ajouté")
    

# Main app
def main():
    st.set_page_config(page_title="Présentation du projet", layout="wide")

    # Sidebar navigation
    page = st.sidebar.radio("Navigation", [
        "Accueil", 
        "Informations", 
        "Exploration des données", 
        "Ingénieurie de fonctionnalités", 
        "Visualisation des données", 
        "Développement de modèles", 
        "Faire des Prédictions", 
        "Documentation du projet"
    ])

    # Display the selected page
    if page == "Accueil":
        page_accueil()
    elif page == "Informations":
        page_informations()
    elif page == "Exploration des données":
        page_exploration_des_donnees()
    elif page == "Ingénieurie de fonctionnalités":
        page_transformation_des_donnees()
    elif page == "Visualisation des données":
        page_visualisation_des_donnees()
    elif page == "Développement de modèles":
        page_developpement_de_modeles()
    elif page == "Faire des Prédictions":
        page_faire_des_predictions()
    elif page == "Documentation du projet":
        page_documentation_du_projet()

if __name__ == "__main__":
    main()
