import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

class model():
    def __init__(self):

    def train():
        try:
            df = pd.read_csv('Airlines.csv')
            print("--- Dataset 'Airlines.csv' chargé avec succès ---")
        except FileNotFoundError:
            print("Erreur : Le fichier 'Airlines.csv' est introuvable dans le dossier.")
            exit()

        # Encodage de la compagnie aérienne (texte -> numéro)
        le = LabelEncoder()
        df['Airline_Encoded'] = le.fit_transform(df['Airline'])

        # Création de la 6ème variable (carré de la durée)
        df['Length_Carre'] = df['Length'] ** 2

        # Nos 5 variables numériques finales (on remplace Flight par Airline_Encoded)
        features = ['Airline_Encoded', 'DayOfWeek', 'Time', 'Length', 'Length_Carre']
        target = 'Delay'

        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        knn.fit(X_train_scaled, y_train)

        # Liste des compagnies uniques pour les afficher dans le menu déroulant du site
        liste_airlines = sorted(df['Airline'].unique())

    def predict():
        time_vol = (heure * 60) + minutes
        length_carre = length ** 2
        
        airline_encoded = le.transform([airline_choisie])[0]
        
        nouveau_vol = np.array([[airline_encoded, day_of_week, time_vol, length, length_carre]])
        nouveau_vol_scaled = scaler.transform(nouveau_vol)
        
        probas = knn.predict_proba(nouveau_vol_scaled)[0]
        return probas  # Renvoie un tableau [proba_a_l_heure, proba_retard]
# ---------------------------------------------------------
# CHARGEMENT ET ENTRAÎNEMENT DU MODÈLE
# ---------------------------------------------------------


print("--- Modèle k-NN prêt avec les compagnies aériennes ! ---")
# Extrait de ce que doit contenir avion.py :
def main(airline_choisie, day_of_week, heure, minutes, length):
    