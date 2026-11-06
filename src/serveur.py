from flask import Flask, render_template, request
import modeles.avion as mod
modele = mod.ModeleVol("donnees/Airlines.csv")
modele.entrainer()
print("_liste_airlines",modele._liste_airlines)
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    resultat = None

    if request.method == 'POST':
        try:
            # Récupération des données du formulaire
            airline_choisie = request.form.get('airline')
            day_of_week = float(request.form.get('day_of_week'))
            heure = float(request.form.get('heure'))
            minutes = float(request.form.get('minutes'))
            length = float(request.form.get('length'))

            # Appel de la fonction main du fichier avion.py
            prob = modele.predire(airline_choisie, day_of_week, heure, minutes, length)

            resultat = {
                "a_l_heure": round(prob[0] * 100, 2),
                "retarde": round(prob[1] * 100, 2)
            }

        except Exception as e:
            print("Erreur :", e)
            resultat = {"erreur": "Veuillez entrer des valeurs valides."}

    # Utilisation de la liste des compagnies partagée depuis avion.py
    return render_template('index.html', airlines=modele._liste_airlines, resultat=resultat)

@app.route('/historique', methods=['GET', 'POST'])
def historique():
    print("hello world")
    return "Page d'historique en construction"

if __name__ == '__main__':
    app.run(debug=True)
