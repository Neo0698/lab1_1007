from flask import Flask, render_template, request
import avion  # Votre module avion.py

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
            prob = avion.main(airline_choisie, day_of_week, heure, minutes, length)
            
            resultat = {
                "a_l_heure": round(prob[0] * 100, 2),
                "retarde": round(prob[1] * 100, 2)
            }
            
        except Exception as e:
            print("Erreur :", e)
            resultat = {"erreur": "Veuillez entrer des valeurs valides."}

    # Utilisation de la liste des compagnies partagée depuis avion.py
    return render_template('index.html', airlines=avion.liste_airlines, resultat=resultat)

@app.route('/historique', methods=['GET', 'POST'])
def historique():
    print("hello world")
    return "Page d'historique en construction"

if __name__ == '__main__':
    app.run(debug=True)