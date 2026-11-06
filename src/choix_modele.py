import model as mod
import matplotlib.pyplot as plt
CHEMIN_CSV = "donnees/Airlines.csv"
modele = mod.ModeleVol(CHEMIN_CSV)
test_values=range(20,150,5)
score_list=[]
score_list2=[]
modele.entrainer()
for el in test_values:
	print("___________________")
	score=modele.test_knn(n=el,weight=False)
	score_list.append(score)

	score2=modele.test_knn(n=el,weight=True)
	score_list2.append(score2)
	print(el,score, score2)


plt.figure(figsize=(10, 6))
plt.plot(test_values, score_list, marker='o', color='#2e7d32', linewidth=2)
plt.plot(test_values, score_list, marker='o', color='#8F00FF', linewidth=2)
# Personnalisation
plt.title("Score du modèle k-NN en fonction de k")
plt.xlabel("Valeur de k")
plt.ylabel("Score (exactitude)")
plt.grid(True, alpha=0.3)
plt.xticks(test_values)  # Affiche toutes les valeurs de k
plt.xlim(0, 100)
plt.ylim(0.5, 0.8)  # Ajustez selon vos scores réels
plt.tight_layout()

# Sauvegarde (obligatoire pour le TP)
plt.savefig("donnees/score_vs_k.png")
plt.show()
