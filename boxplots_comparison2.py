import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


resultados_finais = np.loadtxt('resultados_finais.txt')
resultados_finais_stochastic = np.loadtxt('resultados_finais_stochastic.txt')
resultados_finais_um_ponto = np.loadtxt('resultados_finais_ag_um_ponto.txt')
resultados_finais_dois_pontos = np.loadtxt('resultados_finais_ag_dois_pontos.txt')
resultados_finais_uniforme = np.loadtxt('resultados_finais_ag_uniforme.txt')

plt.figure(figsize=(10, 6))
plt.boxplot([resultados_finais, resultados_finais_stochastic, resultados_finais_um_ponto, resultados_finais_dois_pontos, resultados_finais_uniforme], labels=["Standard", "Stochastic", "Um ponto", "Dois Pontos", "Uniforme"], patch_artist=True)
plt.title("Comparação: Hill Climbing Standard vs Stochastic vs AG Um Ponto vs AG Dois Pontos vs AG Uniforme")
plt.ylabel("Fitness final")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig('Boxplots_comparison.png')
plt.show()