import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# ============================================================
# === CONFIGURAÇÃO DO PROBLEMA KNAPSACK (20 DIMENSÕES) =======
# ============================================================
GANHOS = [92, 4, 43, 83, 84, 68, 92, 82, 6, 44, 32, 18, 56, 83, 25, 96, 70, 48, 14, 58]
PESOS = [44, 46, 90, 72, 91, 40, 75, 35, 8, 54, 78, 40, 77, 15, 61, 17, 75, 29, 75, 63]
CAPACIDADE_MAXIMA = 878
DIM = 20

def avaliar(solucao):
    valor = sum(g * s for g, s in zip(GANHOS, solucao))
    peso = sum(p * s for p, s in zip(PESOS, solucao))
    valido = peso <= CAPACIDADE_MAXIMA
    if not valido:
        valor = 0
    return valor, peso, valido


# ============================================================
# =============== SELEÇÃO POR TORNEIO ========================
# ============================================================
def selecao(populacao, fitness, k=3):
    escolhidos = random.sample(range(len(populacao)), k)
    melhor = max(escolhidos, key=lambda i: fitness[i])
    return populacao[melhor]


# ============================================================
# === APLICAR CROSSOVER (UM, DOIS PONTOS, UNIFORME) ==========
# ============================================================
def aplicar_crossover(pai1, pai2, tipo='um_ponto', taxa_crossover=0.8):
    """
    Aplica crossover com probabilidade definida pela taxa.
    Tipos: 'um_ponto', 'dois_pontos', 'uniforme'
    """
    if random.random() < taxa_crossover:
        if tipo == 'um_ponto':
            ponto = random.randint(1, len(pai1) - 1)
            filho1 = pai1[:ponto] + pai2[ponto:]
            filho2 = pai2[:ponto] + pai1[ponto:]
        elif tipo == 'dois_pontos':
            p1 = random.randint(1, len(pai1) - 2)
            p2 = random.randint(p1 + 1, len(pai1) - 1)
            filho1 = pai1[:p1] + pai2[p1:p2] + pai1[p2:]
            filho2 = pai2[:p1] + pai1[p1:p2] + pai2[p2:]
        elif tipo == 'uniforme':
            filho1, filho2 = [], []
            for a, b in zip(pai1, pai2):
                if random.random() < 0.5:
                    filho1.append(a)
                    filho2.append(b)
                else:
                    filho1.append(b)
                    filho2.append(a)
        else:
            raise ValueError("Tipo de crossover inválido.")
        return filho1, filho2
    else:
        return pai1.copy(), pai2.copy()


# ============================================================
# === MUTAÇÃO BIT-FLIP =======================================
# ============================================================
def mutacao_bitflip(individuo, taxa=0.02):
    return [1 - gene if random.random() < taxa else gene for gene in individuo]


# ============================================================
# === EXECUÇÃO DO ALGORITMO GENÉTICO =========================
# ============================================================
def executar_ag(tipo_crossover='um_ponto', 
                tamanho_pop=50, geracoes=500, 
                taxa_crossover=0.8, taxa_mutacao=0.02, 
                elitismo=2, torneio_k=3, seed=None):

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    populacao = [[random.randint(0, 1) for _ in range(DIM)] for _ in range(tamanho_pop)]
    fitness = [avaliar(ind)[0] for ind in populacao]
    historico = []

    for geracao in range(geracoes):
        nova_populacao = []

        # Elitismo: mantém os melhores
        elite_idx = np.argsort(fitness)[-elitismo:]
        for i in elite_idx:
            nova_populacao.append(populacao[i].copy())

        # Trecho principal de cruzamento (seu código adaptado)
        while len(nova_populacao) < tamanho_pop:
            pai1 = selecao(populacao, fitness, k=torneio_k)
            pai2 = selecao(populacao, fitness, k=torneio_k)
            filho1, filho2 = aplicar_crossover(pai1, pai2, tipo=tipo_crossover, taxa_crossover=taxa_crossover)
            filho1 = mutacao_bitflip(filho1, taxa_mutacao)
            filho2 = mutacao_bitflip(filho2, taxa_mutacao)
            nova_populacao.extend([filho1, filho2])

        # Ajusta tamanho (caso ultrapasse)
        nova_populacao = nova_populacao[:tamanho_pop]
        populacao = nova_populacao
        fitness = [avaliar(ind)[0] for ind in populacao]
        historico.append(max(fitness))

    melhor_idx = np.argmax(fitness)
    melhor = populacao[melhor_idx]
    valor, peso, _ = avaliar(melhor)

    return {
        'melhor_valor': valor,
        'melhor_peso': peso,
        'melhor_individuo': melhor,
        'historico': historico
    }


# ============================================================
# === EXECUÇÃO MULTIPLAS E CONFIGURAÇÕES =====================
# ============================================================
CONFIGS = ["um_ponto", "dois_pontos", "uniforme"]
RUNS = 30
GERACOES = 500

resultados = {}
for config in CONFIGS:
    print(f"\nExecutando configuração: {config}")
    finais, historicos = [], []
    for r in range(RUNS):
        res = executar_ag(tipo_crossover=config, seed=r)
        finais.append(res['melhor_valor'])
        historicos.append(res['historico'])
       #if (r+1) % 5 == 0:
        print(f"  Execução {r+1}/{RUNS} -> melhor valor {res['melhor_valor']} (peso {res['melhor_peso']})")
    resultados[config] = {'finais': finais, 'historicos': historicos}
    data_array = np.array(finais)
    np.savetxt(f'resultados_finais_ag_{config}.txt', data_array, fmt='%.0f')
    print(f"\nResultados salvos para configuração {config}.\n")
    print(resultados[config]['finais'])


# ============================================================
# === ESTATÍSTICAS E GRÁFICO DE CONVERGÊNCIA =================
# ============================================================
estatisticas = []
for nome, dados in resultados.items():
    media = np.mean(dados['finais'])
    desvio = np.std(dados['finais'], ddof=1)
    estatisticas.append({'Configuração': nome, 'Média': media, 'Desvio Padrão': desvio})

df = pd.DataFrame(estatisticas).set_index('Configuração')
print("\n=== Estatísticas Finais ===")
print(df)

plt.figure(figsize=(10, 6))
for nome, dados in resultados.items():
    medias = np.mean(dados['historicos'], axis=0)
    desvios = np.std(dados['historicos'], axis=0)
    plt.plot(medias, label=nome)
    plt.fill_between(range(GERACOES), medias - desvios, medias + desvios, alpha=0.2)
plt.title("Convergência média por geração")
plt.savefig('Convergencia_AG.png')
plt.xlabel("Geração")
plt.ylabel("Melhor fitness")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.boxplot(
    [resultados[t]['finais'] for t in CONFIGS],
    labels=CONFIGS, patch_artist=True
)
plt.title("Boxplot dos resultados finais por configuração")
plt.savefig('Boxplots_AG.png')
plt.ylabel("Melhor Fitness final")
plt.grid(True)
plt.show()

