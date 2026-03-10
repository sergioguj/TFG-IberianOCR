import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import pandas as pd
import pruebaTokenExcel 

#importamos la variable filteredTokens
tokens = pruebaTokenExcel.filteredTokens


#crear la gráfica de barras
def plotFrecuencias(tokensFrecuencias):

    tokens, freqs = zip(*tokensFrecuencias[:50])  # Los 50 tokens más frecuentes

    plt.figure(figsize=(10,6))
    plt.bar(tokens, freqs, color='purple')
    plt.xlabel('Tokens')
    plt.ylabel('Frecuencia')
    plt.title('Frecuencia de los tokens más comunes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#Llama a la función para hacer el gráfico
plotFrecuencias(tokens)
