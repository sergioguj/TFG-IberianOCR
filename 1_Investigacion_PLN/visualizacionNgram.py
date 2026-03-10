import nltk
from nltk.util import ngrams
from collections import Counter
import pruebaTokenExcel
import matplotlib.pyplot as plt
from matplotlib import rcParams

tokens = pruebaTokenExcel.allTokens

# Generar bigramas
n = 2  # tamaño
bigrams = list(ngrams(tokens, n))

# Contar la frecuencia de los N-gramas
frecuencia_bigrams = Counter(bigrams)

# Ordenar por frecuencia
bigrams_mas_comunes = frecuencia_bigrams.most_common(20)  # Top X más comunes

# Obtener todos los bigramas (sin limitar el número)
todos_bigrams = frecuencia_bigrams.items()


# Preparar los datos para la gráfica
ngram_strings = [' '.join(ngram) for ngram, _ in bigrams_mas_comunes]  # Convierte los bigramas a cadenas
frecuencias = [freq for _, freq in bigrams_mas_comunes]  # Extrae las frecuencias

# Configuración de fuentes
rcParams['font.family'] = 'DejaVu Sans'  # Usa una fuente estándar para textos generales

# Visualización gráfica
plt.figure(figsize=(12, 8))  # Aumentar el tamaño de la figura para una mejor visualización
bars = plt.bar(ngram_strings, frecuencias, color='orange')  # Crear un gráfico de barras

# Etiquetas y títulos (fuente estándar)
plt.xlabel('Bigramas', fontsize=14)  # Etiqueta del eje X con mayor tamaño
plt.ylabel('Frecuencia', fontsize=14)  # Etiqueta del eje Y con mayor tamaño
plt.title(f'Top {n}-gramas más comunes', fontsize=16, fontweight='bold')  # Título más grande y negrita

# Rotar las etiquetas para mejor legibilidad
plt.xticks(rotation=45, ha='right', fontsize=12)  # Aumentar el tamaño de las etiquetas de los ejes X

# Aplicar la fuente Iberian solo a los bigramas en las etiquetas del eje X
for tick in plt.gca().get_xticklabels():
    tick.set_fontname('iberian')  # Aplica la fuente Iberian a las etiquetas de bigramas

# Mostrar los valores encima de las barras (en una fuente estándar)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom', fontsize=12, fontname='DejaVu Sans')

# Ajustar el layout para que no se corten las etiquetas
plt.tight_layout()

# Mostrar el gráfico
plt.show()
