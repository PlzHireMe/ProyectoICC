import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 1) Investigar cómo funciona la clase TfidfVectorizer y usarla en reemplazo de CountVectorizer para Generar la matriz tf-idf utilizando una cantidad de n-gramas elegida por usted (unigramas, bigramas, trigramas, etc.).
csv = pandas.read_csv('smogon.csv')

stopWords = pandas.read_csv('stopwords.csv')
listaStopWords = stopWords["stopWords"].tolist()

vec = TfidfVectorizer(ngram_range=(1, 2), stop_words= listaStopWords)
x = vec.fit_transform(csv["moves"])
km = KMeans(n_clusters=8, n_init=40)
lista = km.fit_predict(x)

# 2) Mostrar el número total de columnas que tiene su matriz tf-idf

print("Mostrar el número total de columnas que tiene su matriz tf-idf: ")
print(len(vec.vocabulary_))

# 3) Imprimir todos los tokens (elementos de su vocabulario)

print(vec.vocabulary_)

csv_parte1 = pandas.DataFrame(csv["Pokemon"])
csv_parte1["cluster"] = lista

csv_parte1.sort_values(["cluster"], inplace=True)
csv_parte1.reset_index(drop=True, inplace= True)



#Segunda parte:

tipos = ["fire", "water", "grass", "electric", "ice", "fighting", "poison", "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy"]

#Limpiar csv:

print("-----------------------------------------")

nuevo_move = []
for sentence in csv["moves"]:
    for tipo in tipos:
        sentence = sentence.replace(tipo, f' {tipo} ')
    sentence = sentence.lower()
    nuevo_move.append(' '.join(sentence.split()))
csv["moves"] = nuevo_move

vec_parte2 = TfidfVectorizer(vocabulary=tipos)
x = vec_parte2.fit_transform(csv["moves"])

km2 = KMeans(n_clusters=15, n_init=30)
lista = km2.fit_predict(x)

csv_parte2 = pandas.DataFrame(csv["Pokemon"])
csv_parte2["cluster"] = lista

csv_parte3 = pandas.DataFrame()
csv_parte3["Pokemon"] = csv_parte2["Pokemon"]
csv_parte3["cluster"] = csv_parte2["cluster"]

csv_parte2.sort_values(["cluster"], inplace=True)
csv_parte2.reset_index(drop=True, inplace= True)


#Tercera Parte:

frequent_moves = []
frequent_moves2 = []

for sentence in csv["moves"]:
    counts = {}
    for tipo in tipos:
        counts[tipo] = sentence.count(tipo)
    ordenado = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    frequent_moves.append(ordenado[0])
    frequent_moves2.append(ordenado[1])

csv_parte3["Tipo Principal"] = frequent_moves
csv_parte3["Tipo Secundario"] = frequent_moves2
csv_parte3.sort_values(["cluster"], inplace=True)
csv_parte3.reset_index(drop=True, inplace= True)

