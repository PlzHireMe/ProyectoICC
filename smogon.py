import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# 1) Investigar cómo funciona la clase TfidfVectorizer y usarla en reemplazo de CountVectorizer para Generar la matriz tf-idf utilizando una cantidad de n-gramas elegida por usted (unigramas, bigramas, trigramas, etc.).
csv = pandas.read_csv('smogon.csv')

stopWords = pandas.read_csv('stopwords.csv')
listaStopWords = stopWords["stopWords"].tolist()

vec = TfidfVectorizer(ngram_range=(1, 2), stop_words= listaStopWords)
x = vec.fit_transform(csv["moves"])

# 2) Mostrar el número total de columnas que tiene su matriz tf-idf

print("Mostrar el número total de columnas que tiene su matriz tf-idf: ")
print(len(vec.get_feature_names_out()))

# 3) Imprimir todos los tokens (elementos de su vocabulario)

print("Imprimir todos los tokens (elementos de su vocabulario)")
print(vec.get_feature_names_out())

# 4) Generar un DataFrame que tenga como cabecera los elementos del vocabulario.

data_Frame = pandas.DataFrame(x.toarray(), columns=vec.get_feature_names_out())
data_Frame.insert(0, "Pokemon", csv["Pokemon"])
print("Generar un DataFrame que tenga como cabecera los elementos del vocabulario.")
print(data_Frame)

# 5) Agrupar las filas de nuestro nuevo cluster.

km = KMeans(n_clusters=8, n_init=40)
lista = km.fit_predict(data_Frame.drop(columns=["Pokemon"]))
data_Frame["Cluster"] = lista
#Ordenar por Clusters
data_Frame.sort_values(["Cluster"], inplace=True) 
data_Frame.reset_index(drop=True, inplace= True)
print("Agrupar las filas de nuestro nuevo cluster.")
print(data_Frame)

# 6) Crear un csv, con los nombres y el cluster al que pertenecen.

nuevo_csv = pandas.DataFrame()
nuevo_csv["Pokemon"] = data_Frame["Pokemon"]
nuevo_csv["Cluster"] = data_Frame["Cluster"]
nuevo_csv.to_csv('csv_clusters')

# 7) Nombrar los grupos

dictionary_caracteristicas = {
    0: "Pokemon electricos, o hechos de un material resistente: Acero, Roca, Hielo",
    1: "Magicos: psiquicos, siniestros y fantasmas",
    2: "Acuaticos",
    3: "Voladores",
    4: "Plantas",
    5: "Terrestres, de fuego y luchadores",
    6: "Insectos",
    7: "Legendarios"
}

nuevo_csv = pandas.read_csv("csv_cluster_copia")
nuevo_csv_caracteristicas = nuevo_csv
nuevo_csv_caracteristicas["Descripcion"] = nuevo_csv["Cluster"].map(dictionary_caracteristicas)
nuevo_csv_caracteristicas.to_csv("Pokemon_Cluster_Descripcion")

#Segunda parte:

tipos = ["fire", "water", "grass", "electric", "ice", "fighting", "poison", "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy"]

#Limpiar csv:

print("-----------------------------------------")

nuevo_move = []
for sentence in csv["moves"]:
    sentence = sentence.lower()
    for tipo in tipos:
        sentence = sentence.replace(tipo, f' {tipo} ')
    nuevo_move.append(' '.join(sentence.split()))
csv["moves"] = nuevo_move

vec_parte2 = TfidfVectorizer(vocabulary=tipos)
x = vec_parte2.fit_transform(csv["moves"])

km2 = KMeans(n_clusters=15, n_init=30)
lista = km2.fit_predict(x)

csv_parte2 = pandas.DataFrame(csv["Pokemon"])
csv_parte2["cluster"] = lista

csv_parte2.sort_values(["cluster"], inplace=True)
csv_parte2.reset_index(drop=True, inplace= True)

print(csv_parte2)


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

nuevo_csv.sort_values(["Pokemon"], inplace=True) 
nuevo_csv["Tipo Principal"] = frequent_moves
nuevo_csv["Tipo Secundario"] = frequent_moves2
nuevo_csv.sort_values(["Cluster"], inplace=True) 

csv_parte2.sort_values(["Pokemon"], inplace=True) 
csv_parte2["Tipo Principal"] = frequent_moves
csv_parte2["Tipo Secundario"] = frequent_moves2
csv_parte2.sort_values(["cluster"], inplace=True) 

print(nuevo_csv)

print(csv_parte2)


