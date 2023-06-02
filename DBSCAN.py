"""
ATIVIDADE PRÁTICA – DBSCAN
Jadson Goulart de Matos (21103270) e Luan Daniel de Oliveira Melo (20102404)
DEC0014-06655 (20231) - Inteligência Artificial e Computacional, UFSC
6 de junho de 2023
"""

import pandas as pd
import numpy as np
from sklearn import tree
import time
from sklearn.preprocessing import LabelEncoder
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load data from the "RICE.csv" file into a pandas dataframe
df = pd.read_csv("RICE.csv")


# Drop rows with missing values and reset the index
df = df.dropna().reset_index(drop=True)


# Use LabelEncoder to encode categorical variables to numeric values
collection_type = LabelEncoder()
pest_name = LabelEncoder()
location = LabelEncoder()

# Convert 'Location', 'PEST NAME', and 'Collection Type' columns from categorical to numerical using LabelEncoders
df['Location'] = collection_type.fit_transform(df['Location'])
df['PEST NAME'] = pest_name.fit_transform(df['PEST NAME'])
df['Collection Type'] = location.fit_transform(df['Collection Type'])

# Print the data types of the dataframe's columns to verify the transformations
print(df.dtypes)

# Plot the scatter plot between 'RF(mm)' and 'MaxT' variables, colored by 'Location'
plt.scatter(df['RF(mm)'], df['MaxT'], c=df['Location'], alpha=0.6)
plt.xlabel('RF(mm)')
plt.ylabel('MaxT')
plt.show()

# Normalizando os dados
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(df)
scaled_X_test = scaler.transform(df)

# Initialize DBSCAN object for clustering
dbscan = DBSCAN(eps=0.5, min_samples=5) 

# Perform clustering on the normalized data
dbscan.fit(scaled_X_train)

# Retrieve the cluster labels for each data point
cluster = np.array(dbscan.labels_)

# Visualizando os clusters
plt.scatter(df['RF(mm)'], df['MaxT'], c=cluster, alpha=0.6)
plt.xlabel('RF(mm)')
plt.ylabel('MaxT')
plt.show()

# Set up data and target for the decision tree model
X = df
Y = cluster

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create an instance of the DecisionTreeClassifier
model = tree.DecisionTreeClassifier(max_depth=22, min_samples_split=10, min_samples_leaf=5, max_leaf_nodes=22)

# Treinar o modelo de árvore de decisão usando os dados de treinamento
start = time.time()
model.fit(X_train, y_train)
tempo = time.time() - start
print(tempo)


# Export the decision tree as a Graphviz object
feature_names = ['Observation Year','Standard Week', 'Pest Value', 'Collection Type','MaxT', 'MinT', 'RH1(%)', 'RH2(%)', 'RF(mm)', 'WS(kmph)', 'SSH(hrs)', 'EVP(mm)','PEST NAME','Location']

unique_clusters = np.unique(cluster)
target_names = [str(i) for i in unique_clusters]

dot_data = tree.export_graphviz(model, out_file=None,
                                feature_names=feature_names,
                                class_names=target_names,
                                filled=True,
                                rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.view()

# Calcular a pontuação de precisão do modelo nos dados de teste
print(f'Training accuracy tree: {model.score(X_train, y_train)*100}%')
print(f'Testing accuracy tree: {model.score(X_test, y_test)*100}%')


