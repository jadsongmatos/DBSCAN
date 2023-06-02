# ATIVIDADE PRÁTICA – DBSCAN

1. lê um arquivo CSV chamado "RICE.csv" em um DataFrame pandas, descartando linhas com valores ausentes.

2. usa o LabelEncoder do sklearn para converter variáveis categóricas em numéricas. Isso é necessário, pois muitos algoritmos de aprendizado de máquina funcionam apenas com entradas numéricas.

3. Traça um gráfico de dispersão de duas variáveis 'RF(mm)' e 'MaxT' codificadas por cores por localização.

4. Os dados são então normalizados usando StandardScaler, que dimensiona recursos para ter média zero e variação de unidade.

5. O clustering DBSCAN é aplicado aos dados dimensionados. DBSCAN é um algoritmo de clustering baseado em densidade que pode descobrir clusters de forma arbitrária e é robusto para outliers.

6. O gráfico de dispersão é criado novamente, mas desta vez codificado por cores por atribuições de cluster.

7. O conjunto de dados (df) é então dividido em um conjunto de treinamento e um conjunto de teste.

8. Um classificador de árvore de decisão é então treinado nos dados de treinamento. Em seguida, imprime o tempo de treinamento.

9. O modelo de árvore de decisão treinado é então exportado como um objeto Graphviz que permite sua visualização.

10. Finalmente, a precisão de treinamento e teste do modelo é calculada e impressa.

## Dados

Os dados foram obtidos do Kaggle: [https://www.kaggle.com/datasets/zsinghrahulk/rice-pest-and-diseases](https://www.kaggle.com/datasets/zsinghrahulk/rice-pest-and-diseases)

#### Gráfico da árvore de decisão

![Decision Tree](Source.gv.min.svg)

## Referências

```py
@article{scikit-learn,
  title={Scikit-learn: Machine Learning in {P}ython},
  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}

@misc{kaggleDiabetesPrediction,
    author={MOHAMMED MUSTAFÁ},
    title={Diabetes prediction dataset - kaggle.com},
    url={https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset},
    journal={Kaggle},
    note={[Accessed 27-May-2023]},
}
```