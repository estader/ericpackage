# Imports

import pandas as pd
import numpy as np



def dc_nulos(dados):
    lista=[]
    for i in range(len(dados.columns)):
        coluna = dados.iloc[:,i].name
        tipo = dados.iloc[:,i].dtypes
        nulos = dados.iloc[:,i].isnull().sum()
        percent_nulos = (dados.iloc[:,i].isnull().sum())/len(dados)
        lista.append([coluna, tipo, nulos, percent_nulos])

    tabela = pd.DataFrame(lista, columns=['coluna','dtype','nulos', 'percent_nulos'])
    tab_cat = tabela[(tabela.dtype == 'object') & (tabela.nulos > 0)]
    tab_num = tabela[((tabela.dtype =='int64') | (tabela.dtype =='float64')) & (tabela.nulos > 0)]

    lista_cat_nulos = list(tab_cat.coluna)
    lista_num_nulos = list(tab_num.coluna)
    return tabela, lista_cat_nulos, lista_num_nulos

def dc_features_tipos(dados):
    lista=[]
    for i in range(len(dados.columns)):
        coluna = dados.iloc[:,i].name
        tipo = dados.iloc[:,i].dtypes
        lista.append([coluna, tipo])

    tabela = pd.DataFrame(lista, columns=['coluna','dtype'])
    tab_cat = tabela[(tabela.dtype == 'object')]
    tab_num = tabela[(tabela.dtype =='int64') | (tabela.dtype =='float64')]

    lista_cat = list(tab_cat.coluna)
    lista_num = list(tab_num.coluna)
    return tabela, lista_cat, lista_num