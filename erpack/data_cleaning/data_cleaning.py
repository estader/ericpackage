# Funções de preparação e modelagem de dados

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from feature_engine.imputation import MeanMedianImputer
from feature_engine.imputation import ArbitraryNumberImputer
from feature_engine.imputation import EndTailImputer
from feature_engine.imputation import CategoricalImputer
from feature_engine.imputation import RandomSampleImputer
from feature_engine.imputation import AddMissingIndicator
from feature_engine.encoding import OneHotEncoder
from feature_engine.encoding import CountFrequencyEncoder
from feature_engine.encoding import OrdinalEncoder
from feature_engine.encoding import MeanEncoder
from feature_engine.encoding import DecisionTreeEncoder
from feature_engine.encoding import RareLabelEncoder
from feature_engine import transformation as vt
from reg_resampler import resampler
from imblearn.over_sampling import SMOTE
from scipy.special import boxcox, inv_boxcox, boxcox1p, inv_boxcox1p
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import  AdaBoostRegressor, GradientBoostingRegressor
import time
from sklearn.metrics import r2_score

def dc_fillna_custom(X_train, lista_variaveis, forma, valor = None):
    """
        Preenchimento de Missing:
            -> Média
            -> Mediana
            -> Arbitrário
            -> Endtail
            -> Categórico
            -> Aleatório
            -> Indicadores de Missing
    """
    print('Método aplicado:')
    print(forma)
    print('Imputação nas variáveis:')
    print(lista_variaveis)
    
    if len(lista_variaveis)!=0:

        if forma == 'mean':
            imputer = MeanMedianImputer(imputation_method='mean', variables=lista_variaveis)       
        elif forma == 'median':
            imputer = MeanMedianImputer(imputation_method='median', variables=lista_variaveis)
        elif forma == 'arbitrary':
            imputer = ArbitraryNumberImputer(arbitrary_number = valor, variables=lista_variaveis)
        elif forma == 'endtail':
            imputer = EndTailImputer(imputation_method='gaussian', tail='right',
                                            fold=3, variables=lista_variaveis)
        elif forma == 'categorical':
            imputer = CategoricalImputer(variables=lista_variaveis)
        elif forma == 'RandomSample':
            imputer = RandomSampleImputer(random_state=lista_variaveis,
                                            seed='observation',seeding_method='add')
        elif forma == 'missing_indicator':
            imputer = AddMissingIndicator( variables=lista_variaveis)

        imputer.fit(X_train)              
        train_t = imputer.transform(X_train)

        return train_t, imputer

    else:
        return 

    
    

def dc_duplicatas(df, y):
    print('Número de registros antes da filtragem:' + str(len(df)))
    df.drop_duplicates(keep='first', inplace=True) 
    print('Número de registros depois da filtragem:' + str(len(df)))
    dfy = pd.concat([df,y],axis=1, join='inner')
    return df, dfy.iloc[:,-1]

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