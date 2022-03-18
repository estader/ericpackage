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


def fe_categorical_transform(X_train, y_train, lista_variaveis, forma, valor=None):
    """
        Tipos de transformações categóricas:
            -> One Hot Encoding
            -> CountFrequency
            -> Ordinal
            -> Mean
            -> DecisionTree
            -> RareLabel
    """
    if forma == 'ohe':
        print(lista_variaveis)
        encoder = OneHotEncoder(top_categories = None ,
                                variables = lista_variaveis,  drop_last = True)
        encoder.fit(X_train)
    elif forma == 'CountFrequency':
        print(lista_variaveis)
        encoder = CountFrequencyEncoder(encoding_method='count',
                                        variables = lista_variaveis)
        encoder.fit(X_train)
    elif forma == 'Ordinal':
        print(lista_variaveis)
        encoder = OrdinalEncoder(encoding_method='ordered',
                                 variables=lista_variaveis)
        encoder.fit(X_train, y_train)
    elif forma == 'Mean':
        print(lista_variaveis)
        encoder = MeanEncoder(variables=lista_variaveis )
        encoder.fit(X_train, y_train)
    elif forma == 'DecisionTree':
        print(lista_variaveis)
        encoder = DecisionTreeEncoder(variables=lista_variaveis , random_state=0)
        encoder.fit(X_train, y_train)
    elif forma =='RareLabel' :
        print(lista_variaveis)
        encoder = RareLabelEncoder(tol=valor, n_categories=1,
                                   variables=lista_variaveis,replace_with='Rare')
        encoder.fit(X_train)

         
    train_t = encoder.transform(X_train)

    
    return train_t, encoder, lista_variaveis


def fe_numerical_transform(X_train, lista_variaveis, forma, valor = None):
    """
        Tipos de transformações Numéricas:
            -> Log
            -> Reciprocal
            -> Power
            -> BoxCox
            -> YeoJohnson
    """
    
    lista_num = list(X_train.select_dtypes(include=['int64', 'float64']).columns)
    lista_variaveis = [i for i in lista_variaveis if i in lista_num]
    
    if forma == 'Log':
        treino = X_train[lista_variaveis]
        lista_variaveis = [ treino.iloc[:,i].name for i in range(len(treino.columns)) if (treino.iloc[:,i] > 0).all()]
        print(lista_variaveis)
        tf = vt.LogTransformer(variables = lista_variaveis)
    elif forma == 'Reciprocal':
        treino = X_train[lista_variaveis]
        lista_variaveis = [ treino.iloc[:,i].name for i in range(len(treino.columns)) if (treino.iloc[:,i] != 0).all()]
        print(lista_variaveis)
        tf = vt.ReciprocalTransformer(variables = lista_variaveis)
    elif forma == 'Power':
        print(lista_variaveis)
        tf = vt.PowerTransformer(variables = lista_variaveis, exp = valor)
    elif forma == 'BoxCox':
        treino = X_train[lista_variaveis]
        lista_variaveis = [ treino.iloc[:,i].name for i in range(len(treino.columns)) if (treino.iloc[:,i] > 0).all()]
        print(lista_variaveis)
        tf = vt.BoxCoxTransformer(variables = lista_variaveis)
    elif forma =='YeoJohnson':
        print(lista_variaveis)
        tf = vt.YeoJohnsonTransformer(variables = lista_variaveis)
        
    tf.fit(X_train)        
    X_train2 = tf.transform(X_train)
        
    return X_train2, tf, lista_variaveis

def original_e_transformada(X_train,X_train2, lista_variaveis,sufixo):
    if sufixo !='':
        X_train2 = pd.concat([X_train, X_train2[lista_variaveis].add_suffix('_'+sufixo)], axis = 1)
    else:
        X_train2 = pd.concat([X_train, X_train2[lista_variaveis]], axis = 1)
    return X_train2

def fe_resampler_regression(X_train,y_train, target):
    base = pd.concat([X_train,y_train],axis=1)
    # You might recieve info about class merger for low sample classes
    # Generate classes
    rs = resampler()
    Y_classes = rs.fit(base, target = target, bins=7, balanced_binning=False )
    # Create the actual target variable
    #Y = base[target]

    # Create a smote (over-sampling) object from imblearn
    smote = SMOTE(random_state=27)#, sampling_strategy={0:10, 1:5, 2:5, 3: 2})

    # Now resample
    X1_train, y_train = rs.resample(smote, base, Y_classes)
    return X1_train, y_train


