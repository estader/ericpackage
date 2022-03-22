# Funções de preparação e modelagem de dados

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
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

def qc_fillna_custom(X_train, lista_variaveis, forma, valor = None):
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
 

def qc_duplicatas(df, y):
    print('Número de registros antes da filtragem:' + str(len(df)))
    df.drop_duplicates(keep='first', inplace=True) 
    print('Número de registros depois da filtragem:' + str(len(df)))
    dfy = pd.concat([df,y],axis=1, join='inner')
    return df, dfy.iloc[:,-1]

