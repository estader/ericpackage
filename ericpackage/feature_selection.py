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
from sklearn.linear_model import LinearRegression
from feature_engine.selection import DropFeatures, DropConstantFeatures
from feature_engine.selection import DropDuplicateFeatures, DropCorrelatedFeatures, SmartCorrelatedSelection
from feature_engine.selection import SelectByShuffling, SelectBySingleFeaturePerformance
from feature_engine.selection import RecursiveFeatureElimination, RecursiveFeatureAddition

def fs_general(X_train, y_train, lista_features, modo, forma, valor):
    """
    
    
    """
    if forma == 'DropFeatures':
        transformer = DropFeatures(features_to_drop=lista_features)
        transformer.fit(X_train)

    elif forma == 'DropConstantFeatures':
        transformer = DropConstantFeatures(tol=valor, variables = lista_features,
                                           missing_values='ignore')
        transformer.fit(X_train)
        
    elif forma =='DropDuplicateFeatures':    
        transformer = DropDuplicateFeatures(variables=lista_features, missing_values='ignnore')
        transformer.fit(X_train)
        
    elif forma =='DropCorrelatedFeatures':
        transformer = DropCorrelatedFeatures(variables=lista_features, method='pearson', threshold=valor)
        transformer.fit(X_train)
        
    elif forma =='SmartCorrelatedSelection':    
        transformer = SmartCorrelatedSelection(variables = lista_features,
                                               method="pearson",
                                               threshold=valor,
                                               missing_values="raise",
                                               selection_method="variance",
                                               estimator=None)
        transformer.fit(X_train)
        
    elif forma == 'SelectByShuffling':
        if modo == 'regression':
            transformer = SelectByShuffling(variables = lista_features,
                                            estimator=RandomForestRegressor(),
                                            scoring="r2", random_state = 0)
        else:
            transformer = SelectByShuffling(variables = lista_features,
                                            random_state = 0)
        transformer.fit(X_train, y_train)
        
    elif forma =='SelectBySingleFeaturePerformance':
        if modo =='regression':
            transformer = SelectBySingleFeaturePerformance(estimator=RandomForestRegressor(),
                                                   scoring="r2",threshold=0.01)
        else:
            transformer = SelectBySingleFeaturePerformance(threshold=0.01)
        transformer.fit(X_train, y_train)
        
    # elif forma == 'SelectByTargetMeanPerformance':
    #     sel = SelectByTargetMeanPerformance(
    #                                         variables=None,
    #                                         scoring="roc_auc_score",
    #                                         threshold=0.6,
    #                                         bins=3,
    #                                         strategy="equal_frequency",
    #                                         cv=2,# cross validation
    #                                         random_state=1, #seed for reproducibility
    #                                     )

    #     sel.fit(X_train, y_train)
    elif forma =='RFE':
        transformer = RecursiveFeatureElimination(estimator = RandomForestRegressor(),
                                                variables = lista_features,
                                                scoring ="r2", cv=3)
        transformer.fit(X_train, y_train)
    elif forma =='RFA':
        
        transformer = RecursiveFeatureAddition(estimator=RandomForestRegressor(),
                                           variables=lista_features,
                                           scoring="r2", cv=3)
        transformer.fit(X_train, y_train)
        
    train_t = transformer.transform(X_train)
    features_selecionadas = list(train_t.columns)
    return train_t, features_selecionadas


def fs_variancia(X_train,
                 ths_var,
                 ver_features_selecionadas=False):
    """
        Seleção das Features a partir de um threshold estabelecido
    """
    print('Número de colunas antes do filtro: ' + str(len(X_train.columns)))
    print('Ths. recebido:' + str(ths_var))

    sel = VarianceThreshold(threshold=ths_var)
    sel.fit(X_train)
    if ver_features_selecionadas == True:
        print('Features selecionadas:')
        print(X_train.columns[sel.get_support()])
    else:
        pass
    X_train = X_train[X_train.columns[sel.get_support()]]

    print('Número de colunas depois do filtro: ' + str(len(X_train.columns)))
    return X_train, X_train.columns


def fs_correlacao_entre_features(dataset,
                                                threshold,
                                                ver_matriz=False):
    """
        Seleção de features pela correlação entre features, retirando as redundâncias
    """

    col_corr = set()  # Conjunto de todas as features correlacionadas

    corr_matrix = dataset.corr()
    if ver_matriz == True:

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        f, ax = plt.subplots(figsize=(11, 9))

        # Gerando o colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Heatmap com a matriz de correlação
        sns.heatmap(corr_matrix,
                    mask=mask,
                    cmap=cmap,
                    vmax=.3,
                    center=0,
                    square=True,
                    linewidths=.5,
                    cbar_kws={"shrink": .5})

    # print(corr_matrix)
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            # comparação valor absoluto do coeficiente
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]  # pegando o nome da coluna
                col_corr.add(colname)
    return col_corr





def fs_kbest(X_train, X_val, target_train, target_val):
    """
    
    """
    
    # Processo de escolha de K:
    k_vs_score1 = []
    k_vs_score2 = []
    for k in range(2, len(X_train.columns), 1):
        sel = SelectKBest(score_func=f_regression, k=k)
    
        Xtrain2 = sel.fit_transform(X_train, target_train)
        Xval2 = sel.transform(X_val)
    
        mdl = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)
        mdl.fit(Xtrain2, target_train)
    
        p = mdl.predict(Xval2)
    
        score = mean_absolute_error(target_val, p)
    
        print("k = {} - MAE = {}".format(k, score))
    
        k_vs_score1.append(score)
        k_vs_score2.append([k, score])
        
    # Gráfico
    # Erro pelo número de features:
    pd.Series(k_vs_score1, index=range(2,len(X_train.columns),1)).plot(figsize=(10,7))
    
    df_erro = pd.DataFrame(k_vs_score2, columns=['k','score'])
    df_erro['dif']= abs(df_erro['score'] - df_erro['score'].shift(1))
    df_erro['dif'] = df_erro['dif'].fillna(10000)
    
    df_erro = df_erro[df_erro.dif < 0.05 ]
    df_erro.head(30)
    
    k_max = int(df_erro['k'].max()) + 1
    k_min = int(df_erro['k'].min())
    k_medio = (k_max + k_min) / 2
    k_median = df_erro['k'].median()
    print('k_max: ' + str(k_max))
    print('k_min: ' + str(k_min))
    print('k_medio: ' + str(k_medio))
    print('k_median: ' + str(k_median))
    
    # Recuperando as features:
    
    # É importante procurar no gráfico um região de vale que seja estável.
    # A seleção de features irá funcionar melhor com gráficos em formato de U
    
    print(k_medio)
    
    sel = SelectKBest(score_func=f_regression, k=int(k_medio))
    Xtrain2 = sel.fit_transform(X_train, target_train)
    mask = sel.get_support()
    features_selecionadas_kbest = set(X_train.columns[mask])
    
    print('Features selecionadas:')
    print(features_selecionadas_kbest)
    
    return features_selecionadas_kbest

def fs_fromModel(estimador, X_train, X_val, target_train, target_val, k_max, k_min):
    """
    
    """
    
    # Seleção da feature a partir do teste de diferentes
    # combinações em diferentes quantidades,
    # em um algoritmo de Machine Learning
    # XGBOOST regressor foi o escolhido
    
    fs = []
    for seed in range(0, 30):
    
        np.random.seed(seed)
        # Escolhido a faixa de quantidade de features que devem ser selecionadas
        k = np.random.randint(k_min, k_max, 1)[0]
        selected = np.random.choice(X_train.columns, k, replace=False)
    
        Xtrain2 = X_train[selected]
        Xval2 = X_val[selected]
    
        mdl = XGBRegressor(eval_metric='rmse',
                           max_depth=10,
                           reg_alpha=0.1,
                           subsample=0.8,
                           random_state=0,
                           verbosity=1)
        
        mdl.fit(Xtrain2, target_train)
    
        p = mdl.predict(Xval2)
    
        score = mean_absolute_error(target_val, p)
    
        print("seed = {} - k = {} - MAE = {}".format(seed, k, score))
        fs.append([seed, k, score])
        
    # Escolher k e seed a partir do menor MAE observado
    
    df_erro = pd.DataFrame(fs, columns=['seed', 'k', 'score'])
    df_menor_erro = df_erro.sort_values(['score', 'k'], ascending=[True, False])
    k = df_menor_erro.iloc[0, 1]
    seed = df_menor_erro.iloc[0, 0]
    print(k)
    print(seed)
    
    np.random.seed(seed)
    selected_pelo_modelo = set(np.random.choice(X_train.columns, k, replace=False))
    
    print('Features selecionadas:')
    print(selected_pelo_modelo)
    
    return selected_pelo_modelo


def fs_common_or_union(features_selecionadas_1, features_selecionadas_2):
    """
        Operações de conjuntos para unificar a seleção de features dos
        diferentes métodos:

    """

    features_sel_and = features_selecionadas_1 & features_selecionadas_2
    features_sel_all = features_selecionadas_1 | features_selecionadas_2
    print(features_sel_and)
    print()
    print('Quantidade de features em comum: ' + str(len(features_sel_and)))
    print()
    print(features_sel_all)
    print()
    print('Quantidade de features em comum: ' + str(len(features_sel_all)))
    return features_sel_and, features_sel_all
    
def fs_feature_importance():
    return
