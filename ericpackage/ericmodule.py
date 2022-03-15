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

def fillna_custom(X_train, lista_variaveis, forma, valor = None):
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



def categorical_transform(X_train, X_test, y_train, lista_variaveis, forma, valor=None):
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
        encoder = OneHotEncoder(top_categories = lista_variaveis,
                                variables = None, drop_last=True)
        encoder.fit(X_train)
    elif forma == 'CountFrequency':
        encoder = CountFrequencyEncoder(encoding_method='frequency',
                                        variables=lista_variaveis)
        encoder.fit(X_train)
    elif forma == 'Ordinal':
        encoder = OrdinalEncoder(encoding_method='ordered',
                                 variables=lista_variaveis)
        encoder.fit(X_train, y_train)
    elif forma == 'Mean':
        encoder = MeanEncoder(variables=lista_variaveis )
        encoder.fit(X_train, y_train)
    elif forma == 'DecisionTree':
        encoder = DecisionTreeEncoder(variables=lista_variaveis , random_state=0)
        encoder.fit(X_train, y_train)
    elif forma =='RareLabel' :
        encoder = RareLabelEncoder(tol=0.03, n_categories=None,
                                   variables=lista_variaveis,replace_with='Rare')
        encoder.fit(X_train)

         
    train_t = encoder.transform(X_train)

    
    return train_t, encoder



def numerical_transform(X_train, lista_variaveis, forma, valor=None):
    """
        Tipos de transformações Numéricas:
            -> Log
            -> Reciprocal
            -> Power
            -> BoxCox
            -> YeoJohnson
    """
    if forma == 'Log':
        tf = vt.LogTransformer(variables = lista_variaveis)
    elif forma == 'Reciprocal':
        tf = vt.ReciprocalTransformer(variables = lista_variaveis)
    elif forma == 'Power':
        tf = vt.PowerTransformer(variables = lista_variaveis, exp = valor)
    elif forma == 'BoxCox':
        tf = vt.BoxCoxTransformer(variables = lista_variaveis)
    elif forma =='YeoJohnson' :
        tf = vt.YeoJohnsonTransformer(variables =lista_variaveis)
        
    tf.fit(X_train) 
    train_t = tf.transform(X_train)

    
    return train_t, tf


def resampler_regression(X_train,y_train, target):
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

def fs_variancia(X_train,
                 X_val,
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
    X_val = X_val[X_val.columns[sel.get_support()]]

    print('Número de colunas depois do filtro: ' + str(len(X_train.columns)))
    return X_train, X_val


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


def features_common_or_union(features_selecionadas_1, features_selecionadas_2):
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

#------------------------------------------------------------------------------

# Funções para transformação de Target

def normalizador_pg(x):
    #y = np.log1p(x)
    y = boxcox(x, 1)
    return y

def inversa_normalizador_pg(x):
    #y = np.expm1(x)
    y = inv_boxcox(x, 1)
    return y

def target_transform_reg(y_train, y_test):
    """
    
    
    """
    
    y_train2 = y_train.values.reshape(-1,1)
    #y_test2 = y_test.values.reshape(-1,1)
    target_transform = FunctionTransformer(normalizador_pg, inverse_func = inversa_normalizador_pg)
    #target_transform = TransformedTargetRegressor(normalizador_pg, inverse_func = inversa_normalizador_pg)

    target_transform.fit(y_train2)

    y_train_trans = target_transform.transform(y_train2)
    #y_test_trans = target_transform.transform(y_test2)
    
    plt.figure(figsize=(15,4))
    ax = plt.subplot(121)
    ax = sns.distplot(y_train,  bins = 10, hist=True,kde=False,rug=False,fit=None,hist_kws=None,kde_kws=None,
                      rug_kws=None,fit_kws=None,color=None,vertical=False, norm_hist=False, axlabel=None,label=None, ax=None)
    ax.set_title('Distribuição de Volumes');
    ax2 = plt.subplot(122)
    ax2 = sns.distplot(y_train_trans,  bins = 10, hist=True,kde=False,rug=False,fit=None,hist_kws=None,kde_kws=None,
                      rug_kws=None,fit_kws=None,color=None,vertical=False, norm_hist=False, axlabel=None,label=None, ax=None)
    ax2.set_title('Distribuição de Volumes');
    return target_transform

# AutoML Regression ----------------------------------------------------------

def plot_feature_importance(importance,names,model_type):
    """
    
    """
    
    # Crindo Arrays a partir do Feature Importance e do nom das variáveis
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Criação do DataFrame uasndo um dicionário
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    # Ordenação do DataFrame em ordem decrescente:
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
      
    print(len(fi_df))
    fi_df = fi_df.head(15)

    # Definição dos tamanhos do Bar plot
    plt.figure(figsize=(25,20))
    plt.rcParams['font.size'] = 50
    
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Adição das labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel(' FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    return list(fi_df.iloc[:,0])

def auto_ml_regressor(X_train, X_test, y_train, y_test,
                      algoritmos = ['randomforest', 'extratree','gbm',
                                    'adaboost','xgboost'],
                      search = False, type_search = None, target_transform = None):
    
    # Início do cronômetro:
    start = time.time()
    
    # Dicionário com a lista de modelos com configuração padrão:
    lista_modelos = {
                     'randomforest': RandomForestRegressor(random_state=0, n_jobs=-1),
                     'extratree': ExtraTreesRegressor(random_state=0, n_jobs=-1),
                     'adaboost':AdaBoostRegressor(random_state=0),
                     'gbm':GradientBoostingRegressor(),
                     'xgboost': XGBRegressor(random_state=0)
                     }
    
    # Dicionário com os parâmetros a serem testados no RandomSearch
    param_grid = {
                  'randomforest': {'n_estimators' : [200], 'max_depth': [5,20, None]},
                  'extratree': {'n_estimators' : [200], 'max_depth': [5,20, None]},
                  'adaboost':{'n_estimators':[200], 'learning_rate':[1,0.1,0.01,0.5,0.05]},
                  'gbm':{'n_estimators':[200], 'subsample' :[0.8,1], 'learning_rate':[1,0.1,0.01,0.001]},
                  'xgboost':{'n_estimators':[200,400],'eta': [0.001,0.01, 0.1, 1],'max_depth': [5,20],'subsample':[0.8,1]}
                 }
    
    # 'criterion':['mse', 'mae'],
    # 'min_samples_split':[2,5],'min_samples_leaf':[1,2,3],'max_features':['auto', 'sqrt', 'log2']

    lista_modelos_construidos=[]
    lista_predicoes=[]
    for i in lista_modelos.keys():
        if i in algoritmos:
            print(i)

            regressor = lista_modelos[i]
        
            if search == False:
                regressor.fit(X_train, y_train)
                importances = regressor.feature_importances_
                top_features = plot_feature_importance(importances,X_test.columns, i)

            else:
                if type_search == 'gridsearch':
                    regressor = GridSearchCV(regressor, param_grid = param_grid[i], cv = 4, verbose=1,
                                             n_jobs=-1, scoring='neg_mean_absolute_error')
                elif type_search =='randomsearch':
                    regressor = RandomizedSearchCV(regressor, param_distributions = param_grid[i],
                                                   n_iter = 10, cv=5, verbose=3,
                                                   n_jobs=-1, scoring='neg_mean_absolute_error',
                                                   random_state=0)
                else: pass

                regressor.fit(X_train, y_train)
                print(regressor.best_params_)
                params = regressor.best_params_
                
                regressor = lista_modelos[i]
                regressor.set_params(**params)
                regressor.fit(X_train, y_train)
                importances = regressor.feature_importances_
                top_features = plot_feature_importance(importances,X_test.columns, i)
                
            if target_transform != None:
                # invert transform on predictions
                yhat = regressor.predict(X_test)
                print(yhat[0])
                y_pred = target_transform.inverse_transform(yhat.reshape(-1,1))
                print(y_pred[0])
            else:
                y_pred = regressor.predict(X_test)
                
            y_pred = pd.DataFrame(y_pred, columns = [i])
            lista_predicoes.append(y_pred)
        
            erro_mse = round(mean_squared_error(y_test, y_pred, squared=False),2)
            erro_mae = round(mean_absolute_error(y_test, y_pred),2)
            r2 = r2_score(y_test, y_pred)
            n = len(y_pred)
            p = len(X_test.columns)
            r2adj = 1-(1-r2)*(n-1)/(n-p-1)
            
            lista_modelos_construidos.append([i, regressor, erro_mse, erro_mae, r2, r2adj])
            
        
    modelos = pd.DataFrame(lista_modelos_construidos, columns=['tipo','modelo','erro_mse', 'erro_mae', 'r2', 'r2adj'])
    melhor_modelo = modelos.sort_values(by=['r2adj'], ascending=False)
    end = time.time()
    print('Tempo de execução (minutos):')
    print((end - start) / 60)
    return melhor_modelo, modelos, lista_predicoes, top_features