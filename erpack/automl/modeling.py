# Funções de preparação e modelagem de dados

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_recall_curve, recall_score
from xgboost import XGBClassifier, XGBRegressor

from reg_resampler import resampler
from scipy.special import boxcox, inv_boxcox, boxcox1p, inv_boxcox1p
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import  AdaBoostRegressor, GradientBoostingRegressor
import time
from sklearn.metrics import r2_score, auc, precision_score, recall_score, f1_score, roc_curve



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
    plt.figure(figsize=(10,10))
    plt.rcParams['font.size'] = 10
    
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Adição das labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel(' FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    return list(fi_df.iloc[:,0])


def auto_ml_classifier(X_train, X_test, y_train, y_test,
                      algoritmos = ['randomforest', 'extratree','gbm',
                                    'adaboost','xgboost'],
                      search = False, type_search = None):
    
    # Início do cronômetro:
    start = time.time()
    
    # Dicionário com a lista de modelos com configuração padrão:
    lista_modelos = {
                     'randomforest': RandomForestClassifier(random_state=0, n_jobs=-1),
                     'extratree': ExtraTreesClassifier(random_state=0, n_jobs=-1),
                     'adaboost':AdaBoostClassifier(random_state=0),
                     'gbm':GradientBoostingClassifier(),
                     'xgboost': XGBClassifier(random_state=0)
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

            classifier = lista_modelos[i]
        
            if search == False:
                classifier.fit(X_train, y_train)
                importances = classifier.feature_importances_
                top_features = plot_feature_importance(importances,X_test.columns, i)

            else:
                if type_search == 'gridsearch':
                    classifier = GridSearchCV(classifier, param_grid = param_grid[i], cv = 3, verbose=1,
                                             n_jobs=-1, scoring='f1')
                elif type_search =='randomsearch':
                    classifier = RandomizedSearchCV(classifier, param_distributions = param_grid[i],
                                                   n_iter = 10, cv=3, verbose=3,
                                                   n_jobs=-1, scoring='f1',
                                                   random_state=0)
                else: pass

                classifier.fit(X_train, y_train)
                print(classifier.best_params_)
                params = classifier.best_params_
                
                classifier = lista_modelos[i]
                classifier.set_params(**params)
                classifier.fit(X_train, y_train)
                importances = classifier.feature_importances_
                top_features = plot_feature_importance(importances,X_test.columns, i)
                

            y_pred = classifier.predict(X_test)
                
            y_pred = pd.DataFrame(y_pred, columns = [i])
            lista_predicoes.append(y_pred)
        
            precision = precision_score(y_test, y_pred, average=None)
            recall = recall_score(y_test, y_pred, average=None)
            f1 = f1_score(y_test, y_pred, average=None)
            fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=2)
            auc1 = auc(fpr, tpr) 
            
            lista_modelos_construidos.append([i, classifier, precision , recall , f1 , auc1 , top_features])
            
    df_predicoes = pd.concat(lista_predicoes, axis=1)
    modelos = pd.DataFrame(lista_modelos_construidos, columns=['tipo','modelo','precision', 'recall', 'f1', 'auc1','top_features'])
    #modelos = modelos.sort_values(by=['f1'], ascending=False)
    end = time.time()
    print('Tempo de execução (minutos):')
    print((end - start) / 60)
    return modelos, df_predicoes


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
                    regressor = GridSearchCV(regressor, param_grid = param_grid[i], cv = 3, verbose=1,
                                            n_jobs=-1, scoring='neg_mean_absolute_error')
                elif type_search =='randomsearch':
                    regressor = RandomizedSearchCV(regressor, param_distributions = param_grid[i],
                                                n_iter = 10, cv=3, verbose=3,
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
            
            lista_modelos_construidos.append([i, regressor, erro_mse, erro_mae, r2, r2adj, top_features])
            
    df_predicoes = pd.concat(lista_predicoes, axis=1)
    modelos = pd.DataFrame(lista_modelos_construidos, columns=['tipo','modelo','erro_mse', 'erro_mae', 'r2', 'r2adj','top_features'])
    modelos = modelos.sort_values(by=['r2adj'], ascending=False)
    end = time.time()
    print('Tempo de execução (minutos):')
    print((end - start) / 60)
    return modelos, df_predicoes