
from mlxtend.evaluate import bias_variance_decomp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def decomposicao_bias_variancia(modelo, X_train, y_train, X_test, y_test):
    mse, bias, var = bias_variance_decomp(modelo, X_train.values,
                                          y_train.ravel(), X_test.values,
                                          y_test.ravel(), loss='mse',
                                          num_rounds=3, random_seed=0)
    print('MSE: %.3f' % mse)
    print('Bias: %.3f' % bias)
    print('Variance: %.3f' % var)
    return


def report_features_erros_modelos(modelos, X_test, y_test, lista_predicoes, top_features, target):
    df_predicoes = pd.concat(lista_predicoes, axis=1)
    df_predicoes = pd.concat([df_predicoes, y_test.reset_index(drop=True)], axis=1)
    df_predicoes = pd.concat([X_test[top_features], df_predicoes], axis=1)
    
    for i in list(modelos.tipo):
        df_predicoes[i+'_real_dif'] = df_predicoes.apply(lambda x: abs(x[i] - x[target]), axis=1)
        
    plt.figure(figsize=(10,10))
    plt.rcParams['font.size'] = 10
    ax1 = sns.scatterplot(data = df_predicoes,sizes=20, x=target, y=modelos.tipo.iloc[0], alpha=0.5, s=15)
    print(int(df_predicoes[target].max()))
    ax1.plot([0,int(df_predicoes[target].max())],[0,int(df_predicoes[target].max())], '--')
    ax1.set_title('Valores reais e previsões');
    
    plt.figure(figsize=(10,10))
    #ax1 = sns.distplot(y_train/1000,  bins = 100, hist=True,kde=False,rug=False,fit=None,hist_kws=None,kde_kws=None,
    #                  rug_kws=None,fit_kws=None,color=None,vertical=False, norm_hist=False, axlabel=None,label=None, ax=None)
    ax1 = sns.lineplot(data = df_predicoes, x = target, y=modelos.tipo.iloc[0]+'_real_dif')

    ax1.set_title('');
    
    return df_predicoes.sort_values(modelos.iloc[0,0]+'_real_dif',ascending=False).head(10)
