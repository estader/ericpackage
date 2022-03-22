
from mlxtend.evaluate import bias_variance_decomp
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
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


def report_features_erros_modelos(modelos, X_val, y_val, df_predicoes, target, modo='regression'):

    if modo == 'regression':
        top_features = modelos.top_features.iloc[0]#.strip('][').replace("'",'').split(', ') 
        df_predicoes = pd.concat([df_predicoes, y_val.reset_index(drop=True)], axis=1, join='inner')
        df_predicoes = pd.concat([X_val[top_features].reset_index(drop=True), df_predicoes], axis=1)
        
        for i in list(modelos.tipo):
            df_predicoes[i+'_real_dif'] = df_predicoes.apply(lambda x: abs(x[i] - x[target]), axis=1)
            
        plt.figure(figsize=(10,10))
        plt.rcParams['font.size'] = 10
        ax1 = sns.scatterplot(data = df_predicoes,sizes=20, x=target, y=modelos.tipo.iloc[0], alpha=0.5, s=15)
        ax1.plot([0,int(df_predicoes[target].max())],[0,int(df_predicoes[target].max())], '--')
        ax1.set_title('Valores reais e previsões')
        
        plt.figure(figsize=(10,10))
        ax1 = sns.lineplot(data = df_predicoes, x = target, y=modelos.tipo.iloc[0]+'_real_dif')
        ax1.set_title('Erros x faixa do target')

        melhores_pred =df_predicoes.sort_values(modelos.iloc[0,0]+'_real_dif',ascending=True).head(10)
        piores_pred = df_predicoes.sort_values(modelos.iloc[0,0]+'_real_dif',ascending=False).head(10)

        return melhores_pred, piores_pred

    if modo == 'classifier':
       
        top_features = modelos.top_features.iloc[0]
        df_predicoes = pd.concat([df_predicoes, y_val.reset_index(drop=True)], axis=1, join='inner')
        df_predicoes = pd.concat([X_val[top_features].reset_index(drop=True), df_predicoes], axis=1)

        for i in list(modelos.tipo):
            df_predicoes[i+'_real_dif'] = df_predicoes.apply(lambda x: abs(x[i] - x[target]), axis=1)


        plt.figure(figsize=(10,10))
        cf_matrix = confusion_matrix(y_val, modelos.tipo.iloc[0])
        labels = ['True Neg','False Pos','False Neg','True Pos']
        labels = np.asarray(labels).reshape(2,2)
        ax1 = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
        ax1.set_title('Erros x faixa do target')

        # acertos
        melhores_pred = df_predicoes.sort_values(modelos.iloc[0,0]+'_real_dif', ascending=True).head(10)

        # erros
        piores_pred = df_predicoes.sort_values(modelos.iloc[0,0]+'_real_dif', ascending=False).head(10)

        return melhores_pred, piores_pred

