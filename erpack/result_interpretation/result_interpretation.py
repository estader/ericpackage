
import lime
import lime.lime_tabular


def ri_lime_tabular(X_train, X_test, modelo, index_sample, class_names, modo = 'regression'):
    """
        Interpretador para dados tabulares
        Modo regressão ou classificação: regression | classification
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names = X_train.columns.values,
                                                       class_names= class_names,
                                                       verbose=True,
                                                       mode=modo,
                                                       random_state=0)

    if modo =='regression':
        exp = explainer.explain_instance(X_test.iloc[index_sample], modelo.predict,
                                         num_features = len(X_train.columns))
    else:
        exp = explainer.explain_instance(X_test.iloc[index_sample], modelo.predict_proba,
                                         num_features = len(X_train.columns))
    exp.show_in_notebook(show_table=True)
    exp.as_list()
    return exp