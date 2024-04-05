import pandas as pd


def getFeatureImportance(model, x_data):
    '''
    Takes a model and features data. Returns a pandas dataframe with the
     features ranked by Gini from most important to least.
    '''
    feature_importances = pd.DataFrame(model.feature_importances_,
                                index=x_data.columns,
                                columns= ['importance']).sort_values('importance',
                                ascending=False)
    return feature_importances


def n_Features(feature_rankings, n, head=True):
    '''
    Takes the features, number of features n, and an optional parameter head
    (default True).
    Returns the top n features if head=True and bottom n features of
    head=False.
    '''
    if head is False:
        return feature_rankings.tail(n)
    return feature_rankings.head(n)
