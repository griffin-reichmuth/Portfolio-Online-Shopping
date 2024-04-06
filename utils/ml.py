from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def PredictionAccuracy(predicted, actual, set_name):
    """Takes predicted data, the actual data, and a string for the type
       of data set input (e.g. TEST, TRAIN).
    Prints and returns the accuracy score.
    """
    # using metrics module for accuracy calculation
    accuracy = accuracy_score(actual, predicted)
    print("ACCURACY OF THE MODEL ON THE", set_name, "SET:  ", accuracy)

    return accuracy


def underSampleTraining(x, y):
    """ Takes an x data set and a y data set and returns undersampled data
    sets for both.
    """
    # sample data using undersampling
    sampler = RandomUnderSampler(random_state=0, replacement=True)
    x_undersample, y_undersample = sampler.fit_resample(x, y)

    return x_undersample, y_undersample


def dataSplitter(x, y, split=0.2, resample=False):
    """Takes x and y data and returns a data set split into a training and
    testing set. By default, takes split=0.2 for a testing set size 20%
    (therefore 80% training set). By default, takes resample=False. If True,
    data is resampled using RandomUnderSampler().

    Returns data sets for X_train, y_train, X_test, y_test, x_undersample,
    and y_undersample.
    """
    # intialize resampled sets
    x_undersample, y_undersample = None, None

    # Split data into an training and testing set with the remaining data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=split,
                                                        random_state=0)

    if resample:
        x_undersample, y_undersample = underSampleTraining(X_train, y_train)

    return X_train, y_train, X_test, y_test, x_undersample, y_undersample


def roc(test, pred):
    """
    Takes actual labels and predicted labels.
    Returns the area under the curve (AUC) for the ROC curve.
    """
    fpr, tpr, thresholds = roc_curve(test, pred)
    plt.plot(fpr, tpr, color='red')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    print("AUC Score is:")
    auc = roc_auc_score(test, pred)
    return auc


def pre_rec(test, pred):
    """
    Takes actual labels and predicted labels.
    Returns the area under the curve (AUC) for the precison-recall curve.
    """
    precision, recall, thresholds = precision_recall_curve(test, pred)
    plt.fill_between(recall, precision, color='blue')
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.title("Test Set Precision vs. Recall")
    print('Precision-Recall curve area is:')
    pr_score = auc(recall, precision)
    return pr_score


# paramters to use for the tuning with cross validation
param_grid_rfc = {
    'max_depth': [int(x) for x in np.linspace(10, 30, num=5)],
    'max_features': [5, 10, 15, 20, 25, 29],    # total features up to 29
    # number of trees
    'n_estimators': [int(x) for x in np.linspace(start=10, stop=100, num=10)],
    'bootstrap': [True, False]}

param_grid_rfc['max_depth'].append(None)


def RandomForestBuilder(X_train, y_train, X_test, y_test, CV=False):
    """""
    Takes feature data (x) and label data (y). Resample=True will resample
    the data using undersampling, tuning=True will use a 5 fold cross
    validation procedure to determine the best parameters.

    Fits a random forest classifier, printing the accuracy of the split
    training and testing of x and y. Also prints the classification report.

    Returns the model, the predicted test set labels, and the precision-recall
    curve AUC value.
    """

    # make predictions on test set

    if CV:
        model = RandomizedSearchCV(estimator=RandomForestClassifier(),
                                   param_distributions=param_grid_rfc,
                                   n_iter=50, cv=5, verbose=2, random_state=0,
                                   n_jobs=-1)

        # fit the model with training data
        model.fit(X_train, y_train)

    else:
        # create the classifier
        model = RandomForestClassifier(random_state=0)

        # fit the model with training data
        model.fit(X_train, y_train)

    # training set
    train_predicted = model.predict(X_train)
    PredictionAccuracy(train_predicted, y_train, 'TRAIN')

    # test set
    test_predicted = model.predict(X_test)
    PredictionAccuracy(test_predicted, y_test, 'TEST')
    print()

    print('\n', 'Classification Report of Model on Testing Set')
    print(classification_report(test_predicted, y_test))
    print()
    precision_recall_AUC = pre_rec(y_test, test_predicted)
    print(precision_recall_AUC)

    return model, test_predicted, precision_recall_AUC
