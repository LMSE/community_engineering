import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from sklearn.base import clone 

def dropcol_importances(rf, X_train, y_train):
    """
    Calculates the drop-column importance for features in the dataset. It assesses the importance of a feature
    by observing the decrease in the out-of-bag score when each feature is dropped.

    Args:
        rf (RandomForestClassifier): The random forest classifier to use for assessing feature importance.
        X_train (DataFrame): Training data features.
        y_train (Series): Training data labels.

    Returns:
        DataFrame: A DataFrame containing the feature importances sorted in ascending order.
    """
    rf_ = clone(rf)
    rf_.set_params(warm_start=True, oob_score=True)
    rf_.random_state = 999
    rf_.fit(X_train, y_train)
    baseline = rf_.oob_score_
    imp = []
    for col in X_train.columns:
        X = X_train.drop(col, axis=1)
        rf_ = clone(rf)
        rf_.random_state = 999
        rf_.fit(X, y_train)
        o = rf_.oob_score_
        imp.append(baseline - o)
    imp = np.array(imp)
    I = pd.DataFrame(
            data={'Feature':X_train.columns,
                  'Importance':imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=True)
    return I


def randomForrest(fileName):
    """
    Trains a random forest classifier to predict a binary outcome based on the 'CSI' value and assesses feature
    importance using the trained model. It uses basic feature importance, permutation importance, and drop-column
    importance to evaluate feature relevance.

    Args:
        fileName (str): The path to the CSV file containing the data for training the classifier.

    Displays:
        - Feature ranking based on their importance.
        - Bar charts showing the feature importances and their standard deviations.
        - Bar chart showing the feature importances using permutation importance.
        - Bar chart showing the feature importances based on drop-column importance.

    Returns:
        None
    """
    df=pd.read_csv(fileName)
    #df = df[df.columns[1:]]
    X = df.drop(['CSI','biomass1_SS', 'biomass2_SS', 'biomass1', 'biomass2'], axis=1)
    feature_names=X.columns.to_list()
    y = np.where(df['CSI']>=0.8, 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42)

    # Build a forest and compute the feature importances
    forest = RandomForestClassifier(n_estimators=120,warm_start=True, max_features=None,
                               oob_score=True,
                                random_state=0)

    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    colNames=[]

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        colNames.append(X.columns[indices[f]])

    print(colNames)

    df = pd.DataFrame(data=colNames, columns=['param'])
    df['importances']=importances[indices]
    df['std']=std[indices]
    print(df)
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), colNames)
    plt.xlim([-1, X.shape[1]])
    plt.show()    

    result = permutation_importance(
        forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)

    forest_importances = pd.Series(result.importances_mean, index=feature_names)
    print(forest_importances)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()

    df=dropcol_importances(forest, X_train, y_train)
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), colNames)
    plt.xlim([-1, X.shape[1]])
    plt.show()   


if __name__ == '__main__':
    randomForrest("/Users/ruhichoudhary/code/gitrepos/DyMMM/communities/communitycoop_cstr/V8s/params_00000_RESULT.csv")
