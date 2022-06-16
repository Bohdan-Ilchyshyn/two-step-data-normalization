import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_score, \
    recall_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from collections import defaultdict
from sklearn.model_selection import train_test_split

from proposed_scaler import SecondCustomScaler, FirstCustomScaler

clfs = {
    "Decision Tree Classifier": DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2,
                                                       random_state=0),
    "Extra Trees Classifier": ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=2,
                                                   random_state=0),
    "Ada Boost Classifier": AdaBoostClassifier(base_estimator=None, n_estimators=100, learning_rate=1.0,
                                               algorithm='SAMME.R', random_state=None),
    "Gradient Boosting Classifier": GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100,
                                                               subsample=1.0, criterion='friedman_mse',
                                                               min_samples_split=2, min_samples_leaf=1,
                                                               min_weight_fraction_leaf=0.0, max_depth=3,
                                                               min_impurity_decrease=0.0, init=None, random_state=None,
                                                               max_features=None, verbose=0, max_leaf_nodes=None,
                                                               warm_start=False, validation_fraction=0.1,
                                                               n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0),
    "Bagging Classifier": BaggingClassifier(base_estimator=None, n_estimators=100, max_samples=1.0, max_features=1.0,
                                            bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False,
                                            n_jobs=None, random_state=None, verbose=0)
}

scalers = {
    "First Custom Scaler": FirstCustomScaler,
    "Second Custom Scaler": SecondCustomScaler,
    "Max Abs Scaler": MaxAbsScaler,
    "Min Max Scaler": MinMaxScaler,
    "Standard Scaler": StandardScaler,
    "Robust Scaler": RobustScaler
}


def train_and_test(clf, train_X, train_y, test_X, test_y):
    clf.fit(train_X, train_y)

    train_predictions = clf.predict(train_X)
    test_predictions = clf.predict(test_X)

    train_accuracy_score = accuracy_score(train_y, train_predictions)
    test_accuracy_score = accuracy_score(test_y, test_predictions)

    train_precision_score = precision_score(train_y, train_predictions,
                                            average='macro'
                                            )
    test_precision_score = precision_score(test_y, test_predictions,
                                           average='macro'
                                           )

    train_recall_score = recall_score(train_y, train_predictions,
                                      average='macro'
                                      )
    test_recall_score = recall_score(test_y, test_predictions,
                                     average='macro'
                                     )

    train_f1_score = f1_score(train_y, train_predictions,
                              average='macro'
                              )
    test_f1_score = f1_score(test_y, test_predictions,
                             average='macro'
                             )

    print("Train Accuracy :: " + str(train_accuracy_score))
    print("Train Precision :: " + str(train_precision_score))
    print("Train Recall :: " + str(train_recall_score))
    print("Train f1_score :: " + str(train_f1_score))

    print("Test Accuracy  :: " + str(test_accuracy_score))
    print("Test Precision :: " + str(test_precision_score))
    print("Test Recall :: " + str(test_recall_score))
    print("Test f1_score :: " + str(test_f1_score))

    return {
        "train_accuracy_score": train_accuracy_score,
        "test_accuracy_score": test_accuracy_score,
        "train_precision_score": train_precision_score,
        "test_precision_score": test_precision_score,
        "train_recall_score": train_recall_score,
        "test_recall_score": test_recall_score,
        "train_f1_score": train_f1_score,
        "test_f1_score": test_f1_score,
    }


def run_for_all_scallers_and_classifier(train_X, train_y, test_X, test_y, dataset_name, save_to_excel: bool = True):
    result = defaultdict(dict)

    for scaler_name, scaler_cls in scalers.items():

        train_X_copy, train_y_copy, test_X_copy, test_y_copy = train_X.copy(), train_y.copy(), test_X.copy(), test_y.copy()

        scaler = scaler_cls()
        scaler.fit(train_X_copy)

        train_X_copy = scaler.transform(train_X_copy)
        test_X_copy = scaler.transform(test_X_copy)

        for classifier_name, classifier_instance in clfs.items():
            print(classifier_name)

            result[classifier_name][scaler_name] = train_and_test(classifier_instance, train_X_copy, train_y_copy,
                                                                  test_X_copy, test_y_copy)

    if save_to_excel:
        for clf_name, clf_content in result.items():
            res_df = pd.DataFrame(clf_content)
            res_df.to_excel(f'{dataset_name}_{clf_name}.xlsx')


run_for_all_scallers_and_classifier(train_X, train_y, test_X, test_y, "dermatology")

