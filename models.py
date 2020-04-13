from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from data_reader import DataProcessor


class Models:
    def __init__(self, x, y):
        self.x_train = x
        self.y_train = y

    def get_logreg_model(self):
        # logistic regression
        logreg_model = LogisticRegression(C=1e5)
        logreg_model.fit(self.x_train, self.y_train)
        return logreg_model

    def get_xgb_model(self):
        # XGB
        xgb_model = XGBClassifier()
        xgb_model.fit(self.x_train, self.y_train)
        return xgb_model


def evaluate_results(pred, y_test):
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    for i in range(len(pred)):
        if pred[i] == y_test[i]:
            if pred[i] == 1:
                TP += 1
            else:
                TN += 1
        else:
            if pred[i] == 1:
                FP += 1
            else:
                FN += 1
    return {'True Negative': TN, 'True Positive': TP, 'False Negative': FN,
            'False Positive': FP, 'recall': TP / (TP + FN),
            'precision': TP / (TP + FP)}


if __name__ == '__main__':
    data_path = 'corona_tested_individuals_ver_001.xlsx'
    dp = DataProcessor(data_path)
    dp.clean_data()
    X_train, X_test, y_train, y_test = dp.split_data()

    models = Models(X_train, y_train)
    xgb_model = models.get_xgb_model()
    y_pred = xgb_model.predict(X_test)
    print(evaluate_results(y_pred, y_test))
