import glob

import torch
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from data_reader import DataProcessor
from neural.nn_models import Net

DEFAULT_MODEL = 'xgb'
NEURAL_MODEL_PATH = './neural/corn_model2.pt'


class Models:
    def __init__(self, x, y):
        self.x_train = x
        self.y_train = y
        self.models_dict = {
            'xgb': self.get_xgb_model,
            'logreg': self.get_logreg_model,
            'bayes': self.get_bayesian_model,
            'forest': self.get_forest_model,
            'neural': self.get_neural_model
        }

    def get_model(self, model):
        return self.models_dict.get(model, DEFAULT_MODEL)()

    def get_logreg_model(self):
        # logistic regression
        logreg_model = LogisticRegression(C=1e5)
        logreg_model.fit(self.x_train, self.y_train)
        print("using logreg model")
        return logreg_model

    def get_xgb_model(self):
        # XGB
        parameters = {
            "max_depth": 4,
            "learning_rate": 0.001,
            "n_estimators": 140,
            "scale_pos_weight": 4,
            # "eta": 0.5,
            # "min_child_weight": 5,
            # "gamma": 0.2,
            # "colsample_bytree": 0.5
        }
        xgb_model = XGBClassifier(**parameters)
        xgb_model.fit(self.x_train, self.y_train)
        print("using xgb model")
        return xgb_model

    def get_bayesian_model(self):
        bayes_model = GaussianNB()
        bayes_model.fit(self.x_train, self.y_train)
        print("using bayes model")
        return bayes_model

    def get_forest_model(self):
        forest_model = RandomForestClassifier(n_estimators=100,
                               random_state=42,
                               max_features = 'sqrt',
                               n_jobs=-1, verbose = 1)
        forest_model.fit(self.x_train, self.y_train)
        print("using forest model")
        return forest_model

    def get_neural_model(self):
        net = Net()
        print("using neural model")
        return NeuralModel(net, NEURAL_MODEL_PATH)


class MockModel:
    @staticmethod
    def predict_proba(data):
        return np.zeros((1,2))


class NeuralModel:
    def __init__(self, net, model_path):
        self.net = net
        self.net.load_state_dict(torch.load(model_path))

    def predict(self, x):
        pred = self.net(x)
        prob, label = torch.topk(torch.nn.functional.softmax(pred), 2)
        risk_profile = prob[1]
        return risk_profile

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

def evaluate_model(model, data_path):
    dp = DataProcessor(data_path)
    dp.clean_data()
    X_train, X_test, y_train, y_test = dp.split_data()
    models = Models(X_train, y_train)
    model = models.get_model(model)
    y_pred = model.predict(X_test)
    results = evaluate_results(y_pred, y_test)
    return results

def compare_model_on_datasets(model, ds_dir):
    datasets = glob.glob(f'{ds_dir}/*')
    all_results = {}
    for ds in datasets:
        ds = ds[2:]
        ds_num = ds.split('.')[0].split('_')[-1]
        all_results[ds_num] = evaluate_model(model, ds)
    sorted_nums = sorted(all_results.keys(), key=int)
    recalls = [all_results[num]['recall'] for num in sorted_nums]
    precisions = [all_results[num]['precision'] for num in sorted_nums]
    return all_results, recalls, precisions

if __name__ == '__main__':
    data_path = 'data/corona_tested_individuals_ver_0049.csv'
    data_dir = './data'
    model = 'xgb'
    print(evaluate_model(model, data_path))
    # all_results, recalls, precisions = compare_model_on_datasets(model, data_dir)
    # print(recalls)
    # print('-' * 50)
    # print(precisions)
    # print('='* 50)
    # print(all_results)
