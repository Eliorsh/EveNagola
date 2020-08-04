from flask import Flask, request
import numpy as np

from data_reader import DataProcessor
from models import Models
app = Flask(__name__)

N_FEATURES = 9
HIGH_TRESHOLD = 80
LOW_TRESHOLD = 50
DATA_PATH = 'data/corona_tested_individuals_ver_005.csv'


def get_model(data_path, model_name='xgb'):
    flip = False if int(data_path.split('.')[0][-3:]) > 5 else True
    dp = DataProcessor(data_path)
    dp.clean_data(flip)
    X_train, X_test, y_train, y_test = dp.split_data()
    models = Models(X_train, y_train)
    return models.get_model(model_name)


model = get_model(DATA_PATH, 'xgb')


@app.route('/predict', methods=['post'])
def predict():
    req_data = request.get_json()
    # print(req_data)
    # {"gender": 1, "isCoughing": 0, "isFever": 1, "isThroatAche": 0,
    #  "isHeadAche": 0, "isTroubleBreathing": 0, "isOver60": 0,
    #  "isReturningFromAbroad": 0, "isContactedInfected": 0}

    if req_data is None:
        return {'success': False, 'error': 'empty body request'}

    features = process_data(req_data)

    if None in features:
        return {'success': False, 'error': 'request missing data'}

    if len(features) < N_FEATURES:
        return {'success': False, 'error': 'request missing feature'}

    score = set_danger_level(model, features)
    label = get_danger_label(score, HIGH_TRESHOLD, LOW_TRESHOLD)
    return {'success': True, 'error': None, 'label': label, 'score':score}


def process_data(req_data):
    features = np.array([req_data.get('isCoughing'),
                         req_data.get('isFever'),
                         req_data.get('isThroatache'),
                         req_data.get('isTroubleBreathing'),
                         req_data.get('isHeadache'),
                         req_data.get('isOver60'),
                         req_data.get('gender'),
                         req_data.get('isReturningFromAbroad'),
                         req_data.get('isContactedInfected')])
    return features


def set_danger_level(model, features):
    person_data = features.reshape((1, len(features)))
    return model.predict_proba(person_data)[0, 1] * 100


def get_danger_label(danger_level, t1=HIGH_TRESHOLD, t2=None):
    if danger_level > t1:
        return 2
    if t2 and danger_level > t2:
        return 1
    return 0
