from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_excel('corona_tested_individuals_ver_001.xlsx')

# Data cleaning
df = df[df.corona_result != 'אחר']
df.corona_result = df.corona_result.apply(lambda x: 1 if x == 'חיובי' else 0)
df.age_60_and_above = df.age_60_and_above.apply(lambda x: 1 if x == 'Yes' else 0 if x == 'No' else None)
df.gender = df.gender.apply(lambda x: 1 if x == 'נקבה' else 0 if x == 'זכר' else None)
df['was_abroad'] = df.test_indication.apply(lambda x: 1 if x == 'Abroad' else 0)
df['had_contact'] = df.test_indication.apply(lambda x: 1 if x == 'Contact with confirmed' else 0)
df = df.drop(columns=['test_date', 'test_indication'])
# Drop NA TODO: אולי לא לזרוק
df_no_na = df.dropna()
# Convert to numpy array
all_subjects = np.array(df_no_na)

# Split data
X = all_subjects[:, 1:]
Y = all_subjects[:, 0]
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42)

# logistic regression
logreg_model = LogisticRegression(C=1e5)
logreg_model.fit(X_train, y_train)

# XGB
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)



if __name__ == '__main__':
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
        return {'True Negative':TN, 'True Positive':TP, 'False Negative':FN, 'False Positive':FP, 'recall':TP/(TP + FN), 'precision': TP/(TP + FP)}

    y_pred = xgb_model.predict(X_test)
    print(evaluate_results(y_pred, y_test))
