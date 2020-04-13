import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataProcessor:
    def __init__(self, path):
        # Load data
        self.df = pd.read_excel(path)
        self.processed_df = None
        self.df_dates = None
        self.np_data = None
        self.test_data = None

    def clean_data(self):
        # Data cleaning
        df = self.df[self.df.corona_result != 'אחר']
        df.corona_result = df.corona_result.apply(lambda x: 1 if x == 'חיובי' else 0)
        df.age_60_and_above = df.age_60_and_above.apply(lambda x: 1 if x == 'Yes' else 0 if x == 'No' else None)
        df.gender = df.gender.apply(lambda x: 1 if x == 'נקבה' else 0 if x == 'זכר' else None)
        df['was_abroad'] = df.test_indication.apply(lambda x: 1 if x == 'Abroad' else 0)
        df['had_contact'] = df.test_indication.apply(lambda x: 1 if x == 'Contact with confirmed' else 0)
        self.df_dates = df.drop(columns=['test_indication'])
        self.all_dates = self.df_dates.test_date.unique()
        self.processed_df = df.drop(columns=['test_date', 'test_indication'])
        # Drop NA TODO: אולי לא לזרוק
        df_no_na = self.processed_df.dropna()
        # Convert to numpy array
        self.np_data = np.array(df_no_na)

    def get_daily_data(self, days_back):
        data_by_day = {}
        plot_days = self.all_dates[-days_back:]
        for day in plot_days:
            df_day = self.df_dates[self.df_dates.test_date == day]
            df_day = df_day.drop(columns=['test_date'])
            df_day_no_na = df_day.dropna()
            np_df_day = np.array(df_day_no_na)
            data_by_day[day] = np_df_day
        return data_by_day

    def split_data(self, test_size=0.33, random_state=42):
        # Split data
        X = self.np_data[:, 1:]
        Y = self.np_data[:, 0]
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state)
        self.test_data = np.concatenate((y_test.reshape((len(y_test), 1)), X_test), axis=1)
        return X_train, X_test, y_train, y_test

    def get_test_data(self):
        return self.test_data

    def get_all_data(self):
        return self.np_data
