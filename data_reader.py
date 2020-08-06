import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

# source: https://data.gov.il/dataset/covid-19

class DataProcessor:
    def __init__(self, path):
        # Load data
        _, file_type = path.split('.')
        if 'xls' in file_type:
            self.df = pd.read_excel(path)
        elif file_type == 'csv':
            self.df = pd.read_csv(path)
        else:
            raise TypeError("MA ZE?")
        self.processed_df = None
        self.df_dates = None
        self.np_data = None
        self.test_data = None

    def clean_data(self, flip=False):
        # Data cleaning
        df = self.df[self.df.corona_result != 'אחר']
        if flip:
            df = df[::-1]
        df.corona_result = df.corona_result.apply(lambda x: 1 if x == 'חיובי' else 0)
        df.age_60_and_above = df.age_60_and_above.apply(lambda x: 1 if x == 'Yes' else 0 if x == 'No' else None)
        df.gender = df.gender.apply(lambda x: 1 if x == 'נקבה' else 0 if x == 'זכר' else None)
        df['was_abroad'] = df.test_indication.apply(lambda x: 1 if x == 'Abroad' else 0)
        df['had_contact'] = df.test_indication.apply(lambda x: 1 if x == 'Contact with confirmed' else 0)
        # df['had_other'] = df.test_indication.apply(lambda x: 1 if x == 'Other' else 0)
        self.df_dates = df.drop(columns=['test_indication'])
        self.all_dates = self.df_dates.test_date.unique()
        self.processed_df = df.drop(columns=['test_date', 'test_indication'])
        # df_no_na = self.processed_df.dropna()
        # Convert to numpy array
        # self.np_data = np.array(df_no_na)
        for col_name in self.processed_df.columns:
            col = self.processed_df[col_name]
            nulls = col.isnull()
            col.loc[nulls] = col.dropna().sample(nulls.sum()).values
        # means = self.processed_df.mean()
        # self.processed_df = self.processed_df.fillna(means)
        self.np_data = np.array(self.processed_df)
        # self.np_data = self.fill_na(self.processed_df)

    def fill_na(self, df, df_no_na):
        c_gen = Counter(df_no_na.gender)
        gen0 = c_gen[0] / (c_gen[0] + c_gen[1])
        gen1 = 1 - gen0

        c_age = Counter(df_no_na.age_60_and_above)
        age0 = c_age[0] / (c_age[0] + c_age[1])
        age1 = 1 - age0

        df_filled_na = df.copy()

        # df_filled_na.head()
        def gender_fill_na(x):
            if np.isnan(x):
                x = np.random.choice([0, 1], p=[gen0, gen1])
            return x

        def age_fill_na(x):
            if np.isnan(x):
                x = np.random.choice([0, 1], p=[age0, age1])
            return x

        df_filled_na.gender = df_filled_na.gender.transform(gender_fill_na)
        df_filled_na.age_60_and_above = df_filled_na.age_60_and_above.transform(
            age_fill_na)
        df_filled_na = df_filled_na.fillna(0)
        np_df_filled = np.array(df_filled_na)
        return np_df_filled

    def get_daily_data(self, date_input, end_date=np.datetime64('today')):
        if type(date_input) == int:
            if date_input < 0:
                plot_days = self.all_dates[date_input:]
            else:
                plot_days = self.all_dates[:date_input]
        elif type(date_input) in [np.datetime64, str]:
            plot_days = np.arange(date_input, np.datetime64(end_date) + 1,
                                  dtype='datetime64[D]')
        elif type(date_input) == list:
            plot_days = np.array(date_input) # of form ['yyyy-mm-dd', ...]
            plot_days = [np.datetime64(date) for date in plot_days]
        data_by_day = {}
        for day in plot_days:
            df_day = self.df_dates[self.df_dates.test_date == str(day)]
            df_day = df_day.drop(columns=['test_date'])
            # df_day_no_na = df_day.dropna()
            # np_df_day = np.array(df_day_no_na)
            np_df_day = np.array(df_day)
            data_by_day[day] = np_df_day
        return data_by_day

    def split_data(self, test_size=0.3, wave=0, random_state=42):
        # Split data
        if wave == 1:
            data = self.processed_df[self.df_dates['test_date'] < '2020-04-25']
            np_data = np.array(data)
        elif wave == 2:
            data = self.processed_df[
                self.df_dates['test_date'] > '2020-05-31']
            np_data = np.array(data)
        else:
            np_data = self.np_data
        X = np_data[:, 1:]
        Y = np_data[:, 0]
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state)
        self.test_data = np.concatenate((y_test.reshape((len(y_test), 1)), X_test), axis=1)
        return X_train, X_test, y_train, y_test

    def get_test_data(self):
        return self.test_data

    def get_all_data(self):
        return self.np_data
