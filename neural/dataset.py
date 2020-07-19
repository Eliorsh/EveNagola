import torch
from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.labels = None
        self.data = None
        self.clean_data()

    def clean_data(self, flip=False):
        # Data cleaning
        df = self.df[self.df.corona_result != 'אחר']
        if flip:
            df = df[::-1]
        df.corona_result = df.corona_result.apply(
            lambda x: 1 if x == 'חיובי' else 0)
        df.age_60_and_above = df.age_60_and_above.apply(
            lambda x: 1 if x == 'Yes' else 0 if x == 'No' else None)
        df.gender = df.gender.apply(
            lambda x: 1 if x == 'נקבה' else 0 if x == 'זכר' else None)
        df['was_abroad'] = df.test_indication.apply(
            lambda x: 1 if x == 'Abroad' else 0)
        df['had_contact'] = df.test_indication.apply(
            lambda x: 1 if x == 'Contact with confirmed' else 0)
        df_no_na = df.dropna()
        self.labels = torch.LongTensor(np.array(df_no_na.corona_result))
        df_no_na = df_no_na.drop(
            columns=['corona_result', 'test_date', 'test_indication'])

        # Convert to numpy array
        self.data = torch.FloatTensor(np.array(df_no_na))

    def __getitem__(self, idx):
        # x = [df.loc[idx, col] for col in df.columns if col != 'corona_result']
        # y = self.df.loc[idx, 'corona_result']
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)