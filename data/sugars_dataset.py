from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SugarsDataset(Dataset):
    def __init__(self, dataset_path: str = "datasets/synthetic_data.csv"):
        super().__init__()
        dataframe = pd.read_csv(dataset_path)
        sugar_1_norm = dataframe['sugar_1'] / dataframe['sugar_1'].max()
        sugar_2_norm = dataframe['sugar_2'] / dataframe['sugar_2'].max()
        sugar_3_norm = dataframe['sugar_3'] / dataframe['sugar_3'].max()
        sugar_4_norm = dataframe['sugar_4'] / dataframe['sugar_4'].max()
        hours_norm = dataframe['hours'] / dataframe['hours'].max()
        dataframe['sugar_1'] = sugar_1_norm
        dataframe['sugar_2'] = sugar_2_norm
        dataframe['sugar_3'] = sugar_3_norm
        dataframe['sugar_4'] = sugar_4_norm
        dataframe['hours'] = hours_norm
        self.dataframe_norm = dataframe
        print(dataframe.head())

    def __getitem__(self, index: int):
        inputs = self.dataframe_norm.iloc[index][['sugar_1', 'sugar_2', 'sugar_3', 'sugar_4', 'hours']].values
        label = self.dataframe_norm.iloc[index]['label']
        inputs = inputs.astype(np.float32)
        label = label.astype(np.int8)
        return inputs, label
    
    def __len__(self):
        return self.dataframe_norm.shape[0]