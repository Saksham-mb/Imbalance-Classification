from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT=Path(__file__).parent.parent
DATASET_DIR=PROJECT_ROOT / 'datasets' / 'Credit-Card-Data' / 'creditcard.csv'


class DataExtractor:
    def __init__(self, DATASET_DIR: Path = DATASET_DIR):
        self.df=pd.read_csv(DATASET_DIR)
        self.X=self.df.drop(columns=['Class'])
        self.y=self.df['Class']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42, stratify=self.y)
        self.scaler=StandardScaler()
        self.scaler.fit(self.X_train[['Time','Amount']])
    def train_data(self):
        self.X_train[['Time','Amount']]=self.scaler.transform(self.X_train[['Time','Amount']])
        return self.X_train, self.y_train
    def test_data(self):
        self.X_test[['Time','Amount']]=self.scaler.transform(self.X_test[['Time','Amount']])
        return self.X_test, self.y_test