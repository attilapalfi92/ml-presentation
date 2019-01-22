import pandas as pd
from numeric_data_processor import NumericDataProcessor
from categorical_data_processor import CategoricalDataProcessor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


class HousingPricePredictor:
    def __init__(self, initial_dataset,
                 estimator=Ridge(alpha=1)):

        self.estimator = estimator

        # numeric data
        self.numericDataProcessor = NumericDataProcessor(initial_dataset)
        dataset = self.numericDataProcessor.getDataset()

        # categorical data
        self.categoricalDataProcessor = CategoricalDataProcessor(dataset)
        X = self.categoricalDataProcessor.getX()
        y = self.categoricalDataProcessor.getY()
        
        # feature scaling
        self.standardScaler = StandardScaler()
        X_train = self.standardScaler.fit_transform(X)
        self.estimator.fit(X=X_train, y=y)

    def predict(self, dataset):
        dataset = self.numericDataProcessor.process(dataset)
        X = self.categoricalDataProcessor.transform(dataset)
        X_scaled = self.standardScaler.transform(X)
        return self.estimator.predict(X_scaled)
    


dataset = pd.read_csv("flats.csv")
dataset = dataset.sample(frac=1).reset_index(drop=True)
test_set = dataset.tail(100).drop('id', 1).reset_index(drop=True)
training_set = dataset[:-100]

predictor = HousingPricePredictor(training_set)
results = predictor.predict(test_set.copy(deep=True))




flats = pd.read_csv("flats.csv")
predictor = HousingPricePredictor(flats)

new_flats = pd.read_csv("new_flats.csv")
result = predictor.predict(new_flats)
