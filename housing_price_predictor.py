from production.numeric_data_processor import NumericDataProcessor
from production.categorical_data_processor import CategoricalDataProcessor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


class HousingPricePredictor:
    def __init__(self, initial_dataset,
                 estimator=Ridge(alpha=1)):

        self.estimator = estimator

        # numeric data
        self.numericDataProcessor = NumericDataProcessor(initial_dataset)
        dataset = self.numericDataProcessor.getDataset()
        y = self.numericDataProcessor.getY()

        # categorical data
        self.categoricalDataProcessor = CategoricalDataProcessor(dataset)
        X = self.categoricalDataProcessor.getX()

        # feature scaling
        self.standardScaler = StandardScaler()
        X_train = self.standardScaler.fit_transform(X)
        self.estimator.fit(X=X_train, y=y)

    def predict(self, dataset):
        dataset = self.numericDataProcessor.process(dataset)
        X = self.categoricalDataProcessor.transform(dataset)
        X_scaled = self.standardScaler.transform(X)
        return self.estimator.predict(X_scaled)
    
    
