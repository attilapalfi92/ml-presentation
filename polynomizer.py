from sklearn.preprocessing import PolynomialFeatures
import pandas as pd


class Polynomizer:
    def __init__(self, dataset, degree=1):
        features_to_poly = dataset[['rooms', 'half_rooms', 'size', 'latitude', 'longitude']].values
        self.polynomialFeatures = PolynomialFeatures(degree=degree)
        self.polynomialFeatures.fit(features_to_poly)

    def transform(self, new_dataset):
        features_to_poly = new_dataset[['rooms', 'half_rooms', 'size', 'latitude', 'longitude']].values
        X_poly = self.polynomialFeatures.transform(features_to_poly)
        poly_dataset = pd.DataFrame(X_poly)

        dropped_dataset = new_dataset.drop('rooms', 1).drop('half_rooms', 1).drop('size', 1) \
            .drop('latitude', 1).drop('longitude', 1)
        return pd.concat([dropped_dataset, poly_dataset], axis=1)