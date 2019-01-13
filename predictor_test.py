from production.housing_price_predictor import HousingPricePredictor
import pandas as pd
from sklearn.utils import shuffle


# Importing da dataset
initial_dataset = pd.read_csv("../data/flats.csv")
dataset = initial_dataset.sample(frac=1).reset_index(drop=True)
test_set = dataset.tail(100).drop('id', 1).drop('price', 1)
dataset = dataset[:-100]

predictor = HousingPricePredictor(dataset)
results = predictor.predict(test_set)

print(results)

