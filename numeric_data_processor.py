from polynomizer import Polynomizer
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd


class NumericDataProcessor:
    def _impute_(self, dataset, column_name):
        imputer = SimpleImputer(strategy="median")
        dataset[column_name]=imputer.fit_transform(dataset[[column_name]]).ravel()
        idx = dataset.columns.get_loc(column_name)        
        return dataset, imputer, idx

    def __init__(self, dataset, poly_degree=1):
        dataset = self.prepareDataset(dataset)

        # keys are column indexes
        self.imputers = {}
        dataset, imputer, i = self._impute_(dataset, 'floor')
        self.imputers[i] = imputer
        dataset, imputer, i = self._impute_(dataset, 'building_levels')
        self.imputers[i] = imputer
        

        # adding polynomial features
        self.polynomizer = Polynomizer(dataset)
        self.dataset = self.polynomizer.transform(dataset)

    def process(self, dataset):
        dataset = self.prepareDataset(dataset)
        X = dataset.values
        for index, imputer in self.imputers.items():
            X[:, index:index + 1] = imputer.transform(X[:, index:index + 1])

        dataset = pd.DataFrame(X, columns=dataset.columns)
        return self.polynomizer.transform(dataset)

    def prepareDataset(self, dataset):
        # clearing dataset
        dataset['floor'] = dataset['floor'].astype(str) \
            .apply(lambda f: f.replace('nincs megadva', 'NaN'))
            
        dataset['floor'] = dataset['floor'].astype(str) \
            .apply(lambda f: f.replace('félemelet', '0.5'))
            
        dataset['floor'] = dataset['floor'].astype(str) \
            .apply(lambda f: f.replace('szuterén', '-1'))
        
        dataset['floor'] = dataset['floor'].astype(str) \
            .apply(lambda f: f.replace('földszint', '0'))
            
        dataset['floor'] = dataset['floor'].astype(str) \
            .apply(lambda f: f.replace('10 felett', '0')).astype(float)
        
        dataset['building_levels'] = dataset['building_levels'].astype(str) \
            .apply(lambda f: f.replace('földszintes', '1'))
        
        dataset['building_levels'] = dataset['building_levels'].astype(str) \
            .apply(lambda f: f.replace('több mint 10', '11'))
            
        dataset['building_levels'] = dataset['building_levels'].astype(str) \
            .apply(lambda f: f.replace('nincs megadva', 'NaN')).astype(float)
            
        room_data = dataset['rooms'].str.split(' \+ ', 1, expand=True)
        
        dataset['rooms'] = room_data[0].astype(str) \
            .apply(lambda hr: hr.replace(' fél', '')).astype(float)
            
        dataset['half_rooms'] = room_data[1].astype(str) \
            .apply(lambda hr: hr.replace(' fél', '')) \
            .apply(lambda hr: hr.replace('None', '0')).astype(float)
            
        dataset['size'] = dataset['size'].astype(str) \
            .apply(lambda s: s.replace(' m²', '')).apply(lambda s: s.replace(' ', '')) \
            .astype(float)
    
        # some more clearing
        dataset = dataset[dataset['size'] > 20]
        dataset = dataset[dataset['size'] < 250]
        dataset = dataset[dataset['rooms'] < 10]
        dataset = dataset[dataset['half_rooms'] < 10]
        dataset = dataset[np.isfinite(dataset['longitude'])]
        dataset = dataset[np.isfinite(dataset['latitude'])]
        
        dataset = dataset.drop('id', 1) if 'id' in dataset.columns else dataset
        dataset = dataset.drop('ac', 1) if 'ac' in dataset.columns else dataset
        dataset = dataset.drop('attic', 1) if 'attic' in dataset.columns else dataset
        dataset = dataset.drop('barrier_free', 1) if 'barrier_free' in dataset.columns else dataset
        dataset = dataset.drop('build_year', 1) if 'build_year' in dataset.columns else dataset
        dataset = dataset.drop('energy_cert', 1) if 'energy_cert' in dataset.columns else dataset
        dataset = dataset.drop('garden_connected', 1) if 'garden_connected' in dataset.columns else dataset
        dataset = dataset.drop('settlement', 1) if 'settlement' in dataset.columns else dataset
        dataset = dataset.drop('settlement_sub', 1) if 'settlement_sub' in dataset.columns else dataset
        dataset = dataset.drop('location', 1) if 'location' in dataset.columns else dataset
        
        if 'price' in dataset.columns:
            dataset = dataset[~dataset.price.str.contains("milliárd")]
            dataset['price'] = dataset['price'] \
                .apply(lambda p: p.replace(' millió Ft', '').replace(',', '.')) \
                .astype(float)

        return dataset
    
    def getDataset(self):
        return self.dataset