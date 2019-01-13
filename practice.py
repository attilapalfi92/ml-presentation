# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


dataset = pd.read_csv("flats.csv")

# clearing dataset
dataset['floor'] = dataset['floor'] \
    .apply(lambda f: f.replace('nincs megadva', 'NaN'))
    
dataset['floor'] = dataset['floor'] \
    .apply(lambda f: f.replace('félemelet', '0.5'))
    
dataset['floor'] = dataset['floor'] \
    .apply(lambda f: f.replace('szuterén', '-1'))

dataset['floor'] = dataset['floor'] \
    .apply(lambda f: f.replace('földszint', '0'))
    
dataset['floor'] = dataset['floor'] \
    .apply(lambda f: f.replace('10 felett', '0')).astype(float)

dataset['building_levels'] = dataset['building_levels'] \
    .apply(lambda f: f.replace('földszintes', '1'))

dataset['building_levels'] = dataset['building_levels'] \
    .apply(lambda f: f.replace('több mint 10', '11'))
    
dataset['building_levels'] = dataset['building_levels'] \
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

dataset = dataset.drop('ac', 1) if 'ac' in dataset.columns else dataset
dataset = dataset.drop('attic', 1) if 'attic' in dataset.columns else dataset
dataset = dataset.drop('barrier_free', 1) if 'barrier_free' in dataset.columns else dataset
dataset = dataset.drop('build_year', 1) if 'build_year' in dataset.columns else dataset
dataset = dataset.drop('energy_cert', 1) if 'energy_cert' in dataset.columns else dataset
dataset = dataset.drop('garden_connected', 1) if 'garden_connected' in dataset.columns else dataset

dataset = dataset[~dataset.price.str.contains("milliárd")]
dataset['price'] = dataset['price'] \
            .apply(lambda p: p.replace(' millió Ft', '').replace(',', '.')) \
            .astype(float)


# imputing 
imp=Imputer(missing_values="NaN", strategy="mean" )
dataset["floor"]=imp.fit_transform(dataset[["floor"]]).ravel()

imp=Imputer(missing_values="NaN", strategy="mean" )
dataset["building_levels"]=imp.fit_transform(dataset[["building_levels"]]).ravel()


# polynomial features
features_to_poly = dataset[['rooms', 'half_rooms', 'size', 'latitude', 'longitude']].values
polynomialFeatures = PolynomialFeatures(degree=3)
polynomialFeatures.fit(features_to_poly)

X_poly = polynomialFeatures.transform(features_to_poly)
poly_dataset = pd.DataFrame(X_poly)

dropped_dataset = dataset.drop('rooms', 1).drop('half_rooms', 1).drop('size', 1) \
    .drop('latitude', 1).drop('longitude', 1)
dataset = pd.concat([dropped_dataset, poly_dataset], axis=1)


# saving X and y variables
y = dataset['price'].values
dataset = dataset.drop('id', 1).drop('price', 1)
X = dataset.values


# processing categoric data
dataset['building_material'] = dataset['building_material'].astype('category')
dataset['comfort'] = dataset['comfort'].astype('category')
dataset['cond'] = dataset['cond'].astype('category')
dataset['heating'] = dataset['heating'].astype('category')
dataset['parking'] = dataset['parking'].astype('category')
dataset['sub_type'] = dataset['sub_type'].astype('category')
dataset['toilet'] = dataset['toilet'].astype('category')
dataset['location_accuracy'] = dataset['location_accuracy'].astype('category')


# encoding categorical data
categoricalIndexes = []
labelEncoders = {}
for column in dataset.columns:
    if dataset[column].dtype.name == 'category':
        idx = dataset.columns.get_loc(column)
        categoricalIndexes.append(idx)
        label_encoder = LabelEncoder()
        X[:, idx] = label_encoder.fit_transform(X[:, idx])
        labelEncoders[idx] = label_encoder

# libs take care of dummy variable trap so I don't have to
# (otherwise should drop first column for each category)
oneHotEncoder = OneHotEncoder(categorical_features=categoricalIndexes)
X = oneHotEncoder.fit_transform(X).toarray()

X = dataset.values
for idx in categoricalIndexes:
    X[:, idx] = labelEncoders[idx].transform(X[:, idx])
X = oneHotEncoder.transform(X).toarray()


# sandbox:



























