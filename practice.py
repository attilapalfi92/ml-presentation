# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from plot_learning_curves import plot_learning_curve
from plot_regularizations import plot_regularizations


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
dataset = dataset.drop('settlement', 1) if 'settlement' in dataset.columns else dataset
dataset = dataset.drop('settlement_sub', 1) if 'settlement_sub' in dataset.columns else dataset
dataset = dataset.drop('location', 1) if 'location' in dataset.columns else dataset

dataset = dataset[~dataset.price.str.contains("milliárd")]
dataset['price'] = dataset['price'] \
            .apply(lambda p: p.replace(' millió Ft', '').replace(',', '.')) \
            .astype(float)

# imputing 
imp=SimpleImputer(strategy="median", fill_value = 0)
dataset["floor"]=imp.fit_transform(dataset[["floor"]]).ravel()

imp=SimpleImputer(strategy="median", fill_value = 0)
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


# processing categoric data
dataset['building_material'] = dataset['building_material'].astype('category')
dataset['comfort'] = dataset['comfort'].astype('category')
dataset['cond'] = dataset['cond'].astype('category')
dataset['heating'] = dataset['heating'].astype('category')
dataset['parking'] = dataset['parking'].astype('category')
dataset['sub_type'] = dataset['sub_type'].astype('category')
dataset['toilet'] = dataset['toilet'].astype('category')
dataset['location_accuracy'] = dataset['location_accuracy'].astype('category')

dataset = dataset.dropna()

y = dataset['price'].values
dataset = dataset.drop('id', 1).drop('price', 1)
X = dataset.values

# encoding categorical data
labelEncoders = {}
categoricalIndexes = []
categoricalColumns = []
for column in dataset.columns:
    if dataset[column].dtype.name == 'category':
        idx = dataset.columns.get_loc(column)
        name = dataset[column].name
        print(idx)
        print(name)
        categoricalColumns.append(name)
        categoricalIndexes.append(idx)
        label_encoder = LabelEncoder()
        X[:, idx] = label_encoder.fit_transform(X[:, idx].astype(str))
        labelEncoders[idx] = label_encoder

# option 1
oneHotEncoder = OneHotEncoder(categorical_features=categoricalIndexes)
X_onehot = oneHotEncoder.fit_transform(X).toarray()

# option 2
ct = ColumnTransformer(
    [('one_hot', OneHotEncoder(sparse=False), categoricalIndexes)],  # the column numbers I want to apply this to
    remainder='passthrough'  # This leaves the rest of my columns in place
)
X_ct = ct.fit_transform(X) # Notice the output is a string

# linear regression
estimator = Ridge(alpha=1)
standardScaler = StandardScaler()
X_train = standardScaler.fit_transform(X_onehot)
estimator.fit(X=X_train, y=y)

plot_learning_curve(estimator, 'lin reg', X, y)

# dev and test set


# sandbox:

dataset[column] = label_encoder.fit_transform(dataset[column].astype(str))

oneHotEncoder = OneHotEncoder(categorical_features=categoricalIndexes)
result = oneHotEncoder.fit_transform(dataset)





























