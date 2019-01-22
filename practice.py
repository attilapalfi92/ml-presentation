# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve

import matplotlib.pyplot as plt


from plot_learning_curves import plot_learning_curve


dataset = pd.read_csv("flats.csv")

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
imp = SimpleImputer(strategy="median")
dataset["floor"] = imp.fit_transform(dataset[["floor"]]).ravel()

imp = SimpleImputer(strategy="median")
dataset["building_levels"] = imp.fit_transform(dataset[["building_levels"]]).ravel()


# polynomial features
features_to_poly = dataset[['rooms', 'half_rooms', 'size', 'latitude', 'longitude']].values
polynomialFeatures = PolynomialFeatures(degree=1)
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
for column in dataset.columns:
    if dataset[column].dtype.name == 'category':
        idx = dataset.columns.get_loc(column)
        name = dataset[column].name
        print(idx)
        print(name)
        categoricalIndexes.append(idx)
        label_encoder = LabelEncoder()
        X[:, idx] = label_encoder.fit_transform(X[:, idx].astype(str))
        labelEncoders[idx] = label_encoder

# onehot
oneHotEncoder = OneHotEncoder(categorical_features=categoricalIndexes)
X_onehot = oneHotEncoder.fit_transform(X).toarray()



# train and test linear regression
# splitting into train and test set
X_train, X_test, y_train, y_test = train_test_split(X_onehot, y, test_size = 0.1)

# feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# fitting regression
regressor = Ridge(alpha = 1) # regularized linear regression
regressor.fit(X = X_train, y = y_train)

# prediction
y_pred = regressor.predict(X_test)






# evaluating
train_score = regressor.score(X_train, y_train)
print('train_score=%s' % train_score)
test_score = regressor.score(X_test, y_test)
print('test_score=%s' % test_score)
scores = cross_val_score(regressor, X=X_train, y=y_train, scoring="neg_mean_squared_error")
print('scores=%s' % scores)
J_test = mean_squared_error(y_test, y_pred)
print('J_test=%s' % J_test)



# plotting

train_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 60000]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_onehot)

estimator = Ridge(alpha = 1)
train_sizes, train_scores, validation_scores = learning_curve(
                                                   estimator = regressor, X = X_train,
                                                   y = y_train, train_sizes = train_sizes, cv = 16,
                                                   scoring = 'neg_mean_squared_error')

print('Training scores:\n\n', train_scores)
print('\n', '-' * 70) # separator to make the output easy to read
print('\nValidation scores:\n\n', validation_scores)

train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)

print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
print('\n', '-' * 20) # separator
print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))

plt.style.use('seaborn')

plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a linear regression model', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(0,2000)





plot_learning_curve(regressor, 'lin reg', X, y)






# sandbox:

# option 2
ct = ColumnTransformer(
    [('one_hot', OneHotEncoder(sparse=False), categoricalIndexes)],  # the column numbers I want to apply this to
    remainder='passthrough'  # This leaves the rest of my columns in place
)
X_ct = ct.fit_transform(X) # Notice the output is a string




























