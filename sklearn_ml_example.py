import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Random Data
from sklearn.ensemble import RandomForestRegressor

#cross-validation rools
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

#metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')
#print(data.head())
#print(data.shape)
#print(data.describe())


# Split data into training and test sets

Y = data.quality
X = data.drop('quality', axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 123, stratify=Y)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_test)
print(X_train_scaled.mean(axis=0))
print(X_train_scaled.std(axis=0))

pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

print(pipeline.get_params())

hyperparameters = {'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'], 'randomforestregressor__max_depth': [None, 5,3,1]}

clf = GridSearchCV(pipeline, hyperparameters, cv = 10)

#Fit and tune model
clf.fit(X_train, Y_train)

print(clf.best_params_)

print(clf.refit)

y_pred = clf.predict(X_test)

print(r2_score(y_test, y_pred))

print(mean_squared_error(y_test, y_pred))

joblib.dump(clf, 'rf_regressor.pkl')
