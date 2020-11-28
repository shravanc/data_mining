#1.
# ==============Importing Libraries====================
import pandas as pd


# ==============Importing Libraries====================


#2.
# ===============Load Data ==========================

# use dataframe and view head and tail for understanding the data
#df = pd.read_csv('mushroom.csv', header=None)
df = pd.read_csv('mushroom.csv')
#print(list(df.columns))
print(df.head())
print(df.tail())
TARGET = list(df.columns)[0]
# ===============Load Data ==========================


#3.
# ==============Data Exploration=====================

# check for number of entries
print(len(df))
# number of classes
classes = list( set( df[TARGET])) 
print("Classes------------->", classes)
print("Number of Classes--->", len(classes))

# distribution of classes
print(df[TARGET].value_counts())

# Check for missing data
print(df.isna().sum())

# Check for description
print(df[TARGET].describe())

# From the above it is seen that Classes are almost equally distributed. Good number of samples are present for both the cases.

# identify for categorical columns
print(df.head())
# Allmost all the columns are categorical
categorical_columns = list(df.columns)[1:]
print(categorical_columns)
# Analyse floating point values
# There are not Floating point values to analyse

# ==============Data Exploration=====================

#4.
# ==============Data Preprocessing===================
from sklearn.model_selection import train_test_split, GridSearchCV

# Converting target to numerical values. 1 represent edible and 0 represent poisonous
mapper = {'e': 1, 'p': 0}
df[TARGET] = df[TARGET].replace(mapper)

# One hot encode categorical values with Nan values true to encounter missing values
df = pd.get_dummies(df, columns=categorical_columns,
                        dummy_na=True,
                        drop_first=True)

print(df)

import matplotlib.pyplot as plt
import numpy as np

# Split data into Train and Test
y = df[TARGET]
X = df.drop(columns=TARGET, axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
print("Training data count---->", len(X_train))
print("Testing data count----->", len(X_test))


# ==============Data Preprocessing===================

#5.
# ==============Modeling=============================
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import make_scorer
from sklearn.dummy import DummyClassifier

# calculate precision-recall area under curve
def pr_auc(y_true, probas_pred):
	# calculate precision-recall curve
	p, r, _ = precision_recall_curve(y_true, probas_pred)
	# calculate area under curve
	return auc(r, p)

# evaluate a model
def evaluate_model(X, y, model):
	# define evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# define the model evaluation the metric
	metric = make_scorer(pr_auc, needs_proba=True)
	# evaluate model
	scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
	return scores

# define the reference model
model = DummyClassifier(strategy='constant', constant=1)
# evaluate the model
scores = evaluate_model(X, y, model)
# summarize performance
print('Mean PR AUC: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))



# Decide on the Model
# List all the models that are relevant for the problem
# define models to test

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def get_models():
	models, names = list(), list()
	# CART
	models.append(DecisionTreeClassifier())
	names.append('CART')
	# KNN
	steps = [('s',StandardScaler()),('m',KNeighborsClassifier())]
	models.append(Pipeline(steps=steps))
	names.append('KNN')
	# Bagging
	models.append(BaggingClassifier(n_estimators=100))
	names.append('BAG')
	# RF
	models.append(RandomForestClassifier(n_estimators=100))
	names.append('RF')
	# ET
	models.append(ExtraTreesClassifier(n_estimators=100))
	names.append('ET')
	return models, names

models, names = get_models()
results = list()
# evaluate each model
for i in range(len(models)):
	# evaluate the model and store results
	scores = evaluate_model(X, y, models[i])
	results.append(scores)
	# summarize performance
	print('>%s %.3f (%.3f)' % (names[i], np.mean(scores), np.std(scores)))
# plot the results
plt.boxplot(results, labels=names, showmeans=True)
plt.show()


# Write a function to iterate over all models
# K-fold training could be one way to train the model if not deep learning

# ==============Modeling=============================

#6.
# ==============Analysing the Model==================


# Analyse the mertics of the modeling
# plot relevant curves for the best model
# Plot ROC and AUC curve to analyse for Binary classification problem
# Fine tune the model on above observation and repeat steps till the metrics is satisfied

# ==============Analysing the Model==================

#7.
# ==============Evaluate=============================


# define model to evaluate
model = KNeighborsClassifier()
# scale, then fit model
pipeline = Pipeline(steps=[('s',StandardScaler()), ('m',model)])
pipeline.fit(X, y)

row = X_train[0]
print(row)
print("length--->", len(row))
yhat = pipeline.predict_proba([row])
# get the probability for the positive class
result = yhat[0][1]
print("Result--->", result, "Expected--->", y_train[0])
# Run the imputation/preprocessing steps which were carried out for training data
# Run the test on model that came out best from above test on the test data.
# Save the model once the test results are satisfied.

# ==============Evaluate=============================

#8.
# ==============Load and Predict=====================

# Load the model that was saved in previous steps and predict for unseen data

# ==============Load and Predict=====================


