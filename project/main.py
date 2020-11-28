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
	#return models, names
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
  record_metrics(model)

plot the results
plt.boxplot(results, labels=names, showmeans=True)
plt.show()

print("lr_tpr_list--->", lr_tpr_list)

plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')

for i in range(len(models)):

  lr_probs  = lr_probs_list[i] #model.predict_proba(X_test)[:, 1]
  lr_auc    = lr_auc_list[i]   #roc_auc_score(y_test, lr_probs)

  #lr_fpr, lr_tpr = lr_fpr_list[i], lr_tpr_list[i] #roc_curve(y_test, lr_probs)
  
  plt.plot(lr_fpr_list[i], lr_tpr_list[i], marker='.', label=names[i])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend()

#plt.show() 


# Write a function to iterate over all models
# K-fold training could be one way to train the model if not deep learning

# ==============Modeling=============================

#6.
# ==============Evaluate=============================


# define model to evaluate
yhat = pipeline.predict_proba([row])
# get the probability for the positive class
result = yhat[0][1]
print("Result--->", result, "Expected--->", y_train[0])
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_fscore_support, roc_auc_score

mean_fpr = np.linspace(start=0, stop=1, num=100)

ns_probs = [0 for _ in range(len(y_test))]

for i in range(len(models)):
  model = models[i]
  label = names[i]
  model.fit(X_train, y_train)
  lr_probs = model.predict_proba(X_test)[:, 1]

  lr_auc = roc_auc_score(y_test, lr_probs)


  lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

  plt.plot(lr_fpr, lr_tpr, marker='.', label=label)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend()

plt.show()
# Run the imputation/preprocessing steps which were carried out for training data
# Run the test on model that came out best from above test on the test data.
# Save the model once the test results are satisfied.

# ==============Evaluate=============================

#8.
# ==============Load and Predict=====================

# Load the model that was saved in previous steps and predict for unseen data

# ==============Load and Predict=====================




"""

from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_fscore_support, roc_auc_score

mean_fpr = np.linspace(start=0, stop=1, num=100)

ns_probs = [0 for _ in range(len(y_test))]

lr_probs = model.predict_proba(X_test)[:, 1]

ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)

print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))

ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend()

plt.show()


"""

"""
_predicted_values = model.predict(X_test)
_accuracy = accuracy_score(y_test, _predicted_values)
print("Accuracy Test Data--->", _accuracy)

_precision, _recall, _f1_score, _ = precision_recall_fscore_support(y_test, _predicted_values, labels=[1])

_fpr, _tpr, _ = roc_curve(y_test, _probabilities)


KNN_trp = _tpr
KNN_auc = _accuracy

print("length--->", len(KNN_trp[0:]))
print("hello---->", KNN_trp[0:])
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)
plt.plot(mean_fpr, KNN_trp[0,:], lw=2, color='blue', label='SVM (AUC = %0.2f)' % (KNN_auc), alpha=0.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves for multiple classifiers')
plt.legend(loc="lower right")
plt.show()


"""
