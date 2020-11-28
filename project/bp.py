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
TARGET = 'p'
# ===============Load Data ==========================


#3.
# ==============Data Exploration=====================

# check for number of entries
print(len(df))
# number of classes
classes = list( set( df[TARGET])) 
print("Classes-->", classes)
print("Number of Classes")
# distribution of classes
# Check for description
# identify for categorical columns
# Analyse floating point values
# Use KNN to analyse more on the data distribution which helps imputation in Data Preprocessing

# ==============Data Exploration=====================

#4.
# ==============Data Preprocessing===================

# Split data into Train and Test
# Data imputation if necessary separately on train and test dataset
# Data Normalization if necessary if the numbers appears to be very big
# One hot encode categorical values with Nan values true to encounter missing values

# ==============Data Preprocessing===================

#5.
# ==============Modeling=============================

# Decide on the Model
# List all the models that are relevant for the problem
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

# Run the imputation/preprocessing steps which were carried out for training data
# Run the test on model that came out best from above test on the test data.
# Save the model once the test results are satisfied.

# ==============Evaluate=============================

#8.
# ==============Load and Predict=====================

# Load the model that was saved in previous steps and predict for unseen data

# ==============Load and Predict=====================


