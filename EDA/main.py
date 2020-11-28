import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('./loans_train.csv')

print(df.head(5))

print(df.tail(5))


print(df.describe())

sns.heatmap(df.corr())
