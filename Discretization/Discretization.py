#----------------------------------------- DISCRETIZATION ----------------------------------

import pandas as pd

df = pd.read_csv(r"C:\Users\User\Desktop\DATA SCIENCE\EDA\EDA\dataset\iris.csv")

df.shape # dimension of dataframe
df.columns # show all columns
df.info()  # gives total information of dataset
df.describe # gives statistics insight
df.head() # gives first five rows by default
df.tail() # gives last five rows by default

df['Sl_new1'] = pd.cut(df['Sepal.Length'], 
                              bins = [min(df['Sepal.Length']), df['Sepal.Length'].mean(), max(df['Sepal.Length'])], 
                              include_lowest = True,
                              labels = ["Low", "High"])
df.Sl_new1.value_counts()


df['pl_new'] = pd.cut(df['Sepal.Length'],
                      bins = [min(df['Sepal.Length']), df['Sepal.Length'].mean(), max(df['Sepal.Length'])],
                      include_lowest = True,
                      labels = ['Low', 'High'])

df.pl_new.value_counts()




df['pw_new'] = pd.cut(df['Petal.Width'],
                      bins = [min(df['Petal.Width']), df['Petal.Width'].mean(), max(df['Petal.Width'])],
                      include_lowest = True,
                      labels = ['Low', 'High'])

df.pw_new.value_counts()




df['pw_new'] = pd.cut(df['Petal.Width'],
                      bins = [min(df['Petal.Width']), df['Petal.Width'].mean(), max(df['Petal.Width'])],
                      include_lowest = True,
                      labels = ['Low', 'High'])

df.pw_new.value_counts()

