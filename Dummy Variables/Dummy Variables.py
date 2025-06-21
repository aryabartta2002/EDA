#------------------------------------------ Dummy Variables ------------------------


import pandas as pd # data manupulation

df = pd.read_csv(r"C:\Users\User\Desktop\DATA SCIENCE\EDA\EDA\dataset\Animal_category.csv") # load data

df.info() # information about dataset

df_new = pd.get_dummies(df).astype('int64')  # get dummy from dataset


df_new_1 = pd.get_dummies(df, drop_first = True).astype('int64') # droping 1st category from dummy
