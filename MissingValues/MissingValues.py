# -------------------------------------------- MissingValues -----------------------------

import pandas as pd  # Data manupulation
import numpy as np   # Numerical Calculation
from sklearn.impute import SimpleImputer # handle missing value 

df = pd.read_csv(r"C:\Users\User\Desktop\DATA SCIENCE\EDA\EDA\dataset\claimants.csv")

df.info()
df.describe()
df.dtypes
df.head()


df.isna().sum() # count the missing values from each columns


'''
CLMSEX       12
CLMINSUR     41
SEATBELT     48
CLMAGE      189
'''

mean_imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean') # commanding to fill nan value with mean value
df["CLMSEX"] = pd.DataFrame(mean_imputer.fit_transform(df[["CLMSEX"]]))  # transforming CLMSEX to mean_imputer
df["CLMSEX"].isna().sum()  # Checking for any remaining missing values in 'CLMSEX'

df['CLMINSUR'] = pd.DataFrame(mean_imputer.fit_transform(df[['CLMINSUR']]))
df['CLMINSUR'].isna().sum()


df['SEATBELT'] = pd.DataFrame(mean_imputer.fit_transform(df[['SEATBELT']]))
df['SEATBELT'].isna().sum()


df['CLMAGE'] = pd.DataFrame(mean_imputer.fit_transform(df[['CLMAGE']]))
df['CLMAGE'].isna().sum()
