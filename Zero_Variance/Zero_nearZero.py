#------------------------------------------------ Zero_NearZero --------------------------------

import pandas as pd  # data manupulation
import numpy as np

df = pd.read_csv(r"C:\Users\User\Desktop\DATA SCIENCE\EDA\EDA\Problem Statement\dataset\Z_dataset.csv")  # load data

df.dtypes  # ststistical insight

df.Id = df.Id.astype('str') # type casting

# zero - near zero variance

numeric_columns = df.select_dtypes(include = np.number) # select only numeric column

numeric_columns.var() # Calculating the variance of each numeric variable in the DataFrame


# Checking if the variance of each numeric variable is equal to 0 and returning a boolean Series
numeric_columns.var() == 0 

# Checking if the variance of each numeric variable along axis 0 (columns) is equal to 0 and returning a boolean Series
numeric_columns.var(axis = 0) == 0 

