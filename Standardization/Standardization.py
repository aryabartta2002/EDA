#------------------------------------ Standardization ---------------------------------------

import pandas as pd # data manupulation
from sklearn.preprocessing import StandardScaler  # import for standardization 


data = pd.read_csv(r"C:\Users\User\Desktop\DATA SCIENCE\EDA\EDA\dataset\Seeds_data.csv") # load data

data.describe()

scaler = StandardScaler() # Initialise the StandardScaler
df = scaler.fit_transform(data) # Scaling the data using StandardScaler

dataset = pd.DataFrame(df) # Converting the scaled array back to a DataFrame


res = dataset.describe() # Generating descriptive statistics of the scaled data

