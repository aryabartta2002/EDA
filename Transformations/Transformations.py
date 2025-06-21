#------------------------------------- Transformations ----------------------------

import pandas as pd
from sklearn.preprocessing import StandardScaler


df = pd.read_csv(r"C:\Users\User\Desktop\DATA SCIENCE\EDA\EDA\dataset\calories_consumed.csv") # Load the dataset


df.columns = ['weight_gained_grams', 'calories_consumed'] # Rename columns

df['calories_per_gram'] = df['calories_consumed'] / df['weight_gained_grams'] # Add derived feature


df.dropna(inplace=True) # Handle missing values (if any)

# Normalize the numerical columns
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[['weight_gained_grams', 'calories_consumed', 'calories_per_gram']] = scaler.fit_transform(
    df_scaled[['weight_gained_grams', 'calories_consumed', 'calories_per_gram']]
)

# Create bins for weight gain
df['weight_category'] = pd.cut(df['weight_gained_grams'],
                                bins=[0, 50, 100, df['weight_gained_grams'].max()],
                                labels=['Low', 'Medium', 'High'])

# View final transformed DataFrame
print(df.head())
