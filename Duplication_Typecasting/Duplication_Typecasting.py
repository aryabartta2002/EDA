#----------------------------------- Duplication_Typecasting ------------------------------


import pandas as pd     # data manupulation
import matplotlib.pyplot as plt # data visualization
import seaborn as sns  # advance data visualization

data = pd.read_csv(r"C:\Users\User\Desktop\DATA SCIENCE\EDA\EDA\Problem Statement\dataset\Online Retail.csv", encoding='unicode_escape')  # load data


data.dtypes

# TypeCasting

data.CustomerID = data.CustomerID.astype('str')
data.UnitPrice = data.UnitPrice.astype('int64')

data.dtypes


# Duplicates

duplicate = data.duplicated(keep = 'last')
sum(duplicate)

drop_duplicate = data.drop_duplicates(keep = 'last')




# EDA


# First Moment Business Decision

data.Quantity.mean()
data.Quantity.median()
data.Quantity.mode()


data.UnitPrice.mean()
data.UnitPrice.median()
data.UnitPrice.mode()


# Second Moment Business Decision

data.Quantity.var()
data.Quantity.std()
range1 = max(data.Quantity) - min(data.Quantity)
print(range1)


data.UnitPrice.var()
data.UnitPrice.std()
range2 = max(data.UnitPrice) - min(data.UnitPrice)
print(range2)


# Graphical Techniques


# Histogram


plt.hist(data.Quantity)
plt.show()

plt.hist(data.UnitPrice)
plt.show()


# Boxplot

sns.boxplot(data.Quantity); plt.show()
sns.boxplot(data.UnitPrice); plt.show()


# Scatter Plot

plt.scatter(x = data['Quantity'], y = data['UnitPrice'])
plt.xlabel('Quantity')
plt.ylabel('UnitPrice')
plt.title('Scatter Plot')
plt.show()


# Third Moment Business Decision

data.Quantity.skew() # Left Skewed
data.UnitPrice.skew() # Right Skewed


# Forth Moment Business Decision

data.Quantity.kurt()  # >3 means Leptokurtic
data.UnitPrice.kurt() # >3 means Leptokurtic
