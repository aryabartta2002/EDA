#------------------------------------- Outlier Treatment ---------------------------

import pandas as pd  # for data manupulation
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for advance data visualization
from feature_engine.outliers import Winsorizer  # for outlier treatment

data = pd.read_csv(r"C:\Users\User\Desktop\DATA SCIENCE\EDA\EDA\Problem Statement\dataset\Boston.csv") # loading data

data.dtypes  # data types of all columns

data.describe()  # gives key statistical insight

sns.boxplot(data.crim); plt.show()   # extream outliers 
sns.boxplot(data.zn); plt.show()     # extream outliers
sns.boxplot(data.indus); plt.show()  # no outliers
sns.boxplot(data.chas); plt.show()   # exception
sns.boxplot(data.nox); plt.show()    # no outliers
sns.boxplot(data.rm); plt.show()     # outliers present
sns.boxplot(data.age); plt.show()    # no outliers 
sns.boxplot(data.dis); plt.show()    # outliers present
sns.boxplot(data.rad); plt.show()    # no outliers
sns.boxplot(data.tax); plt.show()    # no outliers
sns.boxplot(data.ptratio); plt.show() # no outliers
sns.boxplot(data.black); plt.show()   # outliers present
sns.boxplot(data.lstat); plt.show()   # outliers present
sns.boxplot(data.medv); plt.show()    # outliers present


win_crim = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['crim'])
c_f = win_crim.fit_transform(data[['crim']])
sns.boxplot(c_f.crim)
plt.show()


win_zn = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['zn'])
z_f = win_zn.fit_transform(data[['zn']])
sns.boxplot(z_f.zn)
plt.show()


win_rm = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['rm'])
rm_fit = win_rm.fit_transform(data[['rm']])
sns.boxplot(rm_fit.rm)
plt.show()


win_dis = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['dis'])
dis_fit = win_dis.fit_transform(data[['dis']])
sns.boxplot(dis_fit.dis)
plt.show()


win_b = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['black'])
b_fit = win_b.fit_transform(data[['black']])
sns.boxplot(b_fit.black)
plt.show()


win_lstat = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['lstat'])
l_f = win_lstat.fit_transform(data[['lstat']])
sns.boxplot(l_f.lstat)
plt.show()


win_medv = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['medv'])
m_f = win_medv.fit_transform(data[['medv']])
sns.boxplot(m_f.medv)
plt.show()