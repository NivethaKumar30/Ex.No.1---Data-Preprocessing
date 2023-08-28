
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#read the dataset
df=pd.read_csv('data.csv')
df
import pandas as pd
df=pd.read_csv("/content/data.csv")
df.head()

df.duplicated()

df.describe()

df.isnull().sum()

x=df.iloc[:, :-1].values
print(x)

y=df.iloc[:, -1].values
print(y)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1 = pd.DataFrame(scaler.fit_transform(df))
print(df1)

from sklearn.model_selection import train_test_split
xtrain,ytrain,xtest,ytest=train_test_split(x,y,test_size=0.2,random_state=2)
print(xtrain)
print(len(xtrain))
print(xtest)
print(len(xtest))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df1 = sc.fit_transform(df)
print(df1)
x=df.iloc[:, :-1].values
x
y=df.iloc[:, -1].values
y
print(df.isnull().sum)
df.fillna(df.mean().round(1), inplace=True)
print(df.isnull().sum())
y=df.iloc[:, -1].values
print(y)
df.duplicated()
print(df['Calories'].describe())
from sklearn.preprocessing import MinMaxScaler
Scaler = MinMaxScaler()
df1=pd.DataFrame(Scaler.fit_transform(df))
df1
from sklearn.model_selection import train_test_split
x_train, x_test ,y_train, y_test = train_test_split(x, y, test_size= 0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))

```

## OUTPUT:


![Screenshot 2023-08-28 161300](https://github.com/NivethaKumar30/Ex.No.1---Data-Preprocessing/assets/119559844/ca3bc8b9-b302-4989-8e2c-8e76de86379a)

![Screenshot 2023-08-28 161345](https://github.com/NivethaKumar30/Ex.No.1---Data-Preprocessing/assets/119559844/1f00d9a4-8177-49c1-8cee-218619c84858)

![Screenshot 2023-08-28 161416](https://github.com/NivethaKumar30/Ex.No.1---Data-Preprocessing/assets/119559844/1f8bc898-cce2-4bcf-a28c-fb8b3fbb9d71)

![Screenshot 2023-08-28 161428](https://github.com/NivethaKumar30/Ex.No.1---Data-Preprocessing/assets/119559844/34c70802-a329-497d-8052-19a57aa49338)

![Screenshot 2023-08-28 161445](https://github.com/NivethaKumar30/Ex.No.1---Data-Preprocessing/assets/119559844/9746b823-db48-40e9-a5a5-b0d340716443)

![Screenshot 2023-08-28 161528](https://github.com/NivethaKumar30/Ex.No.1---Data-Preprocessing/assets/119559844/9ecc469c-1469-4f9a-a219-06e53cc535ae)

![Screenshot 2023-08-28 161528](https://github.com/NivethaKumar30/Ex.No.1---Data-Preprocessing/assets/119559844/b8c93fdd-f11f-4508-a958-187290835b24)

![Screenshot 2023-08-28 161535](https://github.com/NivethaKumar30/Ex.No.1---Data-Preprocessing/assets/119559844/7c7524dc-c6a7-4e40-b223-97d15769875e)

![Screenshot 2023-08-28 161547](https://github.com/NivethaKumar30/Ex.No.1---Data-Preprocessing/assets/119559844/0e22c5e0-39ec-4e46-ba09-85d402957dde)



## RESULT
/Type your result here/
