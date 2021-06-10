
# Data Processing in AWS 

Data processing is one of the first steps of the machine learning pipeline. As different sources of data have different formats, it becomes almost impossible to handle all the formats inside the model. Hence, we give the data a synchronous structure, and then we try to process different unwanted sections of it. These sections include the null values, outliers, dummification of categorical columns, standardization of numerical columns, etc. We can use SageMaker effectively to process the data in all these domains.


## Preprocessing in Jupyter Notebook

In between receiving the raw data and feeding the data to the model, there are a lot of steps the data goes through. These steps are the data processing steps. Data processing includes feature selection, feature transformation, feature imputation, feature normalization, etc. Once all these steps are done, we proceed to splitting the data into a training set and a validation set, which are finally given to the model.

In this section, we will be looking at some of the basic data processing steps that we can follow.

1. Loading the raw data

2. Imputing the null values, which means how to replace the null values with some actual values 

3. Splitting the data into categorical and numerical data frames

4. “Dummifying” categorical data

5. Imputing the remaining null values

6. Concatenating the categorical and numerical data frames

7. Normalizing the final data frame

8. Splitting the data into train and validation sets


This part assumes that you have hands-on knowledge of Pandas, Numpy, and Scikit-Learn. These packages are required for the data processing steps. If not, then it is recommended that you explore these packages to get some hands-on experience before moving on to learning SageMaker.

The dataset that we will be using for processing is the Big Mart sales dataset, which can be downloaded from Kaggle at  www.kaggle.com/devashish0507/big-mart-sales-prediction 


This dataset contains a lot of information related to the sales of items in a retail shop.

The task is to predict the sales of items. We will not be looking at the prediction part in this chapter. Rather, we’ll be exploring only the data processing part of the process. Let’s start by reading the train file using the Pandas framework.


```py

import pandas as pd

data = pd.read_csv("Train.csv")

```

Now the entire CSV sheet’s columns are saved in a data frame object named data. 

Next, let’s explore the top five rows of the dataset.

```py

data.head()

```


![2](https://user-images.githubusercontent.com/23625821/121323970-92885500-c910-11eb-9b9b-72f923e06912.png)


```py

print(data.shape)

print("*****************************************************************")

print(data .columns)

```

![1](https://user-images.githubusercontent.com/23625821/121324152-c1063000-c910-11eb-955a-4b025125cb67.png)



As we can see, there are 8,523 rows and 12 columns. Also, we can see the names of all the columns in the list given.

As we have seen in the steps of processing, the next step is to impute the null values. So, let’s take a look at all the columns that have null values.


```py

data.isna().sum()

```

![3](https://user-images.githubusercontent.com/23625821/121324442-062a6200-c911-11eb-93ee-95450de0f787.png)


So, there are two columns with null values: Item_Weight and Outlet_Size. We can use the normal imputation methods provided by Scikit-Learn to impute these null values. But, instead, we will be using the help of nearby columns to fill in these null values. Let’s look at the data types of these columns, as that is going to help us in making imputation strategies.

```py

print(data['Item_Weight'].dtype)

print(data['Outlet_Size'].dtype)

```

