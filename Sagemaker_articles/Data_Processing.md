
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

