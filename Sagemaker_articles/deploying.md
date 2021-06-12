
# Building and Deploying Models in SageMaker

In this part, we will be exploring some of SageMaker’s built-in algorithms that are widely used in the industry. We will be exploring the algorithms from the general domain, natural language processing domain, computer vision domain, and forecasting domain.


## SageMaker Application of Linear Learner

The first step will be to read the dataset from the S3 bucket, preprocess the columns to remove the null values, and apply scaling and encoding. We saw how to preprocess the dataset and get to the dependent and independent variables in the previous part. Therefore, we will start this section directly by applying the algorithm on the preprocessed dataset. We will define the role and buckets so that SageMaker can talk to different services properly. 


```py

import boto3
from sagemaker import get_execution_role

bucket = 'slytherins-test'
prefix = 'linear-learner'
role = get_execution_role()


```


Now, we need to decide what algorithm needs to be applied, that is, linear or logistic regression. We will start with logistic regression. To make a logistic regression model, we need a categorical column. We know that our target variable is Sales, and it is a numerical column; hence, logistic regression cannot be applied. So, we will bin the Sales columns into four categories, and then we can start applying algorithms.


```py

y_binned = pd.cut(y['Item_Outlet_Sales'], 4, labels=['A', 'B', 'C', 'D'])


```

Now that we have our categorical column as a target variable, we will apply label encoding on it so that each category can be represented by an integer.


```py

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
temp = le.fit(y_binned)
y_final = temp.transform(y_binned)

```




