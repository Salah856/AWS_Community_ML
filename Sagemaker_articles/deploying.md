
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
