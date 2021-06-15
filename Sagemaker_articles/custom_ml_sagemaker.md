
# Running a Custom Algorithm in SageMaker

In this part, you will see how to run an algorithm of your own, instead of using SageMaker’s built-in algorithms. Although SageMaker provides built-in algorithms for almost any kind of problem statement, many times we want to run our own custom model utilizing the power of SageMaker. We can do so effectively if we have
working knowledge of Docker and hands-on knowledge of Python.

We will create a custom random forest model for our Big Mart dataset. We will deploy the container in ECR and then train the model using SageMaker. Then we will use the model to do real-time inference as well as understand how batch transformation is done.


## The Problem Statement

The problem statement is that we will try to predict the sales of an e-commerce firm using the random forest algorithm (one of the supervised learning ensemble tree algorithms). As it is a regression problem, we will be using the RandomForestRegressor class of the Scikit-Learn package.

We have already explored the dataset in second part: Data Processing in AWS Sagemaker (https://dev.to/aws-builders/data-processing-in-aws-sagemaker-20gi). 


![1](https://user-images.githubusercontent.com/23625821/122001307-6ce7c980-cdb0-11eb-9170-cee5b744fa69.png)

