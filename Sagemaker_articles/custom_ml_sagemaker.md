
# Running a Custom Algorithm in SageMaker

In this part, you will see how to run an algorithm of your own, instead of using SageMaker’s built-in algorithms. Although SageMaker provides built-in algorithms for almost any kind of problem statement, many times we want to run our own custom model utilizing the power of SageMaker. We can do so effectively if we have
working knowledge of Docker and hands-on knowledge of Python.

We will create a custom random forest model for our Big Mart dataset. We will deploy the container in ECR and then train the model using SageMaker. Then we will use the model to do real-time inference as well as understand how batch transformation is done.


## The Problem Statement

The problem statement is that we will try to predict the sales of an e-commerce firm using the random forest algorithm (one of the supervised learning ensemble tree algorithms). As it is a regression problem, we will be using the RandomForestRegressor class of the Scikit-Learn package.

We have already explored the dataset in second part: Data Processing in AWS Sagemaker (https://dev.to/aws-builders/data-processing-in-aws-sagemaker-20gi). 


![1](https://user-images.githubusercontent.com/23625821/122001307-6ce7c980-cdb0-11eb-9170-cee5b744fa69.png)

## Running the Model

Before moving to the application of the model inside the SageMaker environment, let’s first run the algorithm, locally, on the dataset that we have prepared and check the total loss that was incurred.

```py
from sklearn.ensemble import RandomForestRegressor

rfc = RandomForestRegressor(n_estimators=500)

```

In the previous code, we initialized the RandomForestRegressor algorithm and asked to merge the outputs of 500 individual decision trees. Once we have initialized the algorithm, we can start training the model.

```py

rfc.fit(X_train, y_train)

```

The previous code will start the training of the model. Now we can use the trained
model to make predictions on the test set.

```py
predictions = rfc.predict(X_test)
```

All the predictions are not stored in the variable predictions. Let’s calculate the roto
mean squared error of the model that we have created.

```py
from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(predictions, y_test))
```

## Transforming Code to Use SageMaker Resources

The following are the steps to run a custom model in SageMaker:

1. Store the data in S3.
2. Create a training script and name it train.
3. Create an inference script that will help in predictions. We will call it predictor.py.

4. Set up files so that it will help in endpoint generation.
5. Create a Dockerfile that will help in building an image inside which the entire code will run.

6. Build a script to push the Docker image to Amazon Elastic Container Registry (ECR).
7. Use the SageMaker and Boto3 APIs to train and test the model.


### Creating the Training Script

The first thing that should be kept in mind is that the script is going to run inside a container. So, there can be a synchronization issue as the script is inside while the data is coming from S3 bucket, which is outside the container. Also, the results of the algorithm should also be saved in the S3 bucket. We need to keep all this in mind as we create a training script.

The first thing that we should know is that inside the container, no matter what the data is that is coming in, it gets stored inside the folder /opt/ml. Therefore, data from S3 will be downloaded from that folder. So, in this folder we have to create three folders:

one to store the input, one to store the output, and one to store the models. This can be defined by using the following script:

```py

prefix = '/opt/ml/'
input_path = prefix + 'input/data'

output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')

```


