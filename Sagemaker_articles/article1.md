
# Amazon SageMaker

SageMaker makes the life of a data scientist much easier by providing services that you can use to prepare data, build models, test them, and then deploy them into production environments. It provides most of the common algorithms for building your machine learning models, and if you want to make any custom model not supported by SageMaker, then it has a facility to do so by using a bring-your-own container service. It also provides a distributed training option that can make your models run faster, as compared to a single-node run.


Amazon SageMaker comes with following features:

• SageMaker Studio
This is an application where you can build, train, validate, process,
and deploy the models. It’s a single place to do everything.

• SageMaker Ground Truth
This is used to create a labeled dataset.

• Studio Notebooks
This is one of the latest features of SageMaker that includes the
single sign-on feature, faster startup time, and one-click file
sharing.

• Preprocessing
This is used for analyzing and exploring data. It does feature
engineering and transformation of data, as well as all the other
things required to prepare the data for machine learning. 

• Auto-pilot
Without writing a single line of code, if you want SageMaker to
take care of your model building, either regression or classification
problems, auto-pilot is the feature to use. It is generally for users
who have less coding experience.

• Reinforcement Learning
This provides an interface to run a reinforcement learning
algorithm, which runs on a reward and penalty architecture.

• Batch Transform
After building the model, if you want to get predictions on a subset
of data or you want to preprocess a subset of data, you can use the
batch transform feature of SageMaker.

• Model Monitor
This is used to check whether the model quality is persistent or
deviates from the standard model.

• Debugger
This has different debugging usages, such as tracking the
hyperparameters whose values keep changing during the model
training. It can even alert if something abnormal happens with the
parameters or with the data.


![1](https://user-images.githubusercontent.com/23625821/121132433-69dd5e00-c831-11eb-9da8-461916c37f03.png)


## Understanding How SageMaker Works

These are the main steps that the process follows:

1. Fetch data
This is the first step for building any machine learning model.
Once we have decided on the problem statement that we have to solve, we have to accumulate all the data related to it. The data can be in the format of a database table, Excel sheets, text files, Word documents, images, etc. Once we know about all the data sources, these files need to be put inside a single repository so that the model knows about the location.

2. Clean the data
Our data can have null values, outliers, misspelled words, corrupted files, etc. All these things need to be explored and sorted out before the data is being given to the model. There are a lot of statistical methods as well that are used for data cleaning,


3. Prepare data
Once we have made our data clean, it is time to prepare our data.
This includes all the transformations done on the data, scaling and normalization processes, combination of features or splitting of features, etc. After all these things are done, it has to be stored at a specific place so that the model knows the reference to the clean and prepared data files. The first three steps that we have seen, all these things can be done inside the SageMaker Jupyter Notebook, and after that, the cleaned data can be stored inside an S3 bucket.

4. Train the model
Once the data is prepared, we need to train the model. The first
thing is to select the model that needs to be applied. The models
can be chosen from the list of built-in algorithms that SageMaker
provides, or custom models can also be used by making your own
containers and uploading them to AWS or buying them from the
AWS marketplace. Also, for training the model, we must decide on what kind of
computation is required. Selection can be made based on the
RAM size or number of GPU counts, etc. It is decided based on
how big the dataset is or how complex the model is.

5. Evaluate the trained model
Once the model is successfully trained on the dataset, it needs to
be evaluated before deploying it for production. For this, multiple
metrics can be used. For regression models, RMSE scores can be
used, while for classification models precision and recall can be
used. Once the metric crosses the decided threshold, only then
can it be moved toward production.


6. Deploy the model to production
It is easy to deploy the model in SageMaker. Generally, in normal
scenarios one has to make APIs and then serve the model through
an endpoint. For all this, coding requirements are necessary.
But, in SageMaker, with minimal coding efforts the model can be
converted into an API endpoint, and after that live or batch model
inference can be started. Also, to deploy the model, another
computational instance can be chosen, which generally takes less
RAM or GPUs as compared to the training model instance.


7. Monitor the model
Once the model starts serving in production, we can keep
monitoring the model’s performance. We can measure for which
data points the model is performing well, as well as the areas it is
not. This process is called knowing the ground truth.


8. Repeat the process when more data comes (retraining)
Finally, as and when new data comes, the model can be retrained,
and all the previous steps can be repeated. All this can be done
with zero downtime. This means that the old model keeps serving
until the new model is put into production.


## Model Training in SageMaker 

![2](https://user-images.githubusercontent.com/23625821/121134073-461b1780-c833-11eb-8403-12b97f1836af.png)

The figute above shows how exactly model training happens as well as how the model
deployment happens. In this section, we will talk about the training part, while in the
next section we will cover the deployment part. 

To understand how model training in SageMaker works, we will look at the bottom part of the image. We can see that there are five sections contributing to it.

• S3 bucket for training data

• Helper code

• Training code

• Training code image

• S3 bucket for model artifacts


Training a model in SageMaker is called a training job. Any algorithm that is executed in SageMaker requires the training data to be present in an S3 bucket. This
is because the compute instances that are used for training the model are called dynamically during model execution, and they are not persistent. This means the data that is stored there will be deleted once the job is done. Hence, we can save the data in S3, and the model will always know from where to fetch the data, by means of an S3 URL.


The coding part, which is written in Python, consists of two sections. The first section, the helper code, helps you in processing the data, fetching the data, storing the output, etc. The second section, the training code, actually does the model training for you by applying the selected algorithm on the data.


The training code image is a Docker container image that is stored in the ECR of AWS. It contains all the packages and software required for executing your code.

It also contains your training and deployment scripts that you write. We package everything required inside one container and push it to ECR. Then, we just pass the URL of the image to the algorithm selected, and automatically the training script runs. We need to understand that SageMaker works based on Docker containers, and hence it is imperative for users to understand Docker before learning SageMaker.

One thing to notice here is that the Docker image is built by you, but still we have not selected the hardware requirements. Therefore, when we call the SageMaker algorithm and when we pass the parameters such as the S3 URL and Docker Image URL, then only can we pass the type of instance that we have to choose. These instances are the EC2 instances. 

Once we have chosen the instance, the Docker image is downloaded on that instance, along with the training data. Finally, the model training starts.


## Model Deployment in SageMaker

Once the model training is done, all the learned parameters are stored in the S3 bucket and
called model artifacts.


The helper and inference code consists of processing scripts and prediction scripts.
Also, it includes the format in which the predictions need to be sent or saved. For the
predictions, the model artifacts generated during the training part are used.
SageMaker removes the training compute requirements with the deployment
compute requirements. This is because training may require big instances with stronger
computational power, but for predictions we do not require that many big instances.
Hence, the predictions can be done with smaller instances as well. This helps save a lot
of cost.

We can use the same Docker image that we built for training a model for the
inference by just adding a few extra Python scripts that help in deployment. That may
include using packages such as Flask, Gunicorn, etc. To start the deployment, we need
to pass the model artifacts the URL, the ECR image URL, and the compute instance that
we need. By giving these three parameters, the deployment is made, and an endpoint is
created.

The endpoint is a place where we send requests in a particular format, maybe CSV or JSON, and get the response from the model. This is called a RESTful API.

The model that is created is served through this API, and the data on which we want predictions is sent as a CSV, row by row, and we get the predictions in the same way. These are POST and GET requests. We can expose this endpoint to any client objects. It can be a website, a mobile app, an IOT device, or anything else. We just need some records sent to the endpoint and to get the predictions.


Endpoints are used when we make live predictions. Hence, they keep running until
and unless we manually stop them or add a timeout condition. But suppose we want the
predictions for a subset of data, maybe 5,000 rows, and we don’t want a live endpoint.
Then SageMaker supports something called a batch transform. Using this approach,
we provide the same parameters that we provided to deployment code, but one extra
parameter is provided. It is the link to the data on which inference is needed. This data
is again stored in S3 and hence downloaded to the instance when prediction is required.
After the prediction is done, predictions are stored in S3, and then the compute instance
is stopped immediately


![1](https://user-images.githubusercontent.com/23625821/121296641-4c6fc900-c8f1-11eb-9921-f23ac2117319.png)



