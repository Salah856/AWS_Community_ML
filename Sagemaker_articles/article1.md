
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

