
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

