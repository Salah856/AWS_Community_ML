
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


So, there are two columns with null values: ```Item_Weight``` and ```Outlet_Size```. We can use the normal imputation methods provided by Scikit-Learn to impute these null values. But, instead, we will be using the help of nearby columns to fill in these null values. Let’s look at the data types of these columns, as that is going to help us in making imputation strategies.

```py

print(data['Item_Weight'].dtype)

print(data['Outlet_Size'].dtype)

```

The output shows that the ```Item_Weight``` column is a float, while the ```Outlet_Size``` column is categorical (or an object). 


What we will do next is to first split the data into numerical and categorical data frames and then impute the null values.

```py

 import numpy as np
 
 cat_data = data.select_dtypes(object)
 num_data = data.select_dtypes(np.number)

```

Now we have all the categorical columns in cat_data. We can check for the presence of null values again. 

```py
  cat_data.isna().sum()
```

![1](https://user-images.githubusercontent.com/23625821/121466633-07fb3080-c9b8-11eb-931a-6cca9dfd9eef.png)


So, the null value still exists. If we look at the categories present in the ```Outlet_Size``` columns, we will see there are three. 

```py 
cat_data.Outlet_Size.value_counts()

```

![1](https://user-images.githubusercontent.com/23625821/121466801-527cad00-c9b8-11eb-8a54-845a5864db45.png)

We will do anohter thing before moving on to dummification. If we look at the categories of the ```Item Fat Content``` column. 

```py
cat_data.Item_Fat_Content.value_counts()

```

![1](https://user-images.githubusercontent.com/23625821/121467247-0da54600-c9b9-11eb-9dbb-bc2e4ef3fe24.png)


LF means Low Fat, reg means Regular, and low fat is just the lowercase version of Low Fat. Let’s rectify all of this.


```py
cat_data.loc[cat_data['Item_Fat_Content'] == 'LF' , ['Item_Fat_Content']] = 'Low Fat'

cat_data.loc[cat_data['Item_Fat_Content'] == 'reg' , ['Item_Fat_Content']] = 'Regular'

cat_data.loc[cat_data['Item_Fat_Content'] == 'low fat' , ['Item_Fat_ Content']] = 'Low Fat'
```

![1](https://user-images.githubusercontent.com/23625821/121467391-3e857b00-c9b9-11eb-8382-f912322bd6b8.png)


Next, let’s apply label encoding on the categorical data frame. We will use the ```Scikit-Learn``` package for this.


```py

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
cat_data = cat_data.apply(le.fit_transform)

```

![1](https://user-images.githubusercontent.com/23625821/121467555-8b695180-c9b9-11eb-8f22-0f5ce192094c.png)


```py
cat_data.head()
```

![1](https://user-images.githubusercontent.com/23625821/121467608-a6d45c80-c9b9-11eb-9d6f-14541b94f9b2.png)


We will concatenate the two data frames, categorical and numerical, and then normalize the columns. Also, we will remove two of the columns before that, one in
Item_Identifier and the second in Item_Sales. Item_Identifier is not really an important column, while Item_Sales will be our dependent variable; hence, it cannot be in the independent variables list.

```py

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

num_data = pd.DataFrame(ss.fit_transform(num_data.drop(['Item_Outlet_Sales'], axis=1)), columns = num_data.drop(['Item_Outlet_Sales'],axis=1).columns)

cat_data = pd.DataFrame(ss.fit_transform(cat_data.drop(['Item_Identifier'], axis=1)), columns = cat_data.drop(['Item_Identifier'], axis=1).columns)

final_data = pd.concat([num_data,cat_data],axis=1)

final_data.head()


```


![1](https://user-images.githubusercontent.com/23625821/121467985-60cbc880-c9ba-11eb-9092-3be2649b81ad.png)


Now, we have our final data ready. We have used a standard scaler class to normalize all the numerical values to their z-scores. We will be using final_data as independent variables, while we will extract Item Sales as dependent variables.

```py
X = final_data
y = data['Item_Outlet_Sales']

```

The last step is to get our training and validation sets. For this we will use the class model_selection provided by Scikit-Learn. We will take 10 percent of our data as a validation set while remaining as a test set.

```py

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=5)

```

This marks the last step of data processing. Now we can use it to train any kind of model that we want. The code lines that I have shown can be executed in any Jupyter Notebook, either in the localhost or in the cloud. The only requirement is that the necessary packages must be installed.


In the next section, I will show you how to run the same code in SageMaker using the Scikit-Learn container provided by the SageMaker service. The script remains the same, but the process changes, as we have to continuously talk with the S3 bucket and define the instances as well.



## Preprocessing Using SageMaker’s Scikit-Learn Container

We use SageMaker to take advantage of multiple things, especially the computation power, API generation, and ease of storage. Therefore, to achieve these things, the code must be written in a specific format. We will use the same code that we saw in the previous section, but we’ll make some changes in the overall structure so that it becomes compatible with SageMaker.


First, the data should be in the S3 bucket. We have already put our Train.csv file in the bucket, in the first section of this chapter. Once that is done, we can start writing our code. First, we will define the role of the user and the region in which we are using the SageMaker service.


```py

import boto3
import sagemaker
from sagemaker import get_execution_role

region = boto3.session.Session().region_name
role = get_execution_role()

```

The Boot3 package tries to extract the region name automatically if we are using the SageMaker notebook. If we are working from the localhost notebook, then it needs to be custom defined. We will look at that part in the last part of this book. ```get_execution_role()``` extracts the current role with which the user has signed in. It can be the root user or IAM role.


Now that we have defined the region and role, the next step will be to define our Scikit-Learn container. As mentioned in the first part of the book, SageMaker operates on Docker containers. All the built-in algorithms are nothing but Docker containers, and even the custom algorithm must be put inside the Docker container and uploaded to ECR. Since we will be using Scikit-Learn to process our data, already SageMaker has a processing container for that. We just need to instantiate it and then use it.


```py

from sagemaker.sklearn.processing import SKLearnProcessor

sklearn_processor = SKLearnProcessor(framework_version='0.20.0',
                    role=role,
                    instance_type='ml.m5.xlarge',
                    instance_count=1)

```


In the previous code, we created an object called SKLearnProcessor. The parameters passed tell about the version of Scikit-Learn to use, the IAM role to be passed to the instance, the type of compute instance to be used, and finally the number of compute instances to be spinned up. Once this is done, any Python script that we write and that uses Scikit-Learn can be used inside this container.

Now, let’s check whether our data is accessible from SageMaker.

```py
import pandas as pd

input_data = 's3://slytherins-test/Train.csv'
df = pd.read_csv(input_data)
df.head()
```

```slytherins-test``` is the name of the S3 bucket that we created earlier in the article. ```Train.csv``` is the data that we uploaded. If everything works perfectly, you’ll get the output like this: 

![1](https://user-images.githubusercontent.com/23625821/121469046-16e3e200-c9bc-11eb-8f52-af7807f5ddd6.png)


Now, it’s time to define our processing script that will be run inside the container. We have already written this script in the previous part. We will just restructure the code and save it inside a file named ```preprocessing.py```.


```py

import argparse
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Here we have defined all the columns that are present in our data

columns = ['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility','Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Outlet_Sales']

# This method will help us in printing the shape of our data

def print_shape(df):
    print('Data shape: {}'.format(df.shape))

if __name__=='__main__':
    # At the time of container execution we will use this parser to define our train validation split. Default kept is 10%
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-test-split-ratio', type=float, default=0.1)
    
    args, _ = parser.parse_known_args()
    print('Received arguments {}'.format(args))
    
    # This is the data path inside the container where the Train.csv will be downloaded and saved
    
    input_data_path = os.path.join('/opt/ml/processing/input', 'Train.csv')
    print('Reading input data from {}'.format(input_data_path))
    
    data = pd.read_csv(input_data_path)
    data = pd.DataFrame(data=data, columns=columns)
    
    for i in data.Item_Type.value_counts().index:
        data.loc[(data['Item_Weight'].isna()) & (data['Item_Type'] == i), ['Item_Weight']] = \
        data.loc[data['Item_Type'] == 'Fruits and Vegetables',['Item_Weight']].mean()[0]
        
    cat_data = data.select_dtypes(object)
    num_data = data.select_dtypes(np.number)
    
    cat_data.loc[(cat_data['Outlet_Size'].isna()) & (cat_data['Outlet_Type'] == 'Grocery Store'), ['Outlet_Size']] = 'Small'
    cat_data.loc[(cat_data['Outlet_Size'].isna()) & (cat_data['Outlet_Type'] == 'Supermarket Type1'), ['Outlet_Size']] = 'Small'
    cat_data.loc[(cat_data['Outlet_Size'].isna()) & (cat_data['Outlet_Type'] == 'Supermarket Type2'), ['Outlet_Size']] = 'Medium'
    cat_data.loc[(cat_data['Outlet_Size'].isna()) & (cat_data['Outlet_Type'] == 'Supermarket Type3'), ['Outlet_Size']] = 'Medium'
    
    cat_data.loc[cat_data['Item_Fat_Content'] == 'LF' , ['Item_Fat_ Content']] = 'Low Fat'
    cat_data.loc[cat_data['Item_Fat_Content'] == 'reg' , ['Item_Fat_Content']] = 'Regular'
    cat_data.loc[cat_data['Item_Fat_Content'] == 'low fat' , ['Item_Fat_Content']] = 'Low Fat'
    
    le = LabelEncoder()
    
    cat_data = cat_data.apply(le.fit_transform)
    ss = StandardScaler()
    
    num_data = pd.DataFrame(ss.fit_transform(num_data), columns = num_data.columns)
    cat_data = pd.DataFrame(ss.fit_transform(cat_data), columns = cat_data.columns)
    
    final_data = pd.concat([num_data,cat_data],axis=1)
    
    print('Data after cleaning: {}'.format(final_data.shape))
    
    X = final_data.drop(['Item_Outlet_Sales'], axis=1)
    y = data['Item_Outlet_Sales']
    
    split_ratio = args.train_test_split_ratio
    
    print('Splitting data into train and test sets with ratio {}'.format(split_ratio))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=0)
    
    # This defines the output path inside the container from where all the csv sheets will be taken and uploaded to S3 Bucket
    
    train_features_output_path = os.path.join('/opt/ml/processing/train', 'train_features.csv')
    train_labels_output_path = os.path.join('/opt/ml/processing/train', 'train_labels.csv')
    
    test_features_output_path = os.path.join('/opt/ml/processing/test', 'test_features.csv')
    test_labels_output_path = os.path.join('/opt/ml/processing/test', 'test_labels.csv')
    
    print('Saving training features to {}'.format(train_features_output_path))
    
    pd.DataFrame(X_train).to_csv(train_features_output_path, header=False, index=False)
    
    print('Saving test features to {}'.format(test_features_output_path))
    pd.DataFrame(X_test).to_csv(test_features_output_path, header=False, index=False)
    
    print('Saving training labels to {}'.format(train_labels_output_path))
    y_train.to_csv(train_labels_output_path, header=False, index=False)
    
    print('Saving test labels to {}'.format(test_labels_output_path))
    y_test.to_csv(test_labels_output_path, header=False, index=False)

```

As we can see, the previous code is the same, all we have done is defined the place where the data will be stored inside the container and the place where the output will be stored and then uploaded to the S3 bucket from there. Once this script is defined, we are good to go now. All we have to do is spin up the instantiated container, pass this script as a parameter, pass the data as a parameter, pass the directory where output files will be stored, and finally pass the
destination S3 bucket.


```py

from sagemaker.processing import ProcessingInput, ProcessingOutput

sklearn_processor.run(code='preprocessing.py',
           inputs=[ProcessingInput(
             source=input_data,
             destination='/opt/ml/processing/input')],
           outputs=[ProcessingOutput(output_name='train_data',
                     source='/opt/ml/processing/train',
                     destination='s3://slytherins-test/'
                     ),
                    ProcessingOutput(output_name='test_data',
                     source='/opt/ml/processing/test',
                     destination='s3://slytherins-test/'
                     )
                   ],
           arguments=['--train-test-split-ratio', '0.1']
         )


```


In the previous code, we have passed all the parameters. Also, we have defined the argument that tells about the split percentage. Inside the preprocessing.py script, we have code that parses this argument.



The processing job will take some time to finish. It first launches an instance (which is similar to booting up an operating system), and then it downloads the sklearn image on the instance. Then data is downloaded to the instance. Then the processing job starts. When the job finishes, the training and test data is stored back to S3. Then the entire operation finishes. Once the job is finished, we can get detailed information about the job by using the following script:

```py

preprocessing_job_description = sklearn_processor.jobs[-1].describe()

```

Let’s use this script to get the S3 bucket location of the training and test datasets:

```py
output_config = preprocessing_job_description['ProcessingOutputConfig']

for output in output_config['Outputs']:
    if output['OutputName'] == 'train_data':
        preprocessed_training_data = output['S3Output']['S3Uri']
    if output['OutputName'] == 'test_data':
        preprocessed_test_data = output['S3Output']['S3Uri']

```

Now, we can check the output by reading the data using Pandas.

```py

training_features = pd.read_csv(preprocessed_training_data + 'train_features.csv', nrows=10, header=None)

print('Training features shape: {}'.format(training_features.shape))

training_features.head(10)

```

![1](https://user-images.githubusercontent.com/23625821/121471058-5e1fa200-c9bf-11eb-852d-b20a398112ce.png)



##  Creating Your Own Preprocessing Code Using ScriptProcessor

In the previous section, we used SkLearnProcessor, which is a built-in container provided by SageMaker. But, many times, we have to write some code that cannot only be executed in a SageMaker’s predefined containers.

For that we have to make our own containers. We will be looking at making our own containers while training a machine learning model as well. In this section, we will make a container that performs the same tasks as the SKlearnProcessor container. The only difference is that it’s not prebuilt; we will build it from scratch.

To use custom containers for processing jobs, we use a class provided by SageMaker named ScriptProcessor. Before giving inputs to ScriptProcessor, the first task is to create our Docker container and push it to ECR.


## Creating a Docker Container

For this we will be creating a file named Dockerfile with no extension. Inside this we will be downloading an image of a minimal operating system and then install our packages inside it. So, our minimal operating system will be Linux based, and we will have Python, Scikit-Learn, and Pandas installed inside it.


```dockerfile

FROM python:3.7-slim-buster
RUN pip3 install pandas==0.25.3 scikit-learn==0.21.3
ENV PYTHONUNBUFFERED=TRUE
ENTRYPOINT ["python3"]

```


The previous script must be present inside the Dockerfile.

The first line, ```FROM python:3.7-slim-buster```, tells about the minimal operating system that needs to be downloaded from Docker Hub. This only contains Python 3.7 and the minimal packages required to run Python. But, we need to install other packages as well.

That’s why we will use the next line, ```RUN pip3 install pandas==0.25.3 scikit-learn==0.21.3```. This will install Pandas, Scikit-Learn, Numpy, and other important packages.

The next line, ```ENV PYTHONUNBUFFERED=TRUE```, is an advanced instruction that tells Python to log messages immediately. This helps in debugging purposes.

Finally, the last line, ```ENTRYPOINT ["python3"]```, tells about how our preprocessing.py file should execute.


## Building and Pushing the Image

Now that our Docker file is ready, we need to build this image and then push it to Amazon ECR, which is a Docker image repository service. To build and push this image, the following information will be required:

1. Account ID
2. Repository name
3. Region
4. Tag given to the image


All this information can be initialized using the following script:

```py

import boto3

account_id = boto3.client('sts').get_caller_identity().get('Account')

ecr_repository = 'sagemaker-processing-container'

tag = ':latest'

region = boto3.session.Session().region_name

```


Once we have this information, we can start the process by first defining the ECR repository address and then executing some command-line scripts.

```py
processing_repository_uri = '{}.dkr.ecr.{}.amazonaws.com/{}'.format(account_id, region, ecr_repository + tag)
```
```sh

# Create ECR repository and push docker image

! docker build -t $ecr_repository docker # This builds the image

! $(aws ecr get-login --region $region --registry-ids $account_id --no-include-email) # Logs in to AWS
! aws ecr create-repository --repository-name $ecr_repository # Creates ECR Repository

! docker tag {ecr_repository + tag} $processing_repository_uri # Tags the image to differentiate it from other images
! docker push $processing_repository_uri # Pushes image to ECR


```

If everything works fine, then your image will successfully be pushed to ECR. You can go to the ECR service and check the repository.


![1](https://user-images.githubusercontent.com/23625821/121472567-a63fc400-c9c1-11eb-985e-2248eaa2d3bd.png)


## Using a ScriptProcessor Class

Now that our image is ready, we can start using the ```ScriptProcessor class```. We will execute the same code, ```preprocessing.py```, inside this container. Just like how we did in ```SKLearnProcessor```, we will create an object of the class first.

```py

from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker import get_execution_role

role = get_execution_role()

script_processor = ScriptProcessor(command=['python3'],
          image_uri=processing_repository_uri,
          role=role,
          instance_count=1,
          instance_type='ml.m5.xlarge')

input_data = 's3://slytherins-test/Train.csv'

script_processor.run(code='preprocessing.py',
          inputs=[ProcessingInput(
            source=input_data,
            destination='/opt/ml/processing/input')],
          outputs=[ProcessingOutput(source='/opt/ml/processing/train', destination='s3://slytherins-test/'),
            ProcessingOutput(source='/opt/ml/processing/test', destination='s3://slytherins-test/')]
            
         )



```

You will find the code to be almost the same as the ```SKLearnProcessor``` code. It will give the same output as well. Finally, once the processing job is done, we can check the output again in the same way.


```py

preprocessing_job_description = script_processor.jobs[-1].describe()

output_config = preprocessing_job_description['ProcessingOutputConfig']

for output in output_config['Outputs']:
    if output['OutputName'] == 'output-1':
        preprocessed_training_data = output['S3Output']['S3Uri']
    if output['OutputName'] == 'output-2':
        preprocessed_test_data = output['S3Output']['S3Uri']
        


```


```py

import pandas as pd

training_features = pd.read_csv(preprocessed_training_data + 'train_features.csv', nrows=10, header=None)

print('Training features shape: {}'.format(training_features.shape)) 

training_features.head(n=10)


```

## Using Boto3 to Run Processing Jobs

As mentioned, we use the Boto3 package to access the services of AWS from any other computer, including your localhost. So, in this section, we will be running the custom Docker container script that we saw in the previous section, using Boto3.


### Installing Boto3

The first step for using Boto3 is to install it inside the localhost environment. Along with Boto3, we have to install awscli, which will help us in authentication with AWS and s3fs, which in turn will help us in talking with the S3 bucket. To install it, we will be using pip, as shown here:

```sh

pip install boto3
pip install awscli
pip install s3fs

```

Once the installation finishes, we need to configure the credentials of AWS. For this, we will run the following command:

```sh
aws configure
```
This will ask you for the following four inputs:

• AWS access key
• AWS secret access key
• Default region name
• Default output format


Once we provide this information, we can easily use Boto3 to connect with the AWS services. I have already shown you how to get the access key and secret access key when creating the IAM roles. The default region name will be us-east-2, but you can recheck this by looking at the top-right corner of your AWS management console. It will tell you the location.


![1](https://user-images.githubusercontent.com/23625821/121767707-4a5c7300-cb5a-11eb-822a-436af1a88ee2.png)


Once this part is done, we can start our Jupyter Notebook (local system notebook) and create a notebook using the same environment inside which we have installed all the packages and configured AWS.


### Initializing Boto3
Inside the notebook, the first step will be to initialize Boto3. For this we will use the following script:

```py 
import boto3
import s3fs 

region = boto3.session.Session().region_name
client = boto3.client('sagemaker')

```

## Making Dockerfile Changes and Pushing the Image

Now, we will use the Boto3 API to call the processing job method. This will create the same processing job that we saw in the previous section. But, minor changes will be required, and we will explore them one by one.

We will use the method create_processing_job to run the data processing job. To learn more about this method, or all the methods related to SageMaker provided by
Boto3, you can visit https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html . 


But, before that, we have to make some changes in our Docker container and our processing Python file. For the Docker container, we will need to copy our
preprocessing.py script inside it so that the Boto3 method can run the script directly. For this we will make the following changes to our Dockerfile:

```sh 

FROM python:3.7-slim-buster
RUN pip3 install pandas==0.25.3 scikit-learn==0.21.3
ENV PYTHONUNBUFFERED=TRUE
ENV PATH="/opt/ml/code:${PATH}"
COPY preprocessing.py /opt/ml/code/preprocessing.py
WORKDIR /opt/ml/code


```

We have added three new lines to our existing Dockerfile. The line ENV PATH="/opt/ml/code:${PATH}" sets up the environment path to /opt/ml/code. We will be placing our script, preprocessing.py, inside it with COPY preprocessing.py /opt/ml/code/preprocessing.py. Finally, we will be making our working directory the same folder: WORKDIR /opt/ml/code. This is required so that the Docker container will know where the script file is present, and it will help in its execution.


Once we have made changes in the Dockerfile, we will make changes to the script that builds the image and pushes it to the ECR. The only change that we need to do is add a line that gives the permission to the container to play with the preprocessing.py script. Otherwise, Docker may not have the permission to open and look at its contents.

```sh 

# Create ECR repository and push docker image

! chmod +x docker/preprocessing.py # This line gives read and write access to the preprocessing script
! docker build -t $ecr_repository docker # This builds the image

! $(aws ecr get-login --region $region --registry-ids $account_id --no-include-email) # Logs in to AWS
! aws ecr create-repository --repository-name $ecr_repository # Creates ECR Repository

! docker tag {ecr_repository + tag} $processing_repository_uri # Tags the image to differentiate it from other images
! docker push $processing_repository_uri # Pushes image to ECR


```

Once this step is done, we will be ready to run our Boto3 processing job.

## Creating a Processing Job

In a nutshell, we need information about four sections to create a processing job using Boto3.

• Input data information (ProcessingInput)
• Output data information (ProcessingOutput)
• Resource information (ProcessingResources)
• Container information (AppSpecification)


As you can see in the following code, all the previous information is provided. The code is again similar to the code we saw in the previous section; it is just that Boto3 needs information that should be manually put inside it as parameters, while when we run the code from inside SageMaker, most of the information is automatically extracted.


```py

response = client.create_processing_job(       # Initialize the method
    ProcessingInputs=[
        {
            'InputName': "Training_Input",    # Give Input Job a name
            'S3Input': {
                'S3Uri': input_data,          # URL from where the data needs to be taken
                'LocalPath': '/opt/ml/processing/input',  # Local directory where the data will be downloaded
                'S3DataType': 'S3Prefix',  # What kind of Data is it?
                'S3InputMode': 'File'    # Is it a file or a continuous stream of data?
            }
        },
    ],
    ProcessingOutputConfig={
        'Outputs': [
            {
                'OutputName': 'Training',  # Giving Output Name
                'S3Output': {
                    'S3Uri': 's3://slytherins-test/', # Where the output needs to be stored
                    'LocalPath': '/opt/ml/processing/train',  # Local directory where output needs to be searched
                    'S3UploadMode': 'EndOfJob'  # Upload is done when the job finishes
                },
                'OutputName': 'Testing',
                'S3Output': {
                    'S3Uri': 's3://slytherins-test/',
                    'LocalPath': '/opt/ml/processing/test',
                      'S3UploadMode': 'EndOfJob'
                }
            },
        ],
    },
    ProcessingJobName='preprocessing-job-test',  # Giving a name to the entire job. It should be unique
    ProcessingResources={
        'ClusterConfig': {
            'InstanceCount': 1,  # How many instances are required?
            'InstanceType': 'ml.m5.xlarge',   # What's the instance type?
            'VolumeSizeInGB': 5   # What should be the instance size?
        }
    },
    AppSp={
        'ImageUri': '809912564797.dkr.ecr.us-east-2.amazonaws.com/sagemaker-processing-container:latest',
             # Docker Image URL
        'ContainerEntrypoint': [
            'Python3','preprocessing.py'  # How to run the script
        ]
    },
    RoleArn='arn:aws:iam::809912564797:role/sagemaker-full-accss', # IAM role definition
)

```

The previous code will start the processing job. But, you will not see any output. To know the status of the job, you can use CloudWatch, which I will talk about in the next section. For now, we will get help from the Boto3 method describe_processing_job to get the information. We can do this by writing the following code:

```py

client.describe_processing_job(ProcessingJobName='processing-job-test')

```

This will give us detailed information about the job: 

![1](https://user-images.githubusercontent.com/23625821/121768181-28182480-cb5d-11eb-8599-4a21d3ab7138.png)



You will find the key ProcessingJobStatus, which tells about the status, and if the job fails, you will get a reason for the failure key as well. So, now we have seen the three ways of data processing provided by SageMaker.


