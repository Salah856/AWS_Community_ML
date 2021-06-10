
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


