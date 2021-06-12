
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

Our next step will be to store the data in S3. For our linear learner algorithm, we will use a data format called the RecordIO-Protobuf format. Using this data format helps you with a faster training time, and you can train models in live data mode (called pipe mode). We can convert our independent and target variables to RecordIO format using the following lines of code:


```py

import io
import numpy as np
import sagemaker.amazon.common as smac

vectors = np.array(X.values, dtype='float32')
labels = np.array(y_final, dtype='float32')

buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, vectors, labels)

buf.seek(0)

```

The previous lines convert the data into RecordIO format and then open the temporary file so that it can be directly inserted into S3. A RecordIO file is used by
breaking a big file into chunks and then using these chunks for analysis. This file helps us create streaming jobs in SageMaker, which makes the training fast. To send it, we will use the next lines of code:


```py

key = 'recordio-pb-data'

boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)

s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)

print('uploaded training data location: {}'.format(s3_train_data))

```


This will upload the data to S3 and close the buffer that we created. Now, our basic steps are done. All we need to do is to make the connection and train the model. The first step will be to initialize our linear learner algorithm Docker container.


```py

from sagemaker.amazon.amazon_estimator import get_image_uri

session = boto3.Session()

container = get_image_uri(session.region_name, 'linear-learner')

```

After initializing, let’s pass the required parameters for linear learner and initialize the algorithm.

```py

linear = sagemaker.estimator.Estimator(container,
                     role,
                     train_instance_count=1,
                     train_instance_type='ml.m4.xlarge',
                      output_path=output_location,
                      sagemaker_session=session)
                      

```

As we know, the regression algorithms have a few hyperparameters that need to be defined, such as the number of variables, batch size, etc. We will next define these values.


```py

linear.set_hyperparameters(feature_dim=11,
              predictor_type='multiclass_classifier',
              mini_batch_size=100,
              num_classes=4)
           
```


As everything is defined now, next we will start the training.

```py
linear.fit({'train': s3_train_data})
```


You will see the output given next at the start, and then the training will start.


![1](https://user-images.githubusercontent.com/23625821/121774888-d71c2680-cb84-11eb-8a96-a63247899018.png)


It will take some time for the model to be trained. Once the model is trained, we can deploy the model as an endpoint, and then we can start the testing. To deploy the model, we will use the deploy function.

```py

linear_predictor = linear.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

```


It will take some time to deploy the model and then create the endpoint. Once done, we can start the prediction. To start the prediction, we have to first tell what kind of data the endpoint will be receiving. Then we will have to serialize the data. This format helps to efficiently transfer and store the data, regaining the original data perfectly. We can serialize our test data by using the following code:

```py

from sagemaker.predictor import csv_serializer, json_deserializer

linear_predictor.content_type = 'text/csv'
linear_predictor.serializer = csv_serializer
linear_predictor.deserializer = json_deserializer

```


Now, whatever data we will be sending to the endpoint, it will be serialized and sent to the model. A prediction will come in a serialized manner, and then we will see the data in its original structure. To predict, we will be using the test data.

```py

result = linear_predictor.predict(test_vectors[0])
print(result)

```

The previous line gives us the prediction for a single row. But if we want predictions for multiple rows, we can use the following code:

```py
import numpy as np

predictions = []

for array in np.array_split(test_vectors, 100):
    result = linear_predictor.predict(array)
    predictions += [r['predicted_label'] for r in result['predictions']]

predictions = np.array(predictions)

```

