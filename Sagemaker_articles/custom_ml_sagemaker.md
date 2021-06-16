
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


Inside the data folder, we can have multiple files such as training, validation, or testing. We can also have separate files contributing to a single training file. Hence, we can make this kind of segregation as well. For us, we have only one file: the training file. So, we will be using only one channel.

```py
channel_name='training'
training_path = os.path.join(input_path, channel_name)

```

This prepares our training script to handle data. Next is the training script itself. The data will come from S3. First we have to read it. 

```py

input_files = [ os.path.join(training_path, file) for file in os.listdir(training_path) ]

raw_data = [ pd.read_csv(file) for file in input_files ]

data = pd.concat(raw_data)

```

This script also helps if you have multiple CSV sheets to read. But, in that case remember to keep the parameter header=None. Now that we have read the data, we can start the training process. The following is the entire script for the training:


```py 

def train():
    print('Starting the training.')
    try:
        # Take the set of files and read them all into a single pandas dataframe
        input_files = [ os.path.join(training_path, file) for file in os.listdir(training_path) ]
        
        if len(input_files) == 0:
            raise ValueError(('There are no files in {}.\n' +
                 'This usually indicates that the channel ({}) wasincorrectly specified,\n' +
                 'the data specification in S3 was incorrectly specified orthe role specified\n' +
                 'does not have permission to access the data.').format(training_path, channel_name))
                 
        raw_data = [ pd.read_csv(file) for file in input_files ]
        data = pd.concat(raw_data)
        data = data.sample(frac=1)
        
        for i in data.Item_Type.value_counts().index:
           data.loc[(data['Item_Weight'].isna()) & (data['Item_Type'] == i), ['Item_Weight']] = \
           data.loc[data['Item_Type'] == 'Fruits and Vegetables', ['Item_Weight']].mean()[0]
        
        cat_data = data.select_dtypes(object)
        num_data = data.select_dtypes(np.number)
        cat_data.loc[(cat_data['Outlet_Size'].isna()) & (cat_data['Outlet_Type'] == 'Grocery Store'), ['Outlet_Size']] = 'Small'
        
        cat_data.loc[(cat_data['Outlet_Size'].isna()) & (cat_data['Outlet_Type'] == 'Supermarket Type1'), ['Outlet_Size']] = 'Small'
        cat_data.loc[(cat_data['Outlet_Size'].isna()) & (cat_data['Outlet_Type'] == 'Supermarket Type2'), ['Outlet_Size']] = 'Medium'
        cat_data.loc[(cat_data['Outlet_Size'].isna()) & (cat_data['Outlet_Type'] == 'Supermarket Type3'), ['Outlet_Size']] = 'Medium'
        
        cat_data.loc[cat_data['Item_Fat_Content'] == 'LF' , ['Item_Fat_Content']] = 'Low Fat'
        cat_data.loc[cat_data['Item_Fat_Content'] == 'reg' , ['Item_Fat_Content']] = 'Regular'
        cat_data.loc[cat_data['Item_Fat_Content'] == 'low fat' , ['Item_Fat_Content']] = 'Low Fat'
        
        le = LabelEncoder()
        cat_data = cat_data.apply(le.fit_transform)
        
        ss = StandardScaler()
        num_data = pd.DataFrame(ss.fit_transform(num_data.drop(['Item_Outlet_Sales'], axis=1)), columns = ­num_data.drop(['Item_Outlet_Sales'],axis=1).columns)

        cat_data = pd.DataFrame(ss.fit_transform(cat_data.drop(['Item_Identifier'], axis=1)), columns = cat_data.drop(['Item_Identifier'], axis=1).columns)
        
        final_data = pd.concat([num_data,cat_data],axis=1)
        
        X = final_data
        y = data['Item_Outlet_Sales']
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=5)
        
        from sklearn.ensemble import RandomForestRegressor
        
        rfc = RandomForestRegressor(n_estimators=500)
        clf = rfc.fit(X_train, y_train)
        
        # save the model
        with open(os.path.join(model_path, 'randomForest-tree-model.pkl'), 'w') as out:
            pickle.dump(clf, out)
        print('Training complete.')
    
    except Exception as e:
    
        trc = traceback.format_exc()
        
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        
        sys.exit(255)
        

```

We will keep the entire script inside a function called train(). After reading the CSV sheet, we will follow the same procedure we saw in Chapter 5. Later we will fit the random forest model on the data, which we ran in the previous section.

After all this, we have to save this model because later we will have to make predictions using the model. To save the model, we will first serialize it using pickle and then save it in the model location. Later, this model will be saved in S3. Finally, we can run the entire script.


```py

if __name__ == '__main__':
    train()
    sys.exit(0)
```

We have to use sys.exit(0) as it sends the message to SageMaker that the training has successfully completed. Save the file with the name train and no extension.

### Creating the Inference Script

The training script is used to train the model. But, once the model is trained, we need to make predictions, whether with real-time inference. We will save the inference script in a file named predictor.py.

The predictor file consists of the following components:

- ScoringService() class
- ping() method

- transformation() method
- Any other helper function required

The ScoringService() class consists of two functions. The first function, get_model(), loads and deserializes the model, while the second method, predict(), is responsible for making the predictions.


```py

class ScoringService(object):

    model = None
    
    @classmethod
    def get_model(cls):
      if cls.model == None:
          with open(os.path.join(model_path, 'randomForest-tree-model. pkl'), 'r') as inp:
             cls.model = pickle.load(inp)
        return cls.model
        
    @classmethod
    def predict(cls, input):
        clf = cls.get_model()
        return clf.predict(input)
        

```

The ping() method is just used to check whether the Docker container that the code is running in is healthy. If it’s not healthy, then it gives a 404 error, else 202.

```py

@app.route('/ping', methods=['GET'])
def ping():
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')
    

```

transformation() is the method that is responsible for reading the test file and calling the required methods and classes. One thing to understand here is that this entire endpoint generation process is nothing but the creation of an API. Once the API is created, the data is sent as a POST request, and then we get the predictions as a response. This entire architecture is built using the Flask framework.


The data is sent using the POST method, so to read it, we need the StringIO() method to decode the data. Once the data is decoded, we can read it with our normal
Pandas method. The transformation() function sends the data to the predict() function of class ScoringService(). The method sends the output back to the transformation() function.


This prediction output is sent back to the host from where the API is called, with help from the StringIO() function. This finishes the entire cycle of endpoints. The following is the code of transformation():

```py

@app.route('/invocations', methods=['POST'])
def transformation():
    data = None
    
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        s = StringIO.StringIO(data)
        data = pd.read_csv(s, header=None)
    
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')
    
    # Do the prediction
    predictions = ScoringService.predict(data)
    
    # Convert from numpy back to CSV
    out = StringIO.StringIO()
    
    pd.DataFrame({'results':predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()
    
    return flask.Response(response=result, status=200, mimetype='text/csv')


```


## Configuring the Endpoint Generation Files

To run the inference server successfully, we need to configure the following files:

- nginx.conf file
- serve file
- wsgi.py file


The Nginx file is used to spin up the server and make the connection between the Docker containers deployed on EC2 instances and the client outside or inside the
SageMaker network possible. Nginx uses a Python framework called Gunicorn that helps to set up the HTTP server.


Serve uses the running Gunicorn server to make the connection between the different resources feasible. Specifically, it is used for the following purposes:

- Efficiently using the number of CPUs for running the model
- Defining the server timeout
- Generating logs

- Starting the server using Nginx and Gunicorn
- Stopping the server if something doesn’t go as expected


Lastly, the wsgi.py file is used to tell the server about our predictor.py file.


## Setting Up the Dockerfile

Now that all our script files are ready, we have to create a Docker image so that it can be uploaded to ECR and then SageMaker can access the code present in it and run it in an EC2 instance attached.

![1](https://user-images.githubusercontent.com/23625821/122170162-8b1bfb00-ce7e-11eb-90c6-d4f674247f7b.png)


Now, we have to create a Dockerfile script, which will be run to build the image. Then we will use the build_and_push.sh file to push the image to ECR.

These are the steps that we will follow in the Dockerfile:

1. Download an image from DockerHub that will have our operating system. We will download a minimal version of Ubuntu so that our code can run inside it. For this, we will use the following script:
 
```sh 
FROM ubuntu:16.04 
```

2. Name the person, or the organization, who is maintaining and creating this image. 

```sh
MAINTAINER <<your name >> 
```

3. Run some Ubuntu commands so that we can set up the Python environment and update the operating system files. We will also download the server files that will be used to run the inference endpoints. You must be familiar with Linux commands to understand the script.
    
```sh

RUN apt-get -y update && apt-get install -y --no-install-
recommends \
         wget \
         python \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*
    

```

4. Once the setup is done, we can use pip from Python to install the important Python packages.

```sh
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py && \
   pip install numpy==1.16.2 scipy==1.2.1 scikit-learn==0.20.2 pandas flask gevent gunicorn && \
        (cd /usr/local/lib/python2.7/dist-packages/scipy/.libs;
        rm *; ln ../../numpy/.libs/* .) && \
        rm -rf /root/.cache
```

5. Set the environment variables so that Python knows what the default folder is that will contain the code. Also, we will set some features of Python. We first make sure that timely log messages should be received from the container, and then we make sure that once any module is imported in Python, its .pyc file is not
created. This is done using the variables pythonunbuffered and pythondontwritebytecode, respectively.

```sh
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
```

6. Finally, the instance will instruct to copy our Data directory files to the default work directory, and then we will change the default work directory.

```sh
COPY Data /opt/program
WORKDIR /opt/program

```

## Pushing the Docker Image to ECR
We will create a shell script file, which will be used first to build the image from the Dockerfile that we created in the previous section and then to push the image to ECR. Let’s look at the step-by-step procedure for this:

1. Name the image. We will save the name in a variable. 
```sh
algorithm_name=sagemaker-random-forest
```

2. Give full read and write permission to the train and serve files so that once the container is started, there are no access denied errors.
``sh
chmod +x Data/train
chmod +x Data/serve
```

3. Get AWS configurations so that there is no stoppage when the image is being pushed. We will define the account and the region of our AWS. Remember, since we will be running this code from inside SageMaker, the information can be automatically fetched. If we are running this from your local system or anywhere outside of AWS, then we will have to give the custom values.

```sh
account=$(aws sts get-caller-identity --query Account
--output text)
region=$(aws configure get region)
region=${region:-us-east-2}
```

4. Give the path and name to the container. We will use the same name that was given in the first step. We will use this path later to push the image.

```sh
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"
```


5. Check whether the image already exists. If it doesn’t, then a new image will be created; otherwise, the same image will be updated.

```sh
aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi

```

6. Get the login credentials of the AWS account.
```sh
$(aws ecr get-login --region ${region} --no-include-email)
```

7. Build the image with the name already decided, rename it with the full name we decided on that contains the ECR address, and then finally push the code.

```sh
docker build  -t ${algorithm_name} .
docker tag ${algorithm_name} ${fullname}
docker push ${fullname}

```

Once this step is done, we have to go to the terminal, go inside the directory where your Dockerfile is present, and then type the following:

```sh
sh build_and_push.sh
```

This will start running the script and will successfully upload the image to ECR. You can then go to ECR and check whether the image exists.


![1](https://user-images.githubusercontent.com/23625821/122172808-70975100-ce81-11eb-81c1-e0aa38034802.png)


This finishes the process of creating the Docker. 


