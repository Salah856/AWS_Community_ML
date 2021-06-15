
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

