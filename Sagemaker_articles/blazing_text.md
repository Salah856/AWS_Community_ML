
# Exploring the Blazing Text Algorithm

The blazing text algorithm is a highly optimized version of the word2vec algorithm that allows faster training and inference and supports distributed training as well. Once the vectors are generated using this algorithm, we can use them for different tasks such as text classification, summarization, translation, etc. It supports two architectures, similar to that of word2vec.

1. Skip gram architecture
2. Continuous bag of words architecture

## Skip Gram Architecture of Word Vectors Generation

The skip gram algorithm is used to generate word vectors by finding words that are most similar to each other. This algorithm tries to understand the context of a sentence. To do that, it takes a word as input and then tries to predict all the words that have similar context.


![1](https://user-images.githubusercontent.com/23625821/121797858-37fa3c00-cc23-11eb-8e81-0ddefbbb2bc4.png)


To understand the context and generate word vectors, a small neural network architecture is used with hidden layers that have no activation functions. In the
beginning, each word is encoded using the one-hot encoding algorithm and then fed to the network. A weight is assigned to the hidden layer, whose value is learned through a loss function. Once the model is trained, it can be used for generating word vectors or directly used for text classification models.



## Continuous Bag of Words Architecture of Word Vectors Generation

The continuous bag of words (CBOW) method, you could say, is the reverse of skip gram. It understands the context and then tries to predict the word in that context. For example, if the sentence is “Delhi is the capital of India” and we then write “Delhi is the capital,” then it should predict India. The architecture is again the same, where we have a hidden layer and an output layer. Each word passed to the network is one-hot encoded.


![1](https://user-images.githubusercontent.com/23625821/121797933-96271f00-cc23-11eb-8c43-1c4b0e9dd09a.png)


## SageMaker Application of Blazing Text

Before starting the coding, we must understand the dataset for which we will be generating the word vectors using the blazing text algorithm. The dataset that we’ll be using is called the text8 dataset. It is a small, cleaned version of the entire Wikipedia text.


The entire Wikipedia dump is called wiki9, which is then cleaned and converted into fil9. A subset (100 MB) of this cleaned dataset is taken and called text8. We can download the dataset from http://mattmahoney.net/dc/text8.zip 

As you may already know by now, the data downloaded must be sent to the S3 bucket so that our resources and the algorithm container can access it. We can upload
the data using the following script:

```py 

train_channel = prefix + '/train'

sess.upload_data(path='text8', bucket=bucket, key_prefix=train_channel)

s3_train_data = 's3://{}/{}'.format(bucket, train_channel)

```

Now that we have stored the data and defined the path, the next step will be to initialize the blazing text Docker container.

```py

container = sagemaker.amazon.amazon_estimator.get_image_uri(region_name, "blazingtext", "latest")

```

Once the container is ready, we have to initialize the instance/resource.
```py

bt_model = sagemaker.estimator.Estimator(container,
                     role,
                     train_instance_count=1,
                     train_instance_type='ml.m4.xlarge',
                     train_volume_size = 5,
                     train_max_run = 360000,
                     input_mode= 'File',
                     output_path=s3_output_location,
                     sagemaker_session=sess)
                     
```

Don’t forget to define the S3 output location before running this code.

```py
s3_output_location = 's3://{}/{}/output'.format(bucket, prefix)
```

Remember, the ml.m4.xlarge instance comes under the free tier. So if you want to play around with different algorithms, always use this instance. Next, we will
set up the algorithm hyperparameters.

```py

bt_model.set_hyperparameters(mode="batch_skipgram",
              epochs=5,
              min_count=5,
              sampling_threshold=0.0001,
              learning_rate=0.05,
              window_size=5,
              vector_dim=100,
              negative_samples=5,
              batch_size=11,
              evaluation=True,
              subwords=False)
              

```

Now we will pass the data as a JSON file to train the algorithm. Before doing that, we must tell the algorithm that the data is coming from S3.

 ```py
 
train_data = sagemaker.session.s3_input(s3_train_data, content_type='text/plain', s3_data_type='S3Prefix')

data_channels = {'train': train_data}

bt_model.fit(inputs=data_channels, logs=True)
 
 ```
The logs parameter will not only train the model but will also show the model output in the same Jupyter Notebook. Otherwise, we would have to look at the output in the CloudWatch. The next steps will be the same as before. Deploy the model and test it.


```py

bt_endpoint = bt_model.deploy(initial_instance_count = 1,instance_type ='ml.m4.xlarge')

words = ["awesome", "blazing"]
payload = {"instances" : words}

response = bt_endpoint.predict(json.dumps(payload))
vecs = json.loads(response)

print(vecs)


```

Here we will get the output, which will be the word vectors generated for the words awesome and blazing. Finally, we will delete the model endpoint.

```py
sess.delete_endpoint(bt_endpoint.endpoint)
```


