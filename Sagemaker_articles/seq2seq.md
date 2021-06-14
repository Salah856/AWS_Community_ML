
# Exploring the SeqToSeq Algorithm

Amazon’s sequence-to-sequence algorithm is based upon recurrent neural networks, convolutional neural networks, and an encoder-decoder architecture to understand the context more efficiently. The next section is a brief overview of the RNN and encoder-decoder architectures.


## Recurrent Neural Networks

When we deal with sequential data or time-based data, it becomes necessary to remember a few things from the past and understand how it can be used to predict the outcome. This is not possible with using normal artificial neural networks or convolutional neural networks. Therefore, a new architecture called RNN is used whenever we deal with sequential data.


![1](https://user-images.githubusercontent.com/23625821/121837323-95e75c00-ccd5-11eb-91bf-9adeaa03cdc6.png)



For example, in text classification, each word of the text is taken, some neural network–based computations are applied, and important aspects are stored and then
passed to the next RNN layer. Storage happens in h, words are sent through x, while the output is received through y. The words are not directly passed, but they are converted into vectors and then passed. We can use algorithms such as word2vec, glove, or blazing text in SageMaker to generate these word vectors.

There are various modifications to RNNs that solve the shortcomings present in the original versions. Two of the most used are long short-term memory (LSTM) and gated recurrent units (GRU).


## Encoder-Decoder Architecture

![1](https://user-images.githubusercontent.com/23625821/121837386-bc0cfc00-ccd5-11eb-9668-13619e937266.png)

An encoder is mostly used to not only memorize the past and give accurate predictions but also to understand the context of the text passed. We can use normal
RNNs or LSTMS and GRUs. Once the encoders look at all the word vectors, they generate the encoder vectors and pass them to the decoder. The encoder vector suffices all the information that the encoder has received, and the decoder uses it to make efficient predictions.


The decoder takes these encoder vectors, feeds them to RNNs of its own, and then applies a softmax activation function to give the output. The best advantage of this architecture, apart from understanding the context, is its ability to take variable-length input and give variable-length output.



## SageMaker Application of SeqToSeq

Let’s understand the algorithm in more detail by applying it to the machine translation use case; that is, let’s translate something from English to German.


```py
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
from time import gmtime, strftime

import time
import numpy as np
import os

import json
import boto3
import re


region_name = boto3.Session().region_name
bucket = 'slytherins-test'

prefix = 'seq2seq-E2G'
role = get_execution_role()

container = get_image_uri(region_name, 'seq2seq')


```


So, in the previous steps we have defined the container of the algorithm and defined our bucket and the folder inside where the entire model-related files will be saved. The next step will be to have a dataset. The Seq2Seq algorithm has two approaches. In the first, you can use the pretrained model available for the predictions. So, for our example, a model already exists that is trained on English to German machine translation. Or, we can train the model on our own corpus and then use it for the predictions. This process may take a lot of time, but it is the best when used for domain-specific translation tasks.


We will first see how to train the model on a corpus, and then we will use the pretrained model for predictions. The data that we will be using is news data. We will have files that contain news commentary in English and its translation in German. We can get these files from http://data.statmt.org/wmt17/translation-task/.


Let’s download the data from inside the notebook and create our training and validation sets.

```sh 

! wget http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/corpus.tc.de.gz
! wget http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/corpus.tc.en.gz

! gunzip corpus.tc.de.gz
! gunzip corpus.tc.en.gz

! mkdir validation
! curl http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/dev.tgz | tar xvzf - -C validation

```

The previous files that we have downloaded are big, around 250 MB each. So, if we train the model on the entire dataset, it may take days to finish. Therefore, we can take a subset of the entire data and use it for training.

```sh

! head -n 10000 corpus.tc.en > corpus.tc.en.small
! head -n 10000 corpus.tc.de > corpus.tc.de.small

```

The previous subset created has 10,000 rows. We will use this small dataset for training. The next step will be to generate English and German vocabulary from the
previous files. This will use the tokenization and other NLP components to generate the vocabulary.


```bash

python3 create_vocab_proto.py \
      --train-source corpus.tc.en.small \
      --train-target corpus.tc.de.small \
      --val-source validation/newstest2014.tc.en \
      --val-target validation/newstest2014.tc.de
      


```

The previous Python script takes as input the source English text and target German text. It applies the preprocessing to generate the vocabulary. Finally, it saves the English and German vocabulary in the validation folder. 


```py

def upload_to_s3(bucket, prefix, channel, file):
    s3 = boto3.resource('s3')
    data = open(file, "rb")
    key = prefix + "/" + channel + '/' + file
    s3.Bucket(bucket).put_object(Key=key, Body=data)
    

upload_to_s3(bucket, prefix, 'train', 'train.rec') 
upload_to_s3(bucket, prefix, 'validation', 'val.rec') 

upload_to_s3(bucket, prefix, 'vocab', 'vocab.src.json') 
upload_to_s3(bucket, prefix, 'vocab', 'vocab.trg.json') 


```

The code that we just executed generates two files. One is the vocabulary that is generated, and the second is the RecordIO-Protobuf version of the data. We will upload both of these files to S3 using the previous code.

All the basic steps are complete now, and we want to now initialize the algorithm. We will do that using the code shown here:


```py

job_name = 'seq2seq-E2G'
print("Training job", job_name)

create_training_params = {
    "AlgorithmSpecification": {
        "TrainingImage": container,
        "TrainingInputMode": "File"
    },
    "RoleArn": role,
    "OutputDataConfig": {
        "S3OutputPath": "s3://{}/{}/".format(bucket, prefix)
    },
    "ResourceConfig": {
          # Seq2Seq does not support multiple machines. Currently, it only supports single machine, multiple GPUs
          "InstanceCount": 1,
          
          "InstanceType": "ml.m4.xlarge",
          # We suggest one of ["ml.p2.16xlarge", "ml.p2.8xlarge", "ml.p2.xlarge"]
          "VolumeSizeInGB": 5
    },
    "TrainingJobName": job_name,
    "HyperParameters": {
        # Please refer to the documentation for complete list of parameters
        "max_seq_len_source": "60",
        "max_seq_len_target": "60",
        "optimized_metric": "bleu",
        "batch_size": "64",  # Please use a larger batch size (256 or 512) if using ml.p2.8xlarge or ml.p2.16xlarge
        "checkpoint_frequency_num_batches": "1000",
        "rnn_num_hidden": "512",
        "num_layers_encoder": "1",
        "num_layers_decoder": "1",
        "num_embed_source": "512",
        "num_embed_target": "512"
    },
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 48 * 3600
    },
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://{}/{}/train/".format(bucket, prefix),
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
        },
        {
            "ChannelName": "vocab",
            "DataSource": {
                "S3DataSource": {"S3DataType": "S3Prefix",
                    "S3Uri": "s3://{}/{}/vocab/".format(bucket, prefix),
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
        },
        {
            "ChannelName": "validation",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://{}/{}/validation/".format(bucket, prefix),
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
        }
    ]
}


sagemaker_client = boto3.Session().client(service_name='sagemaker')
sagemaker_client.create_training_job(**create_training_params)

```


This code will start the execution of the training and will take a lot of hours to finish. Remember, this algorithm requires a GPU instance for execution. So, whatever instance you select will be chargeable. Choose wisely.

Now, let’s look at how we can use the pretrained model that already exists and do the inference on the test dataset by exposing the endpoint. When we train the previous model, we will get three files:

- Model.tar.gz
- Vocab.src.json
- Vocab.trg.json

