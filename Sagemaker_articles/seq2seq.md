
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

