
# Using TensorFlow and the Serverless Framework for deep learning and image recognition


Deep and machine learning is becoming essential for a lot of businesses, be it for internal projects or external ones. The data-driven approach allows companies to build analytics tools based on their data, without constructing complicated deterministic algorithms.


Deep learning allows them to use more raw data than a machine learning approach, making it applicable to a larger number of use cases. Also, by using pre-trained neural networks, companies can start using state of the art applications like image captioning, segmentation and text analysis—without significant investment into data science team.

But one of the main issues companies face with deep/machine learning is finding the right way to deploy these models. I wholeheartedly recommend a serverless approach. Why? Because serverless provides a cheap, scalable and reliable architecture for deep learning models.


In this article, we’ll cover how to build your first deep learning API using the Serverless Framework, TensorFlow, AWS Lambda and API Gateway.


## Why Serverless + TensorFlow?

First of all, let’s briefly cover what TensorFlow is: an open source library that allows developers to easily create, train and deploy neural networks. It’s currently the most popular framework for deep learning, and is adored by both novices and experts.

Currently, the way to deploy pre-trained TensorFlow model is to use a cluster of instances.

So to make deep learning API, we would need stack like this:


![deep-learning-api-stack](https://user-images.githubusercontent.com/23625821/122518916-22fe1e00-d012-11eb-9ce7-e766a007487f.gif)


The main pain points in this infrastructure is that:

- you have to manage the cluster - its size, type and logic for scaling. 
- you have to pay for unused server power.
- you have to manage the container logic - logging, handling of multiple requests, etc.


With AWS Lambda, we can make the stack significantly easier and use simpler architecture:

![serverless-tensorflow-architecture](https://user-images.githubusercontent.com/23625821/122519107-6062ab80-d012-11eb-91dd-b78bd1a0e6e6.png)



## The difference in both approaches

First of all, a serverless serverless approach is very scalable. It can scale up to 10k concurrent requests without writing any additional logic. It’s perfect for handling random high loads, as it doesn’t take any additional time to scale.

Second, you don’t have to pay for unused server time. Serverless architectures have pay-as-you-go model. Meaning, if you have 25k requests per month, you will only pay for 25k requests.

And not only does it make pricing completely transparent, it’s just a lot cheaper. For the example TensorFlow model we’ll cover in this article, it costs 1$ for about 25k requests. A similar cluster would cost a lot more, and you’d only achieve pricing parity once you hit 1M requests.

Third, infrastructure itself becomes a lot easier. You don’t have to handle Docker containers, logic for multiple requests, or cluster orchestration.

Bottom line: you don’t have to hire someone to do devops for this, as any backend developer can easily handle it.

As we’ll see in a minute, deploying a serverless deep/machine learning infrastructure can be done with as little as 4 lines of code.


That said, when wouldn’t you go with a serverless approach? There are some limitations to be aware of:

- Lambda has strict limits in terms of processing time and used memory, you’ll want to make sure you won’t be hitting those

- As mentioned above, clusters are more cost effective after a certain number of requests. In cases where you don’t have peak loads and the number of requests is really high (I mean 10M per month high), a cluster will actually save you money.

- Lambda has a small, but certain, startup time. TensorFlow also has to download the model from S3 to start up. For the example in this part, a cold execution will take 4.5 seconds and a warm execution will take 3 seconds. It may not be critical for some applications, but if you are focused on real-time execution then a cluster will be more responsive.


## The basic 4 line example

Let’s get started with our serverless deep learning API! For this example, I’m using a pretty popular application of neural networks: image recognition. Our application will take an image as input, and return a description of the object in it.

These kinds of applications are commonly used to filter visual content or classify stacks of images in certain groups.


We’ll use the following stack:

- API Gateway for managing requests
- AWS Lambda for processing
- Serverless framework for handling deployment and configuration


### “Hello world” code

To get started, you’ll need to have the Serverless Framework installed ( https://www.serverless.com/framework/docs/providers/aws/guide/installation/ ) 


Create an empty folder and run following commands in the CLI:

```bash

serverless install -u https://github.com/ryfeus/lambda-packs/tree/master/Tensorflow/source -n tensorflow

cd tensorflow

serverless deploy

serverless invoke --function main --log


```

### Code decomposition - breaking down the model

Let’s start with serverless YAML file. Nothing uncommon here—we’re using a pretty standard deployment method:

```yaml

service: tensorflow

frameworkVersion: ">=1.2.0 <2.0.0"

provider:
  name: aws
  runtime: python2.7
  memorySize: 1536
  timeout: 300

functions:
  main:
    handler: index.handler
    

```


### Model download from S3:
```py

strBucket = 'ryfeuslambda'
strKey = 'tensorflow/imagenet/classify_image_graph_def.pb'
strFile = '/tmp/imagenet/classify_image_graph_def.pb'

downloadFromS3(strBucket,strKey,strFile)
print(strFile)

```


  ### Model import 
  
  ```py
  
  def create_graph():
    with tf.gfile.FastGFile(os.path.join('/tmp/imagenet/', 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
        
  
  ```
  
  
  ### Getting the image
  
  
  ```py
  
  strFile = '/tmp/imagenet/inputimage.jpg'
  
 if ('imagelink' in event):
    urllib.urlretrieve(event['imagelink'], strFile)
 
 else:
    strBucket = 'ryfeuslambda'
    strKey = 'tensorflow/imagenet/cropped_panda.jpg'
    
    downloadFromS3(strBucket, strKey, strFile)
    print(strFile)
        
  
  ```
