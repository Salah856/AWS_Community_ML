
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


