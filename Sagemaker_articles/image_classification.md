

# Exploring the Image Classification Algorithm

SageMaker’s image classification algorithm is based upon a special convolutional neural network architecture called a ResNet. Before looking at the application of this algorithm, let’s first explore and understand the ResNet architecture used for image classification.


### ResNet

A ResNet is an architecture that is based on the framework of convolutional neural networks and used for problem statements such as image classification. To understand a ResNet, we must first look at the operation of convolutional neural networks. 


![1](https://user-images.githubusercontent.com/23625821/121798505-0f744100-cc27-11eb-99b9-4b90764e1ceb.png)


A typical CNN consists of the following operations:

1. The first operation is the convolution operation, which is also considered an application of filters. We apply different filters on the image so that we can get different versions of the same image, which helps us understand the image perfectly. But, instead of hard-coding the filters, the values of these filters are learned using the backpropagation approach.

2. The next step is called pooling or subsampling. Here, we reduce the size of the image so that the training time becomes faster. There are different types of pooling approaches such as max- pooling, average-pooling, etc.

3. The previous two processes are repeated multiple times, and then the final pooling operation’s output is given to a fully connected neural network layer. Here the major learning happens, and finally the classification task is done.

