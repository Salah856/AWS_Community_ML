
#  End-to-End Pipeline in SageMaker

In this part, we will see how we can make an end-to-end pipeline of an entire machine learning process. We can use a combination of AWS services to automate the entire process of machine learning.


## Overview of Step Functions
AWS Step Functions is the service provided by Amazon that you can use to create workflows and automate them. These workflows consist of AWS resources, algorithms,
and processing. They may also include resources that are outside AWS. We can use Step Functions to create an end-to-end automation framework that helps us in building an effective continuous integration and continuous development (CI/CD) DevOps pipeline.

Each component in a step function is called a state machine. In this part, we will be creating multiple state machines, as follows:

- State machine for training a model
- State machine for saving the model

- State machine for configuring endpoints
- State machine for model deployment


Then we will combine all the state machines in a sequential format so that the entire process can be automated. 

![1](https://user-images.githubusercontent.com/23625821/122373789-102c1080-cf62-11eb-96af-e86e1659e6dc.png)
