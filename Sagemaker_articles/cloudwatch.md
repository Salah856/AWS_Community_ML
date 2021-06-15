
# Using CloudWatch with SageMaker

In this part, we will explore CloudWatch functionality in AWS in detail. Specifically, we will look at two components of CloudWatch, CloudWatch Logs and CloudWatch
Metrics, that we will use a lot while using SageMaker.


## Amazon CloudWatch

Amazon CloudWatch is a service provided by Amazon that tracks the resource activities of AWS and provides metrics related to it. It also stores the logs that are provided by every resource used.

Through these logs and metrics, a user can explore the performance of an AWS resource being used and what can be done to improve it.

When it comes to machine learning, especially with SageMaker, CloudWatch Logs gives us the output of containers in which the code is running. As we have already seen in the previous parts, machine learning algorithms run inside a Docker container attached to an EC2 instance. So, the output that originates from these containers is not directly visible.

To look at this output, we must make some adjustments to our code, and then the status can be seen directly in the Jupyter Notebook in use, or we can use CloudWatch Logs to get this output in a step-by-step manner. The output can include your model outputs, the reason why your model failed, insights into the step-by-step execution, etc. Containers are required for three jobs, and hence we have three log groups in machine learning.

1. Processing Jobs log group
2. Training Jobs log group
3. Transform Jobs log group

CloudWatch Metrics provides us with information in the form of values to variables. For example, when it comes to machine learning, CloudWatch Metrics can provide
values such as the accuracy of a model, precision, error, etc. It can also provide metrics related to resources, such as GPU utilization, memory utilization etc.

![1](https://user-images.githubusercontent.com/23625821/121852639-11560700-ccf0-11eb-8d2d-0306a20a0fe0.png)



By looking at this architecture, we can see that in addition to accessing the services through the Amazon Management Console, we can integrate alarms through Amazon SNS, which can be connected to your email. We can also set custom rules based on certain criteria. This includes starting, stopping, and terminating a process or using features such as autoscaling.


## CloudWatch Logs

In the SageMaker console, on the left side, we have a sidebar that guides us through the different operations that are possible in it. We can create notebook instances, look at different algorithms that we ran, and analyze the endpoints. We can look at the logs of all the services that we have used by viewing the log details.


![1](https://user-images.githubusercontent.com/23625821/121852903-71e54400-ccf0-11eb-880c-0de250b3e74b.png)


![1](https://user-images.githubusercontent.com/23625821/121852944-80cbf680-ccf0-11eb-82c3-6559583c1956.png)


![1](https://user-images.githubusercontent.com/23625821/121853038-a78a2d00-ccf0-11eb-80e3-e241a28c853e.png)


Click the “View logs” link, and you will see the CloudWatch page open.

![1](https://user-images.githubusercontent.com/23625821/121853123-ca1c4600-ccf0-11eb-8e86-949004978d97.png)


We are looking at the log groups. As you know, because we have selected the training job, this information is present at the top of the console.

When we scroll down, at the bottom we can see the name of the algorithm. We must click it to get to the logs


![1](https://user-images.githubusercontent.com/23625821/121853256-ffc12f00-ccf0-11eb-9e88-6ab9ee24819e.png)


Remember, the logs will start appearing only once the container has successfully started and the algorithms have started running.

If there is a problem with your Docker script, then you will not find any logs generated. But, if your code related to the algorithm has an error, then you will find the information in the logs section, as the container had successfully started, and hence the logs have started generating.

So, if your model is not running, you can come to the logs to check the error in the code. With custom containers, if the logs don’t start running, in general the error is probably in the Docker script. 





