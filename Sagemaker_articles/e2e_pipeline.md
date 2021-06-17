
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


Let’s start the process of creating the previous workflow. The first step will be to upgrade the Step Functions package so that we can make sure we are using the latest version of the module.

## Upgrading Step Functions

We will simply use pip from Python to upgrade the Step Functions package and all the dependent packages.

```sh
python -m pip install --upgrade stepfunctions
```

You can run this either from the terminal or from the Jupyter Notebook as well by adding a prefix of exclamation mark (!).


## Defining the Required Parameters

Let’s now define the required objects that we will use to run our code. This includes the roles, region, bucket, etc.

```py

import boto3
import sagemaker
import time
import random
import uuid

import logging
import stepfunctions
import io
import random

from sagemaker.amazon.amazon_estimator import get_image_uri
from stepfunctions import steps
from stepfunctions.steps import TrainingStep, ModelStep, TransformStep

from stepfunctions.inputs import ExecutionInput
from stepfunctions.workflow import Workflow

from stepfunctions.template import TrainingPipeline
from stepfunctions.template.utils import replace_parameters_with_jsonpath


sagemaker_execution_role = sagemaker.get_execution_role()

workflow_execution_role = "arn:aws:iam::809912564797:role/<<your name>>-step-functions"

session = sagemaker.Session()
stepfunctions.set_stream_logger(level=logging.INFO)
region = boto3.Session().region_name

prefix = 'sagemaker/big-mart-pipeline'
bucket_path = 'https://s3-{}.amazonaws.com/{}'.format(region, "slytherins-test")


```



As you can see in the code, we require two roles. One is the SageMaker execution role, and the second is the workflow execution role. In the next section, we will see how to define the role for workflow execution. In addition, we have created a SageMaker session and defined the region and S3 bucket location. We have also set the Step Functions logger so that whatever important messages are there, we will not miss them.

Now let’s see how we can create the required IAM role for workflow execution.


## Setting Up the Required Roles

We need to set up two things to be able to execute the workflow:

1. We need to add a policy on the already existing SageMaker role.
2. We need to create a new Step Functions IAM role.


### Adding a Policy to the Existing SageMaker Role

It’s easy to update the policy so that it can access the features of Step Functions. In the SageMaker console, we need to click the name of the notebook instance that we are using. This will lead us to a page showing the properties of the notebook instance.

In that page there will be a section named “Permissions and encryption.” There you will find your ARN role mentioned for the instance.


![1](https://user-images.githubusercontent.com/23625821/122375345-6e0d2800-cf63-11eb-868b-4cecd1574393.png)


Once you click that role, you’ll move to the IAM role for that ARN. On that page, you’ll need to click Attach Policies and search for AWSStepFunctionsFullAccess. Attach this policy, and now your SageMaker instance is ready to use Step Functions.


![2](https://user-images.githubusercontent.com/23625821/122375462-88470600-cf63-11eb-8b54-3c8180456994.png)


### Creating a New IAM Role for Step Functions

Once we are done with enabling the instance to execute a Step Functions job, we need to create an execution role so that Step Functions is able to execute the jobs that are created. For this, again we need to go to the IAM console and create this role. Go to the IAM console, go to the Roles section, and then click “Create role.”

![4](https://user-images.githubusercontent.com/23625821/122375821-deb44480-cf63-11eb-8cd1-6f08204a8082.png)


Select the Step Functions service. You may need to search for the service.

![3](https://user-images.githubusercontent.com/23625821/122375852-e378f880-cf63-11eb-8034-73486273e809.png)


Now, continue the process and keep clicking Next until you arrive at the section where you need to provide the role name. Give any role name you want and then click “Create role.” Next, once we have created the role, we need to attach a policy to it. Here we will list all the services that the Step Functions service is allowed to do. We provide this list in JSON format.


Click the role that you have just created, and then in the Permissions section click “Add inline policy.”


![5](https://user-images.githubusercontent.com/23625821/122376154-2fc43880-cf64-11eb-983d-c47886c7b720.png)


Here, you need to add a JSON file on the JSON tab. The file contents are shown here:

```json

{
  "Version": "2012-10-17",
  "Statement": [
      {
         "Effect": "Allow",
         "Action": [
           "sagemaker:CreateTransformJob",
           "sagemaker:DescribeTransformJob",
           
           "sagemaker:StopTransformJob",
           "sagemaker:CreateTrainingJob",
           "sagemaker:DescribeTrainingJob",
           
           "sagemaker:StopTrainingJob",
           "sagemaker:CreateHyperParameterTuningJob",
           
           "sagemaker:DescribeHyperParameterTuningJob",
           "sagemaker:StopHyperParameterTuningJob",
           
           "sagemaker:CreateModel",
           "sagemaker:CreateEndpointConfig",
           
           "sagemaker:CreateEndpoint",
           "sagemaker:DeleteEndpointConfig",
           
           "sagemaker:DeleteEndpoint",
           "sagemaker:UpdateEndpoint",
           "sagemaker:ListTags",
           
           "lambda:InvokeFunction",
           "sqs:SendMessage",
           "sns:Publish",
           
           "ecs:RunTask",
           "ecs:StopTask",
           
           "ecs:DescribeTasks",
           "dynamodb:GetItem",
           
           "dynamodb:PutItem",
           "dynamodb:UpdateItem",
           
           "dynamodb:DeleteItem",
           "batch:SubmitJob",
           "batch:DescribeJobs",
           
           "batch:TerminateJob",
           "glue:StartJobRun",
           
           "glue:GetJobRun",
           "glue:GetJobRuns",
           "glue:BatchStopJobRun"
            ],
         "Resource": "*"
     },
     {
         "Effect": "Allow",
         "Action": [
            "iam:PassRole"
       ],
         "Resource": "*",
          "Condition": {
            "StringEquals": {
               "iam:PassedToService": "sagemaker.amazonaws.com"
            }
         }
      }, 
      {
         "Effect": "Allow",
         "Action": [
            "events:PutTargets",
            "events:PutRule",
            "events:DescribeRule"
         ],
         "Resource": [
           
           "arn:aws:events:*:*:rule/StepFunctionsGetEventsForSageMakerTrainingJobsRule",
           "arn:aws:events:*:*:rule/StepFunctionsGetEventsForSageMakerTransformJobsRule",
           "arn:aws:events:*:*:rule/StepFunctionsGetEventsForSageMakerTuningJobsRule",
           
           "arn:aws:events:*:*:rule/StepFunctionsGetEventsForECSTaskRule",
           "arn:aws:events:*:*:rule/StepFunctionsGetEventsForBatchJobsRule"
          
          ]
      }
   ]
}


```



