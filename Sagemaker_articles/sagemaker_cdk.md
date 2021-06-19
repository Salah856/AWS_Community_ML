
# Automate Amazon SageMaker Studio setup using AWS CDK

Amazon SageMaker Studio is the first fully integrated development environment (IDE) for machine learning (ML). Studio provides a single web-based visual interface where you can perform all ML development steps required to prepare data, as well as build, train, and deploy models. You can quickly upload data, create new notebooks, train and tune models, move back and forth between steps to adjust experiments, compare results, and deploy models to production all in one place, making you much more productive.

In this article, we see how to use the AWS Cloud Development Kit (AWS CDK) to use the new native resource in AWS CloudFormation to set up Studio and configure its access for data scientists and developers in your organization. This way you can set up Studio quickly and consistently, enabling you to apply DevOps best practices and meet safety, compliance, and configuration standards across all AWS accounts and Regions. We use Python as the main language, but the code can be easily changed to other AWS CDK supported languages.


The AWS CDK is an open-source software development framework to model and provision your cloud application resources using familiar programming languages.



## Prerequisites

To get started, make sure you have the following prerequisites:

- The AWS Command Line Interface (AWS CLI) installed
- The AWS CDK installed
- An AWS profile with permissions to create AWS Identity and Access Management (AWS IAM) roles, Studio domains, and Studio user profiles

First, let’s clone the demo code from GitHub using your method of choice at: https://github.com/aws-samples/aws-cdk-sagemaker-studio. 

As you clone the repo, you can observe that we have a classic AWS CDK project with the following components:

- app.py – The entry point to deploy the AWS CDK stack sagemakerStudioCDK
- sagemakerStudioConstructs – Our AWS CDK constructs using the AWS CloudFormation resources from sagemakerStudioCloudformationStack

- sagemaker-domain-template and sagemaker-user-template – The CloudFormation templates for the native resources to create the Studio domain and user profile
- sagemakerStudioCDK/sagemaker_studio_stack.py – The AWS CDK stack that calls our constructs to create first the Studio domain and add the user profile


## AWS CDK constructs

When we open sagemakerStudioConstructs/__init__.py, we find two AWS CDK constructs:

- SagemakerStudioDomainConstruct – The construct takes as input the mandatory fields required for the native resource AWS::SageMaker::Domain and outputs the Studio domain ID with the following parameters:

-- sagemaker_domain_name – The name of the newly created Studio domain
-- vpc_id – The ID of the Amazon Virtual Private Cloud (Amazon VPC) that the domain uses for communication

-- subnet_ids – The VPC subnets that the domain uses for communication
-- default_execution_role_user – The IAM execution role for the user by default

- SagemakerStudioUserConstruct – The construct takes as input the mandatory fields required for the native resource AWS::SageMaker::UserProfile, with the following parameters :

-- sagemaker_domain_id – The Studio domain ID
-- user_profile_name – The user profile name
