
# Automate Amazon SageMaker Studio setup using AWS CDK

Amazon SageMaker Studio is the first fully integrated development environment (IDE) for machine learning (ML). Studio provides a single web-based visual interface where you can perform all ML development steps required to prepare data, as well as build, train, and deploy models. You can quickly upload data, create new notebooks, train and tune models, move back and forth between steps to adjust experiments, compare results, and deploy models to production all in one place, making you much more productive.

In this article, we see how to use the AWS Cloud Development Kit (AWS CDK) to use the new native resource in AWS CloudFormation to set up Studio and configure its access for data scientists and developers in your organization. This way you can set up Studio quickly and consistently, enabling you to apply DevOps best practices and meet safety, compliance, and configuration standards across all AWS accounts and Regions. We use Python as the main language, but the code can be easily changed to other AWS CDK supported languages.

