
# Build accurate ML training datasets using point-in-time queries with Amazon SageMaker Feature Store and Apache Spark 

GoDaddy is the world’s largest services platform for entrepreneurs around the globe, empowering their worldwide community of over 20 million customers—and entrepreneurs everywhere—by giving them all the help and tools they need to grow online. GoDaddy needs a robust, validated, and customizable ML feature management solution, and has chosen Amazon SageMaker Feature Store to manage thousands of features across dozens of feature groups with unique data pipelines and update schedules. Feature Store lets GoDaddy use point-in-time queries to support accurate training and deployment of machine learning (ML) models, covering everything from personalizing content, to preventing fraud, to helping customers find the perfect domain name.

Feature Store lets you define groups of features, use batch ingestion and streaming ingestion, retrieve features with as low as single-digit millisecond latency for highly accurate online predictions, and extract point-in-time correct datasets for training. Instead of building and maintaining these infrastructure capabilities, you get a fully managed service that scales as your data grows, enables feature sharing across teams, and lets data scientists focus on building great ML-driven products for game-changing business use cases. Teams can now deliver robust features and reuse them in a variety of models that may be built by different teams.


## Avoid data leakage by using point-in-time correct queries

In ML, data leakage or target leakage is accidentally using data in model training that wouldn’t be available at the time of prediction. Leakage can be subtle and difficult to detect, yet the business impact can be significant. Models with leakage perform unrealistically well in development, but they deliver poor model accuracy in production without the benefit of future data.


So how do data science teams provide a rich set of ML features, while ensuring they don’t leak future data into their trained models? Companies are increasingly adopting the use of a feature store to help solve this model training challenge. A robust feature store provides an offline store with a complete history of feature values. Feature records include timestamps to support point-in-time correct queries. Data scientists can query for the exact set of feature values that would have been available at a specific time, without the chance of including data from beyond that time.

Let’s use a diagram to explain the concept of a point-in-time feature query. Imagine we’re training a fraud detection model on a set of historical transactions. Each transaction has features associated with various entities involved in the transaction, such as the consumer, merchant, and credit card. Feature values for these entities change over time, and they’re updated on different schedules. To avoid leaking future feature values, a point-in-time query retrieves the state of each feature that was available at each transaction time, and no later. For example, the transaction at time t2 can only use features available before time t2, and the transaction at t1 can’t use features from timestamps greater than t1.



![1-3177-Diagram](https://user-images.githubusercontent.com/23625821/123223189-fabe6580-d4d0-11eb-850f-83bf265f1e64.jpg)




## Sources 

<a href="https://aws.amazon.com/blogs/machine-learning/build-accurate-ml-training-datasets-using-point-in-time-queries-with-amazon-sagemaker-feature-store-and-apache-spark/"> AWS Blog </a>



