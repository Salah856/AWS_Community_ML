
# Data Processing in AWS 

Data processing is one of the first steps of the machine learning pipeline. As different sources of data have different formats, it becomes almost impossible to handle all the formats inside the model. Hence, we give the data a synchronous structure, and then we try to process different unwanted sections of it. These sections include the null values, outliers, dummification of categorical columns, standardization of numerical columns, etc. We can use SageMaker effectively to process the data in all these domains. This chapter assumes that you have knowledge about different data processing techniques and their implementation in Python. This chapter will be dedicated to using SageMaker to do this.


## Preprocessing in Jupyter Notebook
