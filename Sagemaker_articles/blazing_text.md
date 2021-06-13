
# Exploring the Blazing Text Algorithm

The blazing text algorithm is a highly optimized version of the word2vec algorithm that allows faster training and inference and supports distributed training as well. Once the vectors are generated using this algorithm, we can use them for different tasks such as text classification, summarization, translation, etc. It supports two architectures, similar to that of word2vec.

1. Skip gram architecture
2. Continuous bag of words architecture

## Skip Gram Architecture of Word Vectors Generation

The skip gram algorithm is used to generate word vectors by finding words that are most similar to each other. This algorithm tries to understand the context of a sentence. To do that, it takes a word as input and then tries to predict all the words that have similar context.


![1](https://user-images.githubusercontent.com/23625821/121797858-37fa3c00-cc23-11eb-8e81-0ddefbbb2bc4.png)


To understand the context and generate word vectors, a small neural network architecture is used with hidden layers that have no activation functions. In the
beginning, each word is encoded using the one-hot encoding algorithm and then fed to the network. A weight is assigned to the hidden layer, whose value is learned through a loss function. Once the model is trained, it can be used for generating word vectors or directly used for text classification models.



## Continuous Bag of Words Architecture of Word Vectors Generation

The continuous bag of words (CBOW) method, you could say, is the reverse of skip gram. It understands the context and then tries to predict the word in that context. For example, if the sentence is “Delhi is the capital of India” and we then write “Delhi is the capital,” then it should predict India. The architecture is again the same, where we have a hidden layer and an output layer. Each word passed to the network is one-hot encoded.

