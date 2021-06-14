
# Exploring the SeqToSeq Algorithm

Amazon’s sequence-to-sequence algorithm is based upon recurrent neural networks, convolutional neural networks, and an encoder-decoder architecture to understand the context more efficiently. The next section is a brief overview of the RNN and encoder-decoder architectures.


## Recurrent Neural Networks

When we deal with sequential data or time-based data, it becomes necessary to remember a few things from the past and understand how it can be used to predict the outcome. This is not possible with using normal artificial neural networks or convolutional neural networks. Therefore, a new architecture called RNN is used whenever we deal with sequential data.


![1](https://user-images.githubusercontent.com/23625821/121837323-95e75c00-ccd5-11eb-91bf-9adeaa03cdc6.png)



For example, in text classification, each word of the text is taken, some neural network–based computations are applied, and important aspects are stored and then
passed to the next RNN layer. Storage happens in h, words are sent through x, while the output is received through y. The words are not directly passed, but they are converted into vectors and then passed. We can use algorithms such as word2vec, glove, or blazing text in SageMaker to generate these word vectors.

There are various modifications to RNNs that solve the shortcomings present in the original versions. Two of the most used are long short-term memory (LSTM) and gated recurrent units (GRU).

