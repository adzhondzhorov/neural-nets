Building LLMs from scratch with step-by-step examples in notebooks.
Using only numpy for randomization of lists and nothing else outside of standard Python libraries. Partially influenced by [Karpathy's micrograd](https://github.com/karpathy/micrograd). 

1. Linear Algebra - Matrix, Vector and Tensor operations
2. Basic Calculus - Value and Derivative classes for backpropagation
3. Basic NN - Linear, Sigmoid, ReLU layers, training loop
4. Regression example - Boston House Price Dataset, includes Normalizer and MSE
5. Classification example - Iris Dataset, includes Softmax, OHE and NLL loss
6. Basic tokenization - BPE algorithm
7. Text classification example - IMDB dataset for tokenizer training, Flipkart reviews for sentiment analysis, Embedding, Flatten layers
   
8*. Hitting an issue with speed in several areas:
   - Value class creation
   - Matrix multiplication
   - backpropagation
Might need to reconsider not using external libraries and algorithms.

9. Losses and metrics in separate module - MSE, NLL, BCE, accuracy

