# RNN implementation from scratch using Pytorch

The main goal is to use RNN to solve the memory task.The task is to remember the first element in the input sequence. That is, given a sequence of random numbers, the RNN should output at the last time step the first element of the input sequence.This can be framed as a regression task.

In order to complete this goal the following steps must be done:

1. Implement a generator that yields unlimimted training batches for the memory task.
2. Implement classical many-to-one RNN
3. Implement the mean squared error loss function and its derivative to be able to compute the gradients of the loss w.r.t. the network parameters.
4. Implement a learner class to facilitate the model training.
5. Train models for different sequence lengths and test up to which sequence length you are able to train the RNN succesfully. Visualize your results.
6. Visualize the vanishing gradient problem.


### Memory Task with different sequence length
In order to show the results of the 5th task I have used four different lengths of the sequences(file memory_task.py):

1. Sequence length = 10:
* The result was very good. Model learns well and MSE loss decreases over time.
2. Sequence length  = 15:
* It starting to have problems with vanishing gradient, but not so dramatic,so it still can handle the task, but not that well as with sequence length = 10
3. Sequence length = 20:
* Problem with vanishing gradient descent is obvious, with the number of iterations loss increases
4. Sequence length = 25:
* Problem with vanishing gradient descent is obvious, with the number of iterations loss increases. Loss is huge compared to sequence with length = 10

### Vanshing gradient decsent vizualisation

For this task the gradients over time for two different sequence lengths were compared - one sequence length, where the RNN still is able to learn(Sequence length = 10) and one, where it does not(Sequence length = 25).
