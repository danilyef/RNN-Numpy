from collections import defaultdict
from typing import Tuple, Generator, Dict

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from model import RNN, MSELoss

def generate_samples(batch_size: int, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Data generator for memory task
    
    Note: Implement this function as a Python generator.
    
    Parameters
    ----------
    batch_size : int
        Number of samples in one batch
    seq_length : int
        Length of sequence of random numbers
        
    Returns
    -------
    x : np.ndarray
        Array of shape [sequence length, batch size, 1], where each sample is a sequence
        of random generated numbers between -1 and 1.
    y : np.ndarray
        Array of shape [batch size, 1], where each element i contains the label corresponding
        to sample i of the input array. The label is the first element of the sequence.
    
    """
    while True:
        x = np.random.uniform(-1,1,size = (seq_length,batch_size,1))
        y = x[0,:,:]
        yield (x,y)


class Learner(object):
    def __init__(self, model: RNN, loss_obj: MSELoss, data_generator: Generator):
        """The initialization method
        
        Parameters
        ----------
        model : RNN
            An instance of the NumPy RNN implementation.
        loss_obj: MSELoss
            An instance of the mean squared error loss class.
        data_generator : Generator
            The data generator function implemented above
        """
        self.model = model
        self.loss_obj = loss_obj
        self.data_generator = data_generator
        
        self.loss_values = {}
        self.y_hats = None
        self.y_trues = None
        self.gradients = []
        
    def train(self, iter_steps: int, lr: float, log_steps: int = 50):
        """The training method.
        
        This function implements the training loop for a given number of
        iteration steps.
        
        Parameters
        ----------
        iter_steps : int
            Number of training iteration steps.
        lr : float
            Learning rate used for the weight update.
        log_steps : int
            Interval to log the training loss, by default 50.
        """
        if not self.loss_values:
            start_step = 0
        else:
            start_step = list(self.loss_values.keys())[-1]
        pbar = tqdm.tqdm_notebook(self.data_generator, total=iter_steps)
        for x, y in pbar:
            pbar.update()
            if pbar.n < iter_steps:
   
                #training steps
                y_hat = self.model.forward(x)
                self.y_trues = y
                self.y_hats = y_hat
                
                loss = self.loss_obj.forward(self.y_hats,self.y_trues)
                loss_grad = self.loss_obj.backward()
                gradients = self.model.backward(loss_grad)
                self.gradients.append(gradients)
                model.update(lr)
                
                # log loss value
                if (pbar.n == 1) or (pbar.n % log_steps == 0):
                    self.loss_values[start_step + pbar.n] = loss
                    pbar.set_postfix_str(f"Loss: {loss:5f}")
            else:
                tqdm.tqdm.write("finished training")
                break
    
    def make_predictions(self, n_batches: int) -> Tuple[np.ndarray, np.ndarray]:
        """Predictions for a given number of random batches.
        
        Parameters
        ----------
        n_batches : int
            Number of batches to get networks predictions for.
            
        Returns
        -------
        y_hats : np.ndarray
            NumPy array containing the concatenated network predictions for all batches.
        y_trues : np.ndarray
            NumPy array containing the concatenated true labels for all batches.
        """
        for i in range(n_batches):
            x,y = next(self.data_generator)
            y_hat = self.model.forward(x)
            if i == 0:
                y_hat_all = y_hat
                y_all = y
            else:
                y_hat_all = np.concatenate((y_hat_all, y_hat), axis=0)
                y_all = np.concatenate((y_all, y), axis=0)
        return (y_hat_all,y_all)
                    
    def plot_loss(self, figsize: Tuple[float, float]=(10, 8)):
        """Plot training loss curve.
        
        Parameters
        ----------
        figsize : Tuple[float, float]
            Matplotlib figure size, by default (10,8)
        """
        if not self.loss_values:
            raise RuntimeError("You have to train the network first.")
            
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(list(self.loss_values.keys()), list(self.loss_values.values()))
        ax.set_xlabel("Weight updates")
        ax.set_ylabel("MSE")
    
    def scatter_plot(self,figsize: Tuple[float, float]=(10, 8)):
        """Scatter plot of true vs predicted values.
        
        Parameters
        ----------
        figsize : Tuple[float, float]
            Matplotlib figure size, by default (10,8)
        """        
        if any(val is None for val in [self.y_hats, self.y_trues]):
            raise RuntimeError("Call the .make_predictions() method first")
    
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(self.y_trues, self.y_hats)
        ax.plot([-1,1], [-1,1], '--', color='k', zorder=0)
        ax.set_xlabel("True values")
        ax.set_ylabel("Predicted values")



loss_obj = MSELoss()
model = RNN(input_size=1, hidden_size=10, output_size=1)
data_generator = generate_samples(batch_size=1024, seq_length=10)

learner = Learner(model=model, loss_obj=loss_obj, data_generator=data_generator)
learner.train(iter_steps=5000, lr=1e-2, log_steps=50)
learner.plot_loss()
_ = learner.make_predictions(n_batches=1)
learner.scatter_plot()