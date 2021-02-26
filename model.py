from collections import defaultdict
from typing import Tuple, Generator, Dict

import matplotlib.pyplot as plt
import numpy as np
import tqdm

class RNN(object):
    """Numpy implementation of sequence-to-one recurrent neural network for regression tasks."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):

        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # create and initialize weights of the network
        
        self.W = np.zeros((input_size, hidden_size))
        self.R = np.zeros((hidden_size, hidden_size))
        self.bs = np.zeros((hidden_size, 1))
        self.V = np.zeros((hidden_size, output_size))
        self.by = np.zeros((output_size, 1))
        self.reset_parameters()

        # place holder to store intermediates for backprop
        self.a = None
        self.y_hat = None
        self.grads = None
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        seq_length, batch_size, _ = x.shape
              
        self.a = {0:np.zeros((batch_size,self.hidden_size))}
        for i in range(len(x)):
            s_t = x[i] @ self.W  + self.a[i] @ self.R + self.bs.T
            a_t = np.tanh(s_t)
            self.a[i + 1] = a_t
            
        y_hat = a_t @ self.V + self.by.T
        
        self.x = x.copy()
        self.y_hat = y_hat.copy()
        
        
        return y_hat

    def backward(self, d_loss: np.ndarray) -> Dict:
        seq_len, _ , _ = self.x.shape
        dV = np.zeros(self.V.shape)
        dby = np.zeros((self.by.shape))
        
        dW = np.zeros((seq_len ,*self.W.shape))
        dR = np.zeros((seq_len ,*self.R.shape))
        dbs = np.zeros((seq_len ,*self.bs.shape))
        
        for i in reversed(range(seq_len)):
            if i == seq_len - 1:
                dV =  self.a[i+1].T @ d_loss
                dby = d_loss.sum(axis=0).reshape(self.by.shape)
                delta_t = (d_loss @ self.V.T) * (1 - self.a[i+1] * self.a[i+1])
            else:
                delta_t = (delta_t @ self.R.T) * (1 - self.a[i+1] * self.a[i+1])
        

            dW[i] += self.x[i].T @ delta_t
            dR[i] += self.a[i].T @ delta_t
            dbs[i] = delta_t.sum(axis=0).reshape(dbs[i].shape)
            
        self.grads = {'dW': dW, 'dR': dR, 'dV': dV, 'dbs': dbs, 'dby': dby}
        return self.grads
        
        
    def update(self, lr: float):

        if not self.grads:
            raise RuntimeError("You have to call the .backward() function first")
            
            
            
        for key, grad in self.grads.items():
            if len(grad.shape) == 3:
                self.grads[key] = grad.sum(axis=0)
       
                                           
                                           
        self.V -= lr * self.grads['dV']
        self.W -= lr * self.grads['dW']
        self.R -= lr * self.grads['dR']
        self.bs -= lr * self.grads['dbs']
        self.by -= lr * self.grads['dby']
        # reset internal class attributes
        self.grads = {}
        self.y_hat, self.a = None, None
        
    def get_weights(self) -> Dict:
        return {'W': self.W, 'R': self.R, 'V': self.V, 'bs': self.bs, 'by': self.by}
    
    def set_weights(self, weights: Dict):
        if not all(name in weights.keys() for name in ['W', 'R', 'V']):
            raise ValueError("Missing one of 'W', 'R', 'V' keys in the weight dictionary")
        self.W = weights["W"]
        self.R = weights["R"]
        self.V = weights["V"]
        self.bs = weights["bs"]
        self.by = weights["by"]
        
    def reset_parameters(self):
        # recurrent weights initialized as identity matrix
        self.R = np.eye(self.hidden_size)
        
        # input to hidden and hidden to output initialized with LeCun initialization
        gain = np.sqrt(3 / self.input_size)
        self.W = np.random.uniform(-gain, gain, self.W.shape)
        gain = np.sqrt(3 / self.hidden_size)
        self.V = np.random.uniform(-gain, gain, self.V.shape)



class MSELoss(object):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.y_hat = None
        self.y_true = None
        
    def forward(self, y_hat: np.ndarray, y_true: np.ndarray) -> float:
        """ MSE loss
        
        Parameters
        ----------
        y_hat : np.ndarray
            Array containing the network predictions of shape [batch_size, 1]
        y_true : np.ndarray
            Array containing the true values of shape [batch_size, 1]
        
        Returns:
        The mean square error as a floating number.
        """

        self.y_hat = y_hat
        self.y_true = y_true
        return np.mean(np.square(self.y_true - self.y_hat))
    
    def backward(self) -> np.ndarray:
        """Backward pass of the MSE
        
        Returns
        -------
        The gradient w.r.t the network output of shape [batch_size, 1]
        """
          
        res = -2 * np.sum(self.y_true - self.y_hat,axis = 1)/self.y_true.shape[0]
        res = np.expand_dims(res, axis=1)
        return res