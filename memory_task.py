from collections import defaultdict
from typing import Tuple, Generator, Dict

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from model import RNN, MSELoss
from main import Learner,generate_samples

#Memory task visualization:

loss_obj = MSELoss()
#Sequence length = 10:
loss_10 = []
for i in range(10):
    model = RNN(input_size=1, hidden_size=10, output_size=1)
    data_generator_10 = generate_samples(batch_size=1024, seq_length= 10)
    learner = Learner(model=model, loss_obj=loss_obj, data_generator=data_generator_10)
    learner.train(iter_steps=5000, lr=1e-2, log_steps=500)
    loss_10.append(learner.loss_values)

#Sequence length = 15:
loss_15 = []
for i in range(10):
    model = RNN(input_size=1, hidden_size=10, output_size=1)
    data_generator_15 = generate_samples(batch_size=1024, seq_length= 15)
    learner = Learner(model=model, loss_obj=loss_obj, data_generator=data_generator_15)
    learner.train(iter_steps=5000, lr=1e-2, log_steps=500)
    loss_15.append(learner.loss_values)    
    
#Sequence length = 20: 
loss_20 = []
for i in range(10):
    model = RNN(input_size=1, hidden_size=10, output_size=1)
    data_generator_20 = generate_samples(batch_size=1024, seq_length= 20)
    learner = Learner(model=model, loss_obj=loss_obj, data_generator=data_generator_20)
    learner.train(iter_steps=5000, lr=1e-2, log_steps=500)
    loss_20.append(learner.loss_values)   
    
#Sequence length = 25:
loss_25 = []
for i in range(10):
    model = RNN(input_size=1, hidden_size=10, output_size=1)
    data_generator_25 = generate_samples(batch_size=1024, seq_length= 25)
    learner = Learner(model=model, loss_obj=loss_obj, data_generator=data_generator_25)
    learner.train(iter_steps=5000, lr=1e-2, log_steps=500)
    loss_25.append(learner.loss_values)



#calculating average losses(for every times) for every setting:
def loss_avg(loss):
    loss_avg = dict((key,0) for key in loss[0].keys())
    for i in range(len(loss)):
        for key,values in loss[i].items():
            loss_avg[key] += values
            if i == len(loss) - 1:
                  loss_avg[key] /= len(loss)
                    
    return loss_avg

loss_10 = loss_avg(loss_10)
loss_15 = loss_avg(loss_15)
loss_20 = loss_avg(loss_20)
loss_25 = loss_avg(loss_25)


labels = list(loss_10.keys())
loss_10_means = list(loss_10.values())
loss_15_means = list(loss_15.values())
loss_20_means = list(loss_20.values())
loss_25_means = list(loss_25.values())


x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots()
bar1 = ax.bar(x - width, loss_10_means, width, label='Length 10')
bar2 = ax.bar(x , loss_15_means, width, label='Length 15')
bar3 = ax.bar(x + width, loss_20_means, width, label='Length 20')
bar4 = ax.bar(x + 2*width, loss_25_means, width, label='Length 25')

#labels and title
ax.set_ylabel('MSE')
ax.set_xlabel('Number of iterations')
ax.set_title('MSE by sequence length')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()

#Vanishing Gradient visualization: 

#getting gradients d_R(mean)
def get_grads(grads):
    d_R_grads = [] 
    for i in range(len(grads)):
        d_R_grads.append(np.abs(np.mean(grads[i]['dR'])))
                    
    return d_R_grads


#grads 10
model = RNN(input_size=1, hidden_size=10, output_size=1)
data_generator_10 = generate_samples(batch_size=1024, seq_length= 10)
learner = Learner(model=model, loss_obj=loss_obj, data_generator=data_generator_10)
learner.train(iter_steps=5000, lr=1e-2, log_steps=200)
grads_10 = get_grads(learner.gradients)

#grads 25
model = RNN(input_size=1, hidden_size=10, output_size=1)
data_generator_25 = generate_samples(batch_size=1024, seq_length= 25)
learner = Learner(model=model, loss_obj=loss_obj, data_generator=data_generator_25)
learner.train(iter_steps=5000, lr=1e-2, log_steps=200)
grads_25 = get_grads(learner.gradients)
    
    



