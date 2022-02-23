"""RNN for path integration"""

import numpy as np
import torch

class PathRNN(torch.nn.Module):
    
    def __init__(self, n_units):

        super(PathRNN, self).__init__()

        self.n_units = n_units

        # RNN Layer
        self.rnn = torch.nn.RNN(input_size=2, hidden_size=n_units, num_layers=1, nonlinearity='tanh', batch_first=True)

        # Output layer
        self.output = torch.nn.Linear(n_units, 2)
    
    def forward(self, vel):
        
        # Run RNN on velocity sequences to get hidden unit values
        u_vals, _ = self.rnn(vel)
        
        # Apply output weights to get estimated position
        pos_est = self.output(u_vals)
        
        return pos_est, u_vals
    