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

        # Initialize RNN weights
        for name, param in self.rnn.named_parameters():
            if name == 'bias_ih_l0':
                 torch.nn.init.zeros_(param)
            elif name == 'bias_hh_l0':
                 torch.nn.init.zeros_(param)
            elif name == 'weight_ih_l0':
                 torch.nn.init.normal_(param, mean=0.0, std=np.sqrt(1/2))
            elif name == 'weight_hh_l0':
                 torch.nn.init.orthogonal_(param)

        # Initialize output weights
        torch.nn.init.zeros_(self.output.weight.data)
        torch.nn.init.zeros_(self.output.bias.data)
    
    def forward(self, vel):
        
        # Run RNN on velocity sequences to get hidden unit values
        u_vals, _ = self.rnn(vel)
        
        # Apply output weights to get estimated position
        pos_est = self.output(u_vals)
        
        return pos_est, u_vals
    