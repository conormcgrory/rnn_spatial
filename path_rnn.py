"""RNN for path integration"""

import numpy as np
import torch

class PathRNN(torch.nn.Module):
    
    def __init__(self, n_units):

        super(PathRNN, self).__init__()

        self.n_units = n_units

        # RNN Layer
        self.rnn = torch.nn.RNN(input_size=2, hidden_size=n_units, num_layers=1, nonlinearity='tanh', batch_first=True, bias=True)

        # Output layer
        self.output = torch.nn.Linear(n_units, 2, bias=False)

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

    def init_hidden(self, batch_size):

        return torch.zeros(1, batch_size, self.n_units)
    
    def forward(self, vel):

        # Initializing hidden state for first input
        batch_size = vel.size(0)
        u_init = self.init_hidden(batch_size)
        
        # Run RNN on velocity sequences to get hidden unit values
        u_vals, _ = self.rnn(vel, u_init)
        
        # Apply output weights to get estimated position
        pos_est = self.output(u_vals)
        
        return pos_est, u_vals
    