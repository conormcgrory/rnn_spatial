"""RNN for path integration"""

import numpy as np


N_INPUTS = 2
N_UNITS = 100
N_OUTPUTS = 2


class RNNParams:

    def __init__(self, w_in, w_rec, w_out, b):

        self.w_in = w_in
        self.w_rec = w_rec
        self.w_out = w_out
        self.b = b

    @classmethod
    def from_vec(cls, vec):
        pass

    def to_vec(self):
        pass

def get_init_params(rng):

    w_in = rng.normal(0, np.sqrt(1 / N_INPUTS), (N_UNITS, N_INPUTS))
    w_out = np.zeros((N_OUTPUTS, N_UNITS))
    b = np.zeros(N_UNITS)

    # TODO: Change this to orthogonal initialization from Saxe et al., 2014
    w_rec = rng.normal(0, np.sqrt(1 / N_UNITS), (N_UNITS, N_UNITS))

    return RNNParams(w_in, w_rec, w_out, b)

def run_batch(vel, params):

    n_pts = vel.shape[0]

    # Unit inputs
    x = np.full((n_pts, N_UNITS), np.nan)

    # Unit activity values
    u = np.full((n_pts, N_UNITS), np.nan)

    # Output values
    y = np.full((n_pts, 2), np.nan)

    # Initialize units to zero
    u[-1] = np.zeros(N_UNITS)

    for t in range(n_pts):

        # Update unit inputs
        x[t] = params.w_in @ vel[t] + params.w_rec @ u[t - 1]

        # Update unit activities
        u[t] = np.tanh(x[t])

        # Update outputs
        y[t] = params.w_out @ u[t]

    return y

def compute_batch_error(vel, pos, params):

    pos_est = run_batch(vel, params)
    sq_err = np.sum((pos_est - pos) ** 2, axis=1)

    return np.mean(sq_err)
