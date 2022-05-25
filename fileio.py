"""Functions for saving/loading training runs to/from files."""

import os
import json

import torch

from model import PathRNN


PARAMS_FNAME = 'params.json'
RUNINFO_FNAME = 'runinfo.json'
CHECKPOINT_FNAME_FMT = 'model_{:02d}.pt'


def save_params(out_dir: str, params: dict):
    """Save parameters to JSON file in run directory."""

    fpath = os.path.join(out_dir, PARAMS_FNAME)
    with open(fpath, 'w') as f:
        json.dump(params, f, indent=4)

def load_params(out_dir: str) -> dict:
    """Load parameters rom JSON file in run directory."""

    fpath = os.path.join(out_dir, PARAMS_FNAME)
    with open(fpath, 'r') as f:
        return json.load(f)

def save_checkpoint(out_dir: str, model: PathRNN, epoch: int):
    """Save state of model to file with epoch number in filename."""

    fpath = os.path.join(out_dir, CHECKPOINT_FNAME_FMT.format(epoch))
    torch.save(model.state_dict(), fpath)

def load_checkpoint(out_dir: str, epoch: int) -> PathRNN:
    """Load model saved as checkpoint for epoch."""

    # Load parameters for run
    params = load_params(out_dir)

    # Load state dict from model
    fpath = os.path.join(out_dir, CHECKPOINT_FNAME_FMT.format(epoch))
    state_dict = torch.load(fpath)

    # Create model with correct parameters and set loaded state
    model = PathRNN(**params['model'])
    model.load_state_dict(state_dict)
    model.eval()

    return model

def save_runinfo(out_dir: str, runinfo: dict):
    """Save run info to JSON file in output directory."""

    fpath = os.path.join(out_dir, RUNINFO_FNAME)
    with open(fpath, 'w') as f:
       json.dump(runinfo, f, indent=4)

def load_run(out_dir: str) -> tuple[dict, PathRNN]:
    """Load parameters and final checkpoint model from directory."""

    params = load_params(out_dir)
    model = load_checkpoint(out_dir, params['num_epochs'])

    return params, model