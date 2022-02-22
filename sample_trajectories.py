"""Script for sampling trajectories"""

import numpy as np
import matplotlib.pyplot as plt

from motion import MotionSimulation, save_batch

def main():

    # Create simulation
    sim = MotionSimulation(boundary='square', time_step=0.1, std_norm=0.5)

    # Sample batch of trajectories
    batch_pos, batch_vel = sim.sample_batch(500, 20)

    # Save batch to file
    save_batch(batch_pos, batch_vel, 'data/test_batch.npz')

if __name__ == '__main__':
    main()