"""Motion and environment simulation"""

import abc

import numpy as np
import matplotlib.pyplot as plt


class Boundary(abc.ABC):

    @abc.abstractmethod
    def contains(x, y):
        pass

    @abc.abstractmethod
    def plot(ax):
        pass


class RectangleBoundary:

    def __init__(self, ctr_x, ctr_y, width, height):

        self.x_min = ctr_x - 0.5 * width
        self.x_max = ctr_x + 0.5 * width
        self.y_min = ctr_y - 0.5 * height 
        self.y_max = ctr_y + 0.5 * height

    def contains(self, x, y):

        return ((x > self.x_min) 
                and (x < self.x_max) 
                and (y > self.y_min) 
                and (y < self.y_max))

    def plot(self, ax=None):

        if ax is None:
            ax = plt.gca()

        x_vals = np.array([
            self.x_min, self.x_max, self.x_max, self.x_min, self.x_min])
        y_vals = np.array([
            self.y_min, self.y_min, self.y_max, self.y_max, self.y_min])

        ax.plot(x_vals, y_vals, '--')


class MotionSimulation:

    def __init__(self, boundary='square', time_step=0.01, std_norm=1.0, rng=None):

        # Boundary for spatial environment
        if boundary == 'square':
            self.boundary = RectangleBoundary(0.0, 0.0, 2.0, 2.0)
        else:
            raise ValueError(f'Boundary "{boundary}" not supported.')
            
        # Compute Brownian motion stddev from time step and normalized stddev
        self.time_step = time_step
        self.std_norm = std_norm
        self.std_brownian = np.sqrt(time_step) * std_norm
        
        # Speed distribution of animal
        self.speed_vals = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        self.p_speed = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        # Random number generator
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def smp_speed(self):
        return self.rng.choice(self.speed_vals, p=self.p_speed)

    def smp_init_direction(self):
        return self.rng.random() * 2 * np.pi

    def smp_direction_step(self):
        return self.std_brownian * self.rng.standard_normal()

    def smp_direction_collision(self):
        return self.rng.random() * 2 * np.pi

    def get_xstep(self, speed, theta):
        return speed * self.time_step * np.cos(theta)

    def get_ystep(self, speed, theta):
        return speed * self.time_step * np.sin(theta)

    def sample_trial(self, n_steps):

        # Position (cartesian coordinates)
        pos_x = np.full(n_steps, np.nan)
        pos_y = np.full(n_steps, np.nan)

        # Velocity (polar coordinates)
        speed = np.full(n_steps, np.nan)
        theta = np.full(n_steps, np.nan)

        # Initialize velocity
        speed[0] = self.smp_speed()
        theta[0] = self.smp_init_direction()

        # Initalize position
        pos_x[0] = self.get_xstep(speed[0], theta[0])
        pos_y[0] = self.get_ystep(speed[0], theta[0])

        # Check boundary condition
        if not self.boundary.contains(pos_x[0], pos_y[0]):
            raise ValueError('First step is outside boundary')

        for t in range(1, n_steps):

            # Update velocity
            speed[t] = self.smp_speed()
            theta[t] = theta[t - 1] + self.smp_direction_step()

            # Update position
            pos_x[t] = pos_x[t - 1] + self.get_xstep(speed[t], theta[t])
            pos_y[t] = pos_y[t - 1] + self.get_ystep(speed[t], theta[t])
 
            # If animal collides with wall, sample angle from uniform distribution
            while not self.boundary.contains(pos_x[t], pos_y[t]):

                # Resample direciton
                theta[t] = self.smp_direction_collision()

                # Update position
                pos_x[t] = pos_x[t - 1] + self.get_xstep(speed[t], theta[t])
                pos_y[t] = pos_y[t - 1] + self.get_ystep(speed[t], theta[t])


        # Normalize direction so that it lies in [0, 2pi]
        theta_norm = np.mod(theta, 2 * np.pi)

        pos = np.stack((pos_x, pos_y), axis=-1)
        vel = np.stack((speed, theta_norm), axis=-1)

        return pos, vel

    def plot_position(self, pos, ax=None):

        if ax is None:
            ax = plt.gca()

        # Add origin point to beginning of position sequence
        pos_x = np.concatenate(([0.0], pos[:, 0]))
        pos_y = np.concatenate(([0.0], pos[:, 1]))

        # Make sure x- and y-scales are the same
        ax.set_aspect('equal')

        # Plot boundary and position values
        self.boundary.plot(ax)
        ax.plot(pos_x, pos_y)

    def plot_direction(self, theta, ax=None):

        if ax is None:
            ax = plt.gca()

        ax.plot(theta)
