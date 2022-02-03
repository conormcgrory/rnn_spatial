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

    def __init__(self):

        self.rng = np.random.default_rng()
        self.boundary = RectangleBoundary(0.0, 0.0, 2.0, 2.0)
        self.time_step = 0.01
        self.std = 1.0
        self.speed = 1.0

    def get_init_direction(self):

        return self.rng.random() * 2 * np.pi

    def get_next_direction(self, theta_prev):

        # Standard deviation of Brownian motion step
        std_brownian = np.sqrt(self.time_step) * self.std

        # Take Brownian motion step to get new direction (non-normalized)
        theta_nn = theta_prev + std_brownian * self.rng.standard_normal()

        # Make sure angle is between 0 and 2 * pi
        return np.mod(theta_nn, 2 * np.pi)

    def update_position(self, x, y, theta):
        
        x_next = x + self.speed * self.time_step * np.cos(theta)
        y_next = y + self.speed * self.time_step * np.sin(theta)

        return x_next, y_next


    def sample_trial(self, n_pts):

        x = np.full(n_pts, np.nan)
        y = np.full(n_pts, np.nan)
        theta = np.full(n_pts, np.nan)

        x[0] = 0.0
        y[0] = 0.0
        theta[0] = self.get_init_direction()

        x[1], y[1] = self.update_position(x[0], y[0], theta[0])
        if not self.boundary.contains(x[1], y[1]):
            raise ValueError('First step is outside boundary')

        for t in range(1, n_pts - 1):

            while True:

                theta[t] = self.get_next_direction(theta[t - 1])
                x[t + 1], y[t + 1] = self.update_position(x[t], y[t], theta[t])

                if self.boundary.contains(x[t], y[t]):
                    break

        return x, y, theta

    def plot_position(self, x, y, ax=None):

        if ax is None:
            ax = plt.gca()

        self.boundary.plot(ax)
        ax.plot(x, y)

    def plot_direction(self, theta, ax=None):

        if ax is None:
            ax = plt.gca()

        ax.plot(theta)
