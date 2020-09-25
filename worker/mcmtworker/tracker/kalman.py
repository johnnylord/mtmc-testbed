import numpy as np
import scipy.linalg


__all__ = [ "KalmanFilter" ]

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter:
    """Kalman filter for handling tracker spatial state in image space

    Here are the states kalman filter trying to maintain

        np.ndarray([ x, y, a, h, vx, vy, va, vh ])

    Target explanation:
        x: bounding box center postion along x axis in image space
        y: bounding box center postion along y axis in image space
        a: aspect ratio of bounding box width over bounding box height
        h: bounding box height
        v*: respective velocities of (x, y, a, h)

    The motion model is a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation.

    Attributes:
        _pred_mat (ndarray): a (8, 8) matrix for predicting the next state
        _project_mat (ndarray): a (4, 8) matrix for projecting state vector from
            state space to measurement space.
        _std_position (float): uncertainty of the (x, y, a, h)
        _std_velocity (float): uncertainty of the (vx, vy, va, vh)
    """
    def __init__(self):
        n_dim = 4
        delta_time = 1.

        # Dimension of prediction matrix: (8, 8)
        self._pred_mat = np.eye(2*n_dim, 2*n_dim)
        for i in range(n_dim):
            self._pred_mat[i, n_dim+i] = delta_time

        # Dimension of projection matrix: (4, 8)
        self._project_mat = np.eye(n_dim, 2*n_dim)

        # Uncertainty of the state
        self._std_position = 1. / 20    # For (x, y, a, h)
        self._std_velocity = 1. / 160   # For (vx, vy, va, vh)

    def initiate(self, measurement):
        """Initialize state of kalman filter

        Args:
            measurement (ndarray): a (4,) vector representing (x, y, a, h)

        Returns:
            mean (ndarray): intialized state vector of shape (8,)
            covariance (ndarray): initialized uncertainty matrix of shape (8x8)
        """
        # mean vector (state vector)
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel])

        # covariance matrix (uncertainty matrix)
        std = [
            2 * self._std_position * measurement[3],
            2 * self._std_position * measurement[3],
            1e-2,
            2 * self._std_position * measurement[3],
            10 * self._std_velocity * measurement[3],
            10 * self._std_velocity * measurement[3],
            1e-5,
            10 * self._std_velocity * measurement[3]]
        covariance = np.diag(np.square(std))

        return mean, covariance

    def predict(self, mean, covariance):
        """Predict the next state given the previous state (mean & covariance)

        Args:
            mean (ndarray): previous state vector of shape (8,)
            covariance (ndarray): previous uncertainty matrix of shape (8, 8)

        Returns:
            mean (ndarray): predicted state vector of shape (8,)
            covariance (ndarray): predicted uncertainty matrix of shape (8, 8)
        """
        # Noise for covariance matrix (noise from the "world", unknown factors)
        std_position = [
            self._std_position * mean[3],
            self._std_position * mean[3],
            1e-2,
            self._std_position * mean[3]]
        std_velocity = [
            self._std_velocity * mean[3],
            self._std_velocity * mean[3],
            1e-5,
            self._std_velocity * mean[3]]
        world_noise = np.diag(np.square(np.concatenate([std_position, std_velocity])))

        # Update mean vector and covariance matrix
        mean = np.dot(self._pred_mat, mean)
        covariance = np.linalg.multi_dot((
            self._pred_mat, covariance, self._pred_mat.T)) + world_noise

        return mean, covariance

    def _project(self, mean, covariance):
        """Project state vector and uncertainty matrix to the measurement space

        Args:
            mean (ndarray): predicted state vector of shape (8,)
            covariance (ndarray): predicted uncertainty matrix of shape (8, 8)

        Returns:
            mean (ndarray): projected state vector of shape (4,)
            covariance (ndarray): projected uncertainty matrix of shape (4, 4)
        """
        # Noise for projected covariance matrix
        std_position = [
            self._std_position * mean[3],
            self._std_position * mean[3],
            1e-1,
            self._std_position * mean[3]]
        project_noise = np.diag(np.square(std_position))

        # Projected mean and covariance matrix
        mean = np.dot(self._project_mat, mean)
        covariance = np.linalg.multi_dot((
            self._project_mat, covariance, self._project_mat.T)) + project_noise

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Refine the predicted state with the observed data

        Args:
            mean (ndarray): predicted state vector of shape (8,)
            covariacne (ndarray): predicted uncertainty matrix of shape (8, 8)
            measurement (ndarray): observed data of shape (4,)

        Returns:
            mean (ndarray): refined state vector of shape (8,)
            covariacne (ndarray): refined uncertainty matrix of shape (8, 8)
        """
        # Project mean & covariance so that they are in the same space as measurement
        project_mean, project_covariance = self._project(mean, covariance)

        # Calculate kalman gain
        chol_factor, lower = scipy.linalg.cho_factor(
                                    project_covariance,
                                    lower=True,
                                    check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
                                    (chol_factor, lower),
                                    np.dot(covariance, self._project_mat.T).T,
                                    check_finite=False).T

        # Update mean and covariance with measurement
        mean = mean + np.dot(measurement-project_mean, kalman_gain.T)
        covariance = covariance - np.linalg.multi_dot((
            kalman_gain, project_covariance, kalman_gain.T))

        return mean, covariance
