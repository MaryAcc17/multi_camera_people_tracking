import numpy as np
import scipy.linalg

class KalmanFilterPose(object):
    """
    Kalman filter for tracking yaw angle - orientation of each person.

    The 2-dimensional state space

        yaw, omega

    contains the yaw angle and its derivative.

    Object motion follows a constant velocity model. 

    """

    def __init__(self):
        ndim, dt = 1, 1./18

        # Create Kalman filter model matrices.
        self._motion_mat = np.array([[1., dt],
                                     [0., 1.]])

        self._update_mat = np.array([[1., 0.]])

    def initiate(self, measurement):

        mean_yaw = measurement
        mean_omega = 0.0
        mean = np.array([[mean_yaw], [mean_omega]])

        std = [1.0,
               1000.0]
        
        covariance = np.diag(std) # P0
        return mean, covariance

    def update_motion_mat(self, dt):
        self._motion_mat = np.array([[1, dt],
                                     [0, 1]])

    def predict(self, mean, covariance, dt):

        motion_cov = np.array([[0.02, 0.],
                               [0., 0.4]]) # Qk

        self.update_motion_mat(dt)
        mean = np.dot(self._motion_mat, mean)
        ### unwrap angle
        mean[0] = (mean[0] + np.pi) % (2*np.pi) - np.pi
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov # Pk

        return mean, covariance

    def project(self, mean, covariance):

        innovation_cov = np.array([[50.]]) # Rk

        mean = np.dot(self._update_mat, mean)
        ### unwrap angle
        mean = (mean + np.pi) % (2*np.pi) - np.pi
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov # predicted output, S (projected covariance matrix)

    def update(self, mean, covariance, measurement):

        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        ### unwrap angle
        innovation[0] = (innovation[0] + np.pi) % (2*np.pi) - np.pi
        new_mean = mean + np.dot(kalman_gain, innovation) #updated state
        ### unwrap angle
        new_mean[0] = (new_mean[0] + np.pi) % (2*np.pi) - np.pi

        new_covariance = covariance - np.linalg.multi_dot(( # Pk updated
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance
