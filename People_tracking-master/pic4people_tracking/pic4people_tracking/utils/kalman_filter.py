import numpy as np
import scipy.linalg
from copy import deepcopy


class KalmanFilter():    
    def __init__(self, dim_x, dim_z, dim_u=0):
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')

        self.dim_x = dim_x
        self.dim_z = dim_z

        self.x = np.zeros((dim_x, 1))        # state
        self.P = np.eye(dim_x)               # uncertainty covariance
        self.Q = np.eye(dim_x)               # process uncertainty
        self.B = None                     # control transition matrix
        self.F = np.eye(dim_x)               # state transition matrix
        self.H = np.zeros((dim_z, dim_x))    # Measurement function
        self.R = np.eye(dim_z)               # state uncertainty
        self.M = np.zeros((dim_z, dim_z)) # process-measurement cross correlation
        self.z = np.array([[None]*self.dim_z]).T

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros((dim_x, dim_z)) # kalman gain
        self.y = np.zeros((dim_z, 1))
        self.normalized_residual = 0.0
        self.S = np.zeros((dim_z, dim_z)) # system uncertainty

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        self.inv = np.linalg.inv


    def predict(self):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations.
        """

        Q = self.Q
        F = self.F

        self.x = np.dot(F, self.x)

        # P = FPF' + Q
        self.P = np.linalg.multi_dot((F, self.P, F.T)) + Q
        
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()


    def update(self, z):
        """
        Add a new measurement (z) to the Kalman filter.

        """

        if z is None:
            self.z = np.array([[None]*self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = np.zeros((self.dim_z, 1))
            return

        R = self.R
        H = self.H

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - np.dot(H, self.x)

        z_norm = np.linalg.norm(z)
        if z_norm == 0:
            z_norm = 1e-6  # Avoid division by zero
        self.normalized_residual = np.linalg.norm(self.y) / z_norm

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = np.linalg.multi_dot((H, self.P, H.T)) + R

        chol_factor, lower = scipy.linalg.cho_factor(self.S, lower=True, check_finite=False)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(self.P, H.T).T, check_finite=False).T
        
        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + np.dot(self.K, self.y)

        # P = P - KSK'
        self.P = self.P - np.linalg.multi_dot((self.K, self.S, self.K.T))
            

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
