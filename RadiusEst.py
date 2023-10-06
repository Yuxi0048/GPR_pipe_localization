import numpy as np
from rotmap import exp, hat

class WeightedAvg():

    def __init__(self, pt, d, line_params):
        """
        pt: location of the measurement points
        d: distance between the point and corresponding line
        line_params: line parameters (
            axis angle representation (theta1, theta2, theta3), 
            distance to origin (rho)
            )
        """
        self.b1 = np.array([1, 0, 0]).reshape(3,1)
        self.B1 = hat(self.b1)
        self.b2 = np.array([0,1,0]).reshape(3,1)
        self.pt = pt.reshape(-1, 3, 1)
        self.d = d
        self.line_params = line_params

    def estimate(self, weight = None):
        if weight is not None:
            pass
        else:
            weight = np.ones(self.pt.shape[0])
        rot_vec = - self.line_params[:3]
        R = exp(rot_vec.reshape(1, 3, 1))
        rho = self.line_params[3]
        r_est = - self.d + np.linalg.norm(self.B1 @ R @ self.pt + rho * self.b2, axis=(1,2))
        y = np.sum(weight * r_est)/np.sum(weight)
        return y