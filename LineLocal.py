from scipy.optimize import least_squares
import scipy
import numpy as np
import itertools
from rotmap import exp, hat, log
# from multiprocessing import Pool

class VP():

    def __init__(self, pt, ls):
        """
        pt: location of the GPR corresponding to the vertex in the images
        ls: scan trajectory directions, norm is 1
        """
        self.b1 = np.array([1, 0, 0]).reshape(3,1)
        self.b2 = np.array([0, 1, 0]).reshape(3,1)
        self.b3 = np.array([0, 0, 1]).reshape(3,1)
        self.pt = pt.reshape(-1, 3, 1)
        self.ls = ls

    def residuals(self, x):
        rot_vec = x[:3]
        R = exp(rot_vec.reshape(1, 3, 1))
        rho = x[3]
        """
        <ls, (m-pxl)xl> = 0
        """
        y = - rho * (self.ls.swapaxes(1,2) @ R @ self.b3) \
            - (self.b1.T @ R.swapaxes(1,2) @ self.pt @ self.ls.swapaxes(1,2) @ R @ self.b1) + \
            (self.pt.swapaxes(1,2) @ self.ls)
        y = y.reshape(-1)
        return y

    def loss(self, x):
        res = self.residuals(x)
        y = np.sum(res ** 2)/res.shape[0]
        return y

    def d_lsq(self, x):
        rot_vec = x[:3]
        R, d_R = exp(rot_vec.reshape(1, 3, 1), der=True)
        rho = x[3]
        d_R_y = - rho * (self.ls.swapaxes(1,2) @ d_R @ self.b3) 
        d_R_y = d_R_y - (self.b1.T @ d_R.swapaxes(2,3) @ self.pt @ self.ls.swapaxes(1,2) @ R @ self.b1)
        d_R_y = d_R_y - (self.b1.T @ R.swapaxes(1,2) @ self.pt @ self.ls.swapaxes(1,2) @ d_R @ self.b1)
        d_R_y = d_R_y.swapaxes(0,1).reshape(-1,3)
        d_rho_y = - (self.ls.swapaxes(1,2) @ R @ self.b3).reshape(-1,1)
        return np.hstack([d_R_y, d_rho_y])

    def estimate_lsq(self, x0 = np.zeros(4,)):
        res = least_squares(self.residuals, x0, jac = self.d_lsq, method = "lm")
        return res.x

class PDiff():

    def __init__(self, pt, d):
        """
        pt: location of the measurement points
        d: distance between the point and corresponding line
        """
        self.b1 = np.array([1, 0, 0]).reshape(3,1)
        self.B1 = hat(self.b1)
        self.b2 = np.array([0,1,0]).reshape(3,1)
        self.pt = pt.reshape(-1, 3, 1)
        self.d = d
        self.comb()

    def comb(self):
        self.pt1, self.pt2 = np.empty((0,3,1)), np.empty((0,3,1))
        self.d1, self.d2 = np.empty((0)), np.empty((0))
        for i in itertools.combinations(range(self.pt.shape[0]), 2):
            self.pt1 = np.concatenate([self.pt1, self.pt[None, i[0], :, :]])
            self.pt2 = np.concatenate([self.pt2, self.pt[None, i[1], :, :]])
            self.d1 =np.concatenate([self.d1, self.d[None, i[0]]]) 
            self.d2 =np.concatenate([self.d2, self.d[None, i[1]]])
            self.dd = self.d1-self.d2

    def residuals(self, x):
        rot_vec = - x[:3]
        R = exp(rot_vec.reshape(1, 3, 1))
        rho = x[3]
        y_est = np.linalg.norm(self.B1 @ R @ self.pt1 + rho * self.b2, axis=(1,2)) -\
                np.linalg.norm(self.B1 @ R @ self.pt2 + rho * self.b2, axis=(1,2))
        y = y_est - self.dd
        return y

    def loss(self, x):
        res = self.residuals(x)
        y = np.sum(res ** 2)/res.shape[0]
        return y

    def d_lsq(self, x):
        rot_vec = -x[:3]
        R, d_R = exp(rot_vec.reshape(1, 3, 1), der=True)
        rho = x[3]
        y1 = self.B1 @ R @ self.pt1 + rho * self.b2
        d_R_y1 = self.B1 @ d_R @ self.pt1
        d_R_y1 = np.sum(y1 * d_R_y1, axis=-2) / np.linalg.norm(y1, axis=-2)
        d_R_y1 = d_R_y1.squeeze(-1).T
        y2 = self.B1 @ R @ self.pt2 + rho * self.b2
        d_R_y2 = self.B1 @ d_R @ self.pt2
        d_R_y2 = np.sum(y2 * d_R_y2, axis=-2) / np.linalg.norm(y2, axis=-2)
        d_R_y2 = d_R_y2.squeeze(-1).T
        d_R_y = - d_R_y1 + d_R_y2
        d_rho_y = self.b2
        d_rho_y1 = np.sum(y1 * d_rho_y, axis=-2) / np.linalg.norm(y1, axis=-2)
        d_rho_y2 = np.sum(y2 * d_rho_y, axis=-2) / np.linalg.norm(y2, axis=-2)
        d_rho_y = d_rho_y1 - d_rho_y2
        return np.hstack([d_R_y, d_rho_y])

    def estimate_lsq(self, x0 = None):
        
        if x0 is None:
            x0 = np.array([0,0,0,0])

        res = least_squares(self.residuals, x0, jac = self.d_lsq, method = "lm")
        return res.x

class PDiff_tdoa():

    def __init__(self, pt1, pt2, dd):
        """
        pt1: locations of measurement point set 1
        pt2: locations of measurement point set 2
        dd: difference of distance between the point and corresponding line
        """
        self.b1 = np.array([1, 0, 0]).reshape(3,1)
        self.B1 = hat(self.b1)
        self.b2 = np.array([0,1,0]).reshape(3,1)
        self.pt1 = pt1.reshape(-1, 3, 1)
        self.pt2 = pt2.reshape(-1, 3, 1)
        self.dd = dd

    def residuals(self, x):
        rot_vec = - x[:3]
        R = exp(rot_vec.reshape(1, 3, 1))
        rho = x[3]
        y_est = np.linalg.norm(self.B1 @ R @ self.pt1 + rho * self.b2, axis=(1,2)) -\
                np.linalg.norm(self.B1 @ R @ self.pt2 + rho * self.b2, axis=(1,2))
        y = y_est - self.dd
        return y

    def loss(self, x):
        res = self.residuals(x)
        y = np.sum(res ** 2)/res.shape[0]
        return y

    def d_lsq(self, x):
        rot_vec = -x[:3]
        R, d_R = exp(rot_vec.reshape(1, 3, 1), der=True)
        rho = x[3]
        y1 = self.B1 @ R @ self.pt1 + rho * self.b2
        d_R_y1 = self.B1 @ d_R @ self.pt1
        d_R_y1 = np.sum(y1 * d_R_y1, axis=-2) / np.linalg.norm(y1, axis=-2)
        d_R_y1 = d_R_y1.squeeze(-1).T
        y2 = self.B1 @ R @ self.pt2 + rho * self.b2
        d_R_y2 = self.B1 @ d_R @ self.pt2
        d_R_y2 = np.sum(y2 * d_R_y2, axis=-2) / np.linalg.norm(y2, axis=-2)
        d_R_y2 = d_R_y2.squeeze(-1).T
        d_R_y = - d_R_y1 + d_R_y2
        d_rho_y = self.b2
        d_rho_y1 = np.sum(y1 * d_rho_y, axis=-2) / np.linalg.norm(y1, axis=-2)
        d_rho_y2 = np.sum(y2 * d_rho_y, axis=-2) / np.linalg.norm(y2, axis=-2)
        d_rho_y = d_rho_y1 - d_rho_y2
        return np.hstack([d_R_y, d_rho_y])

    def estimate_lsq(self, x0 = np.zeros(4,)):
        res = least_squares(self.residuals, x0, jac = self.d_lsq, method = "lm")
        return res.x



class combPV_admm():
    
    def __init__(self, pt, d, ver, ls):
        """
        pt: location of the measurement points
        d: distance between the point and corresponding line
        ver: locations of the GPR corresponding to the vertex in the images
        ls: scan trajectory directions, norm is 1
        """
        self.b1 = np.array([1, 0, 0]).reshape(3,1)
        self.B1 = hat(self.b1)
        self.b2 = np.array([0,1,0]).reshape(3,1)
        self.b3 = np.array([0, 0, 1]).reshape(3,1)
        self.pt = pt.reshape(-1, 3, 1)
        self.d = d
        self.comb()
        self.ver = ver.reshape(-1, 3, 1)
        self.ls = ls

    def comb(self):
        self.pt1, self.pt2 = np.empty((0,3,1)), np.empty((0,3,1))
        self.d1, self.d2 = np.empty((0)), np.empty((0))
        for i in itertools.combinations(range(self.pt.shape[0]), 2):
            self.pt1 = np.concatenate([self.pt1, self.pt[None, i[0], :, :]])
            self.pt2 = np.concatenate([self.pt2, self.pt[None, i[1], :, :]])
            self.d1 =np.concatenate([self.d1, self.d[None, i[0]]]) 
            self.d2 =np.concatenate([self.d2, self.d[None, i[1]]])
    
    def PDiff_prox(self, x, v, u, a):
        rot_vec = - x[:3]
        R = exp(rot_vec.reshape(1, 3, 1))
        rho = x[3]
        y_est = np.linalg.norm(self.B1 @ R @ self.pt1 + rho * self.b2, axis=(1,2)) -\
                np.linalg.norm(self.B1 @ R @ self.pt2 + rho * self.b2, axis=(1,2))
        y = y_est - (self.d1 - self.d2)
        y = y + a/2 * (x - v + u).T @ (x - v + u)
        return y

    def d_PDiff_prox(self, x, v, u, a):
        rot_vec = -x[:3]
        R, d_R = exp(rot_vec.reshape(1, 3, 1), der=True)
        rho = x[3]
        y1 = self.B1 @ R @ self.pt1 + rho * self.b2
        d_R_y1 = self.B1 @ d_R @ self.pt1
        d_R_y1 = np.sum(y1 * d_R_y1, axis=-2) / np.linalg.norm(y1, axis=-2)
        d_R_y1 = d_R_y1.squeeze(-1).T
        y2 = self.B1 @ R @ self.pt2 + rho * self.b2
        d_R_y2 = self.B1 @ d_R @ self.pt2
        d_R_y2 = np.sum(y2 * d_R_y2, axis=-2) / np.linalg.norm(y2, axis=-2)
        d_R_y2 = d_R_y2.squeeze(-1).T
        d_R_y = - d_R_y1 + d_R_y2
        d_rho_y = self.b2
        d_rho_y1 = np.sum(y1 * d_rho_y, axis=-2) / np.linalg.norm(y1, axis=-2)
        d_rho_y2 = np.sum(y2 * d_rho_y, axis=-2) / np.linalg.norm(y2, axis=-2)
        d_rho_y = d_rho_y1 - d_rho_y2
        der = np.hstack([d_R_y, d_rho_y]) + (a * (x - v + u))[None,:]
        return der

    def VP_prox(self, x, v, u, a):
        rot_vec = x[:3]
        R = exp(rot_vec.reshape(1, 3, 1))
        rho = x[3]
        """
        <ls, (m-pxl)xl> = 0
        """
        y = - rho * (self.ls.swapaxes(1,2) @ R @ self.b3) \
            - (self.b1.T @ R.swapaxes(1,2) @ self.ver @ self.ls.swapaxes(1,2) @ R @ self.b1) + \
            (self.ver.swapaxes(1,2) @ self.ls)
        y = y.reshape(-1)
        y = y + a/2 * (v - x + u).T @ (v - x + u)
        return y

    def d_VP_prox(self, x, v, u, a):
        rot_vec = x[:3]
        R, d_R = exp(rot_vec.reshape(1, 3, 1), der=True)
        rho = x[3]
        d_R_y = - rho * (self.ls.swapaxes(1,2) @ d_R @ self.b3) 
        d_R_y = d_R_y - (self.b1.T @ d_R.swapaxes(2,3) @ self.ver@ self.ls.swapaxes(1,2) @ R @ self.b1)
        d_R_y = d_R_y - (self.b1.T @ R.swapaxes(1,2) @ self.ver @ self.ls.swapaxes(1,2) @ d_R @ self.b1)
        d_R_y = d_R_y.swapaxes(0,1).reshape(-1,3)
        d_rho_y = - (self.ls.swapaxes(1,2) @ R @ self.b3).reshape(-1,1)
        der = np.hstack([d_R_y, d_rho_y]) - (a * (v - x + u))[None,:]
        return der

    def loss(self, x, a=0.5):
        v = x
        u = np.zeros(4,)
        res1 = self.VP_prox(x, v, u, a)
        res1 = np.sum(res1 ** 2)/res1.shape[0]
        res2 = self.PDiff_prox(x, v, u, a)
        res2 = np.sum(res2 ** 2)/res2.shape[0]
        y = res1 + res2
        return y

    def admm(self, a = 0.5, x0 = np.zeros(4,), u = np.zeros(4,), v0 = np.zeros(4,), max_iter = 50):
        results = []
        x = x0
        v = v0
        for k in range(max_iter):
            x_res = least_squares(self.PDiff_prox, x, jac = self.d_PDiff_prox, args = (v, u, a), method = "lm")
            x = x_res.x
            v_res = least_squares(self.VP_prox, v, jac = self.d_VP_prox, args = (x, u, a), method = "lm")
            v = v_res.x
            Rx_rot = exp(x[:3].reshape(1, 3, 1)).reshape(3, 3)
            Rv_rot = exp(v[:3].reshape(1, 3, 1)).reshape(3, 3)
            x = np.concatenate([log(Rx_rot), x[3].reshape(-1)])
            v = np.concatenate([log(Rv_rot), v[3].reshape(-1)])
            u = u + x - v
            Ru_rot = exp(u[:3].reshape(1, 3, 1)).reshape(3, 3)
            u = np.concatenate([log(Ru_rot), u[3].reshape(-1)])

            #Termination Stop
            results.append((x + v)/2)
            if k>1:
                if np.linalg.norm(results[-1] - results[-2]) < 1e-4:
                    return results[-1]
        return results[-1]        

class combPV_tdoa_admm():
    
    def __init__(self, pt1, pt2, dd, ver, ls):
        """
        pt1: locations of measurement point set 1
        pt2: locations of measurement point set 2
        dd: difference of distance between the point and corresponding line
        ver: locations of the GPR corresponding to the vertex in the images
        ls: scan trajectory directions, norm is 1
        """
        self.b1 = np.array([1, 0, 0]).reshape(3,1)
        self.B1 = hat(self.b1)
        self.b2 = np.array([0,1,0]).reshape(3,1)
        self.b3 = np.array([0, 0, 1]).reshape(3,1)
        self.pt1 = pt1.reshape(-1, 3, 1)
        self.pt2 = pt2.reshape(-1, 3, 1)
        self.dd = dd
        self.ver = ver.reshape(-1, 3, 1)
        self.ls = ls
    
    def PDiff_prox(self, x, v, u, a):
        rot_vec = - x[:3]
        R = exp(rot_vec.reshape(1, 3, 1))
        rho = x[3]
        y_est = np.linalg.norm(self.B1 @ R @ self.pt1 + rho * self.b2, axis=(1,2)) -\
                np.linalg.norm(self.B1 @ R @ self.pt2 + rho * self.b2, axis=(1,2))
        y = y_est - self.dd
        y = y + a/2 * (x - v + u).T @ (x - v + u)
        return y

    def d_PDiff_prox(self, x, v, u, a):
        rot_vec = -x[:3]
        R, d_R = exp(rot_vec.reshape(1, 3, 1), der=True)
        rho = x[3]
        y1 = self.B1 @ R @ self.pt1 + rho * self.b2
        d_R_y1 = self.B1 @ d_R @ self.pt1
        d_R_y1 = np.sum(y1 * d_R_y1, axis=-2) / np.linalg.norm(y1, axis=-2)
        d_R_y1 = d_R_y1.squeeze(-1).T
        y2 = self.B1 @ R @ self.pt2 + rho * self.b2
        d_R_y2 = self.B1 @ d_R @ self.pt2
        d_R_y2 = np.sum(y2 * d_R_y2, axis=-2) / np.linalg.norm(y2, axis=-2)
        d_R_y2 = d_R_y2.squeeze(-1).T
        d_R_y = - d_R_y1 + d_R_y2
        d_rho_y = self.b2
        d_rho_y1 = np.sum(y1 * d_rho_y, axis=-2) / np.linalg.norm(y1, axis=-2)
        d_rho_y2 = np.sum(y2 * d_rho_y, axis=-2) / np.linalg.norm(y2, axis=-2)
        d_rho_y = d_rho_y1 - d_rho_y2
        der = np.hstack([d_R_y, d_rho_y]) + (a * (x - v + u))[None,:]
        return der

    def VP_prox(self, x, v, u, a):
        rot_vec = x[:3]
        R = exp(rot_vec.reshape(1, 3, 1))
        rho = x[3]
        """
        <ls, (m-pxl)xl> = 0
        """
        y = - rho * (self.ls.swapaxes(1,2) @ R @ self.b3) \
            - (self.b1.T @ R.swapaxes(1,2) @ self.ver @ self.ls.swapaxes(1,2) @ R @ self.b1) + \
            (self.ver.swapaxes(1,2) @ self.ls)
        y = y.reshape(-1)
        y = y + a/2 * (v - x + u).T @ (v - x + u)
        return y

    def d_VP_prox(self, x, v, u, a):
        rot_vec = x[:3]
        R, d_R = exp(rot_vec.reshape(1, 3, 1), der=True)
        rho = x[3]
        d_R_y = - rho * (self.ls.swapaxes(1,2) @ d_R @ self.b3) 
        d_R_y = d_R_y - (self.b1.T @ d_R.swapaxes(2,3) @ self.ver@ self.ls.swapaxes(1,2) @ R @ self.b1)
        d_R_y = d_R_y - (self.b1.T @ R.swapaxes(1,2) @ self.ver @ self.ls.swapaxes(1,2) @ d_R @ self.b1)
        d_R_y = d_R_y.swapaxes(0,1).reshape(-1,3)
        d_rho_y = - (self.ls.swapaxes(1,2) @ R @ self.b3).reshape(-1,1)
        der = np.hstack([d_R_y, d_rho_y]) - (a * (v - x + u))[None,:]
        return der

    def loss(self, x, a=0.5):
        v = x
        u = np.zeros(4,)
        res1 = self.VP_prox(x, v, u, a)
        res1 = np.sum(res1 ** 2)/res1.shape[0]
        res2 = self.PDiff_prox(x, v, u, a)
        res2 = np.sum(res2 ** 2)/res2.shape[0]
        y = res1 + res2
        return y

    def admm(self, a = 0.5, x0 = np.zeros(4,), u = np.zeros(4,), v0 = np.zeros(4,), max_iter = 50):
        results = []
        x = x0
        v = v0
        for k in range(max_iter):
            x_res = least_squares(self.PDiff_prox, x, jac = self.d_PDiff_prox, args = (v, u, a), method = "lm")
            x = x_res.x
            v_res = least_squares(self.VP_prox, v, jac = self.d_VP_prox, args = (x, u, a), method = "lm")
            v = v_res.x
            Rx_rot = exp(x[:3].reshape(1, 3, 1)).reshape(3, 3)
            Rv_rot = exp(v[:3].reshape(1, 3, 1)).reshape(3, 3)
            x = np.concatenate([log(Rx_rot), x[3].reshape(-1)])
            v = np.concatenate([log(Rv_rot), v[3].reshape(-1)])
            u = u + x - v
            Ru_rot = exp(u[:3].reshape(1, 3, 1)).reshape(3, 3)
            u = np.concatenate([log(Ru_rot), u[3].reshape(-1)])

            #Termination Stop
            results.append((x + v)/2)
            if k>1:
                if np.linalg.norm(results[-1] - results[-2]) < 1e-4:
                    return results[-1]
        return results[-1]   

class PDist():
    
    def __init__(self, pt, d, r):
        """
        pt: location of the measurement points
        d: distance between the point and corresponding line
        r: radius of the pipe
        """
        self.b1 = np.array([1, 0, 0]).reshape(3,1)
        self.B1 = hat(self.b1)
        self.b2 = np.array([0, 1, 0]).reshape(3,1)
        self.pt = pt.reshape(-1, 3, 1)
        self.d = d
        self.r = r

    def residuals(self, x, r):
        rot_vec = -x[:3]
        R = exp(rot_vec.reshape(1, 3, 1))
        rho = x[3]
        y_est = np.linalg.norm(self.B1 @ R @ self.pt + rho * self.b2, axis=(1,2)) - r
        y = y_est - self.d 
        return y

    def loss(self, x, r):
        res = self.residuals(x, r)
        y = np.sum(res ** 2)/res.shape[0]
        return y

    def d_lsq(self, x, r):
        rot_vec = -x[:3]
        R, d_R = exp(rot_vec.reshape(1, 3, 1), der=True)
        rho = x[3]
        y = self.B1 @ R @ self.pt + rho * self.b2
        d_R_y = self.B1 @ d_R @ self.pt
        d_R_y = np.sum(y * d_R_y, axis=-2) / np.linalg.norm(y, axis=-2)
        d_R_y = -d_R_y.squeeze(-1).T
        d_rho_y = self.b2
        d_rho_y = np.sum(y * d_rho_y, axis=-2) / np.linalg.norm(y, axis=-2)
        return np.hstack([d_R_y, d_rho_y])

    def estimate_lsq(self, x0 = np.zeros(4)):
        res = least_squares(self.residuals, x0, self.d_lsq, args = (self.r,), method = "lm")
        return res.x

    
#bad performance, not used
class combPV_concat():
    
    def __init__(self, pt, d, ver, ls):
        """
        pt: location of the measurement points
        d: distance between the point and corresponding line
        """
        self.b1 = np.array([1, 0, 0]).reshape(3,1)
        self.B1 = hat(self.b1)
        self.b2 = np.array([0,1,0]).reshape(3,1)
        self.b3 = np.array([0, 0, 1]).reshape(3,1)
        self.pt = pt.reshape(-1, 3, 1)
        self.d = d
        self.comb()
        self.ver = ver.reshape(-1, 3, 1)
        self.ls = ls

    def comb(self):
        self.pt1, self.pt2 = np.empty((0,3,1)), np.empty((0,3,1))
        self.d1, self.d2 = np.empty((0)), np.empty((0))
        for i in itertools.combinations(range(self.pt.shape[0]), 2):
            self.pt1 = np.concatenate([self.pt1, self.pt[None, i[0], :, :]])
            self.pt2 = np.concatenate([self.pt2, self.pt[None, i[1], :, :]])
            self.d1 =np.concatenate([self.d1, self.d[None, i[0]]]) 
            self.d2 =np.concatenate([self.d2, self.d[None, i[1]]])

    def residuals(self, x, weight):
        
        rot_vec = - x[:3]
        R = exp(rot_vec.reshape(1, 3, 1))
        rho = x[3]
        """
        Point-Distance Difference equations
        """
        y_est = np.linalg.norm(self.B1 @ R @ self.pt1 + rho * self.b2, axis=(1,2)) -\
                np.linalg.norm(self.B1 @ R @ self.pt2 + rho * self.b2, axis=(1,2))
        y1 = y_est - (self.d1 - self.d2)        
        """
        Line perpendicular equations
        """
        y2 = - rho * (self.ls.swapaxes(1,2) @ R @ self.b3) \
            - (self.b1.T @ R.swapaxes(1,2) @ self.ver @ self.ls.swapaxes(1,2) @ R @ self.b1) + \
            (self.ver.swapaxes(1,2) @ self.ls)
        y2 = y2.reshape(-1)
        return weight * np.concatenate([y1, y2])

    def loss(self, x):
        res = self.residuals(x)
        y = np.sum(res ** 2)/res.shape[0]
        return y

    def d_lsq(self, x, weight):
        rot_vec = -x[:3]
        R, d_R = exp(rot_vec.reshape(1, 3, 1), der=True)
        rho = x[3]
        """
        Point-Distance Difference equations
        """
        y1 = self.B1 @ R @ self.pt1 + rho * self.b2
        d_R_y1 = self.B1 @ d_R @ self.pt1
        d_R_y1 = np.sum(y1 * d_R_y1, axis=-2) / np.linalg.norm(y1, axis=-2)
        d_R_y1 = d_R_y1.squeeze(-1).T
        y2 = self.B1 @ R @ self.pt2 + rho * self.b2
        d_R_y2 = self.B1 @ d_R @ self.pt2
        d_R_y2 = np.sum(y2 * d_R_y2, axis=-2) / np.linalg.norm(y2, axis=-2)
        d_R_y2 = d_R_y2.squeeze(-1).T
        d_R_y = - d_R_y1 + d_R_y2
        d_rho_y = self.b2
        d_rho_y1 = np.sum(y1 * d_rho_y, axis=-2) / np.linalg.norm(y1, axis=-2)
        d_rho_y2 = np.sum(y2 * d_rho_y, axis=-2) / np.linalg.norm(y2, axis=-2)
        d_rho_y = d_rho_y1 - d_rho_y2
        """
        Line perpendicular equations
        """
        d_R_y3 = - rho * (self.ls.swapaxes(1,2) @ d_R @ self.b3) 
        d_R_y3 = d_R_y3 - (self.b1.T @ d_R.swapaxes(2,3) @ self.pt @ self.ls.swapaxes(1,2) @ R @ self.b1)
        d_R_y3 = d_R_y3 - (self.b1.T @ R.swapaxes(1,2) @ self.pt @ self.ls.swapaxes(1,2) @ d_R @ self.b1)
        d_R_y3 = d_R_y3.swapaxes(0,1).reshape(-1,3)
        d_rho_y3 = - (self.ls.swapaxes(1,2) @ R @ self.b3).reshape(-1,1)
        """
        Combine
        """
        d_R_y = weight[:, None] * np.concatenate([d_R_y, d_R_y3], axis = 0)
        d_rho_y = weight[:, None] * np.concatenate([d_rho_y, d_rho_y3], axis = 0)
        return np.hstack([d_R_y, d_rho_y])

    def estimate_lsq(self, x0 = None, weight = None):
        
        if x0 is not None:
            pass
        else:
            x0 = np.array([0,0,0,0])

        if weight is not None:
            pass
        else:
            weight = np.ones((self.pt1.shape[0] + self.ver.shape[0]))

        res = least_squares(self.residuals, x0, jac = self.d_lsq, args = (weight, ))
        return res.x
