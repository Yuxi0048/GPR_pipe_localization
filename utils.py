import pandas as pd
import numpy as np
from rotmap import hat, exp, log

class Eval:
    def __init__(self, hypo, gt):
        self.hypo = hypo
        self.gt = gt

    def RA(hypo, gt):
        l_hypo = exp(hypo[:3].reshape(1,3,1))[:,:,0]
        l_gt = exp(gt[:3].reshape(1,3,1))[:,:,0]
        cos = abs(l_hypo @ l_gt.T)
        if cos > 1:
            cos = np.ones((1, 1))
        return np.arccos(cos)[0][0]

    def SD(hypo, gt):
        l_hypo = exp(hypo[:3].reshape(1,3,1))[:,:,0]
        l_gt = exp(gt[:3].reshape(1,3,1))[:,:,0]
        m_hypo = hypo[3] * exp(hypo[:3].reshape(1,3,1))[:,:,1]
        m_gt = gt[3] * exp(gt[:3].reshape(1,3,1))[:,:,1]
        if l_hypo @ l_gt.T > 1 - 1e-8:
            return np.linalg.norm(np.cross(l_gt.reshape(-1), (m_gt - m_hypo).reshape(-1)))/np.linalg.norm(l_gt)**2
        elif l_hypo @ l_gt.T < -1 + 1e-8:
            return np.linalg.norm(np.cross(l_gt.reshape(-1), (m_gt + m_hypo).reshape(-1)))/np.linalg.norm(l_gt)**2
        return (abs(l_hypo @ m_gt.T + l_gt @ m_hypo.T).reshape(-1)/np.linalg.norm(np.cross(l_hypo.reshape(-3), l_gt.reshape(-3))))[0]

    def RR(hypo, gt):
        return abs(hypo-gt[4])/gt[4]


class Report:
    
    def init_dict():
        data = {}
        return data

    def append_dict(hypo_3d_line, data, i, rad = None, gt = None, noise_level = None):
        data[i, 0] = noise_level
        data[i, 1] = gt
        data[i, 2] = hypo_3d_line
        data[i, 3] = rad
        if gt is not None:
            data[i, 4] = Eval.RA(hypo_3d_line, gt)
            data[i, 5] = Eval.SD(hypo_3d_line, gt)
            if rad is not None:
                data[i, 6] = Eval.RR(rad, gt)
        return data
    
    def OutExcel(data, fn):    
        s = pd.Series(data, index=pd.MultiIndex.from_tuples(data))
        df = s.unstack()
        df.columns = ["noise", "gt", "hypo", "rad", "RA","SD", "RR"]
        df.to_excel(fn)