import numpy as np
from rotmap import exp, hat

def ToyDataGenerator(num_lines, num_pts_per_line, num_pts_sprk, gt, line_step, noise_level):
    b1 = np.array([1, 0, 0]).reshape(3,1)
    b2 = np.array([0, 1, 0]).reshape(3,1)
    b3 = np.array([0, 0, 1]).reshape(3,1)
    B1 = hat(b1)
    v_s = np.random.randn(num_lines, 3, 1)
    rho_s = abs(np.random.randn(num_lines, 1))
    R_s = exp(v_s)
    ls = R_s @ b1
    ms = np.multiply(rho_s[:, None], R_s) @ b2
    rho_gt = gt[3]
    r_gt = gt[4]
    v = gt[:3].reshape(3, 1)
    R_gt = exp(v)
    l_gt = R_gt @ b1
    m_gt = rho_gt * R_gt @ b2
    l_gt_stack = np.tile(l_gt, (num_lines, 1, 1))
    m_gt_stack = np.tile(m_gt, (num_lines, 1, 1))
    numerator1 = -np.cross(ms, np.cross(l_gt_stack, np.cross(ls, l_gt_stack, axis = 1), axis = 1), axis = 1)
    numerator2 = (m_gt_stack.swapaxes(1,2) @ np.cross(ls, l_gt_stack, axis = 1)).swapaxes(1,2) * ls
    denominator = np.linalg.norm(np.cross(l_gt_stack, ls, axis = 1), axis = (1,2)) ** 2
    vert = ((numerator1 + numerator2).reshape(-1,3)/denominator.reshape(-1,1)).reshape(-1,3,1)
    pts = []
    for i in range(vert.shape[0]):
        pos = np.random.randint(-num_pts_per_line+1, 0)
        for j in range(num_pts_per_line):
            pts.append(vert[i, :, :] + ls[i,:,:] * (pos + j) * line_step)
    pts_sprk = np.random.uniform(np.min(pts), np.max(pts), (num_pts_sprk, 3, 1))
    pts_all = np.concatenate([np.array(pts), pts_sprk], axis = 0)
    vert = vert + np.random.randn(vert.shape[0], 3, 1) * noise_level
    pts_all = pts_all + np.random.randn(pts_all.shape[0], 3, 1) * noise_level
    pt_rot = R_gt.T @ pts_all
    d_vec = -B1 @ pt_rot-rho_gt*b2
    d = np.linalg.norm(d_vec, axis=(1,2)) - r_gt + np.random.randn(pts_all.shape[0]) * noise_level
    return vert, pts_all, d, ls, np.array(pts), pts_sprk