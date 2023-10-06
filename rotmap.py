import numpy as np

def hat(v):
    """
    vecotrized version of the hat function, creating for a vector its skew symmetric matrix.

    Args:
        v (np.array<float>(..., 3, 1)): The input vector.

    Returns:
        (np.array<float>(..., 3, 3)): The output skew symmetric matrix.

    """
    E1 = np.array([[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    E2 = np.array([[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]])
    E3 = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]])
    
    return v[..., 0:1, :] * E1 + v[..., 1:2, :] * E2 + v[..., 2:3, :] * E3


def exp(v, der=False):
    """
    Vectorized version of the exponential map.

    Args:
        v (np.array<float>(..., 3, 1)): The input axis-angle vector.
        der (bool, optional): Wether to output the derivative as well. Defaults to False.

    Returns:
        R (np.array<float>(..., 3, 3)): The corresponding rotation matrix.
        [dR (np.array<float>(3, ..., 3, 3)): The derivative of each rotation matrix.
                                            The matrix dR[i, ..., :, :] corresponds to
                                            the derivative d R[..., :, :] / d v[..., i, :],
                                            so the derivative of the rotation R gained 
                                            through the axis-angle vector v with respect
                                            to v_i. Note that this is not a Jacobian of
                                            any form but a vectorized version of derivatives.]

    """
    n = np.linalg.norm(v, axis=-2, keepdims=True)
    H = hat(v)
    
    with np.errstate(all='ignore'):
        R = np.identity(3) + (np.sin(n) / n) * H + ((1 - np.cos(n)) / n**2) * (H @ H)
    R = np.where(n == 0, np.identity(3), R)
    
    if der:
        sh = (3,) + tuple(1 for _ in range(v.ndim - 2)) + (3, 1)
        dR = np.swapaxes(np.expand_dims(v, axis=0), 0, -2) * H
        dR = dR + hat(np.cross(v, ((np.identity(3) - R) @ np.identity(3).reshape(sh)), axis=-2))
        dR = dR @ R
        
        n = n**2  # redifinition
        with np.errstate(all='ignore'):
            dR = dR / n
        dR = np.where(n == 0, hat(np.identity(3).reshape(sh)), dR)
            
        return R, dR
    
    else:
        return R


def log(R):
    """
    log map
    from: https://github.com/nurlanov-zh/so3_log_map/blob/main/so3_log_map_analysis.ipynb
    """   
    trR = R[0, 0] + R[1, 1] + R[2, 2]
    cos_theta = max(min(0.5 * (trR - 1), 1), -1)
    sin_theta = 0.5 * np.sqrt(max(0, (3 - trR) * (1 + trR)))
    theta = np.arctan2(sin_theta, cos_theta)
    R_minus_R_T_vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    if abs(3 - trR) < 1e-8:
        # return log map at Theta = 0
        c = 0.5 * (1 + theta*theta / 6 + 7 / 360 * (theta**4))
        return c * R_minus_R_T_vee

    # it diverges around theta=pi
    v = theta / (2 * sin_theta) * R_minus_R_T_vee
    return v