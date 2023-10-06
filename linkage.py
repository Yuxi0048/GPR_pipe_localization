import numpy as np
from matplotlib import pyplot as plt 

def t_linkage(pts, model, num_hypo, num_samples = 10, sigma = 10, cs_thresh = 2, verbose = False):
    
    initial_preference_matrix = build_preference_matrix_tlinkage(pts, model,num_hypo, num_samples, sigma, cs_thresh)
    preference_matrix, clusters = tlinkage_clustering(initial_preference_matrix)
    
    if verbose == True:
        for i in range(len(clusters)):
            plt.plot(pts[clusters[i]][:,0],pts[clusters[i]][:,1],'.')
    
    return clusters

def j_linkage(pts, model, num_hypo, num_samples = 10, sigma = 10, cs_thresh = 2, verbose = False):
    
    initial_preference_matrix = build_preference_matrix_jlinkage(pts, model,num_hypo, num_samples, sigma, cs_thresh)
    preference_matrix, clusters = jlinkage_clustering(initial_preference_matrix)
    
    if verbose == True:
        for i in range(len(clusters)):
            plt.plot(pts[clusters[i]][:,0],pts[clusters[i]][:,1],'.')
    
    return clusters 
 
    
def sample_pts(pts, sigma, num_samples):
    
    num_pts = pts.shape[0]    
    pts_3d = pts.reshape(pts.shape[0], 1, pts.shape[1])
    dist_matrix = np.sqrt(np.einsum('ijk, ijk->ij', pts-pts_3d, pts-pts_3d))
    prob_matrix = np.exp(-dist_matrix**2 /sigma**2)
    np.fill_diagonal(prob_matrix,0)

    pt_lst = []

    for i in range(num_samples):
        if i == 0:
            sampled_index = np.random.choice(num_pts, 1, False)
        else:
            sampled_index = np.random.choice(num_pts, 1, p = prob_select_next)
        pt_select = pts[sampled_index]
        prob_matrix[:,sampled_index] = 0
        prob_select_next= (prob_matrix[sampled_index,:]/np.sum(prob_matrix[sampled_index,:])).squeeze()
        pt_lst.append(pt_select)
   
    return np.array(pt_lst).reshape(num_samples,2)
        
    
def build_preference_matrix_tlinkage(pts, model, num_hypotheses, num_samples, sigma, consensus_threshold):

    num_pts = pts.shape[0]

    preference_mat = np.zeros((num_pts, num_hypotheses)).astype(np.float)

    for i_hypo in range(num_hypotheses):
        pts_samples = sample_pts(pts, sigma, num_samples)
        hyper_hypothesis = model.estimate(pts_samples)
        distances = model.residuals(pts)
        
        for i_pt in range(num_pts):
            distance = distances[i_pt]
            preference_mat[i_pt, i_hypo] = np.exp(-distance/consensus_threshold) if distance < 5*consensus_threshold else 0

    return preference_mat

def build_preference_matrix_jlinkage(pts, model, num_hypotheses, num_samples, sigma, consensus_threshold):

    num_pts = pts.shape[0]

    preference_mat = np.zeros((num_pts, num_hypotheses)).astype(np.float)

    for i_hypo in range(num_hypotheses):
        pts_samples = sample_pts(pts, sigma, num_samples)
        hyper_hypothesis = model.estimate(pts_samples)
        distances = model.residuals(pts)

        for i_pt in range(num_pts):
            distance = distances[i_pt]
            preference_mat[i_pt, i_hypo] = 1 if distance <= consensus_threshold else 0

    return preference_mat

def tlinkage_clustering(preference_mat):

    keep_clustering = True
    cluster_step = 0

    num_clusters = preference_mat.shape[0]
    clusters = [[i] for i in range(num_clusters)]

    while keep_clustering:
        smallest_distance = 1
        best_combo = None
        keep_clustering = False
        
        preference_mat = np.matrix(preference_mat)
        preference_mat_selfprod = preference_mat * preference_mat.T
        num_clusters = preference_mat.shape[0]

        for i in range(num_clusters):
            for j in range(i):
                distance = 1 - preference_mat_selfprod[i,j] / \
                           np.maximum((preference_mat_selfprod[i,i] +
                                       preference_mat_selfprod[j,j] -
                                       preference_mat_selfprod[i,j]), 1e-8)

                if distance < smallest_distance:
                    keep_clustering = True
                    smallest_distance = distance
                    best_combo = (i,j)

        if keep_clustering:
            clusters[best_combo[0]] += clusters[best_combo[1]]
            clusters.pop(best_combo[1])
            set_a = preference_mat[best_combo[0]]
            set_b = preference_mat[best_combo[1]]
            merged_set = np.minimum(set_a, set_b)
            preference_mat[best_combo[0]] = merged_set
            preference_mat = np.delete(preference_mat, best_combo[1], axis=0)
            cluster_step += 1

    print("clustering finished after %d steps" % cluster_step)

    return preference_mat, clusters


def jlinkage_clustering(preference_mat):

    keep_clustering = True
    cluster_step = 0

    num_clusters = preference_mat.shape[0]
    clusters = [[i] for i in range(num_clusters)]

    while keep_clustering:
        smallest_distance = 0
        best_combo = None
        keep_clustering = False

        num_clusters = preference_mat.shape[0]

        for i in range(num_clusters):
            for j in range(i):
                set_a = preference_mat[i]
                set_b = preference_mat[j]
                intersection = np.count_nonzero(np.logical_and(set_a, set_b))
                union = np.count_nonzero(np.logical_or(set_a, set_b))
                distance = 1.*intersection/np.maximum(union, 1e-8)

                if distance > smallest_distance:
                    keep_clustering = True
                    smallest_distance = distance
                    best_combo = (i,j)

        if keep_clustering:
            clusters[best_combo[0]] += clusters[best_combo[1]]
            clusters.pop(best_combo[1])
            set_a = preference_mat[best_combo[0]]
            set_b = preference_mat[best_combo[1]]
            merged_set = np.logical_and(set_a, set_b)
            preference_mat[best_combo[0]] = merged_set
            preference_mat = np.delete(preference_mat, best_combo[1], axis=0)
            cluster_step += 1

    print("clustering finished after %d steps" % cluster_step)

    return preference_mat, clusters