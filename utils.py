import numpy as np

def compute_adjacency(P, edges):
    p_idxs = np.arange(P.shape[0])
    edges_arr = np.array(edges, dtype=np.uint32)
    senders = edges_arr[:, 0]
    receivers = edges_arr[:, 1]
    sortation = np.argsort(senders)
    senders = senders[sortation]
    receivers = receivers[sortation]
    uni_senders, uni_idxs, uni_counts = np.unique(senders, return_index=True, return_counts=True)
    adjacency_list = []
    all_points_list = []
    for i in range(p_idxs.shape[0]):
        p_idx = p_idxs[i]
        if p_idx not in uni_senders:
            adjacency_list.append([])
            continue
        uni_idx = np.where(uni_senders == p_idx)[0][0]
        start = uni_idxs[uni_idx]
        stop = start + uni_counts[uni_idx]
        #print(start[0], stop[0])
        neighbors = receivers[start:stop]
        adjacency_list.append(neighbors.tolist())
    return adjacency_list


def point_idxs_from_p_vec(p_vec):
    point_idxs = []
    uni_p_vec = np.unique(p_vec)
    for i in range(uni_p_vec.shape[0]):
        label = uni_p_vec[i]
        idxs = np.where(p_vec == label)[0]
        point_idxs.append(idxs)
    return point_idxs