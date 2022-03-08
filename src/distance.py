import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from nltk.metrics import edit_distance


def vector_distance_matrix(activations, metric="cosine"):
    return squareform(pdist(activations, metric=metric))


def string_distance_matrix(text, normalize=False):
    dist_mat = np.zeros((len(text), len(text)))
    for i in range(len(text)):
        for j in range(len(text)):
            dist_mat[i, j] = edit_distance(text[i], text[j])

    if normalize:
        norm = np.linalg.norm(dist_mat)
        dist_mat = dist_mat / norm

    return dist_mat


def compute_rsa_score(RDM_1, RDM_2, score="pearsonr"):
    pdists1 = squareform(RDM_1)
    pdists2 = squareform(RDM_2)
    if score != "pearsonr":
        raise NotImplementedError("currently only supporting pearsonr similarity score")
    rsa_score, p_value = pearsonr(pdists1, pdists2)
    return rsa_score


def rsa_matrix(RDMs_1, RDMs_2, score="pearsonr"):
    rsa_mat = np.zeros((len(RDMs_1), len(RDMs_2)))
    for i in range(len(RDMs_1)):
        for j in range(len(RDMs_2)):
            rsa_mat[i, j] = compute_rsa_score(RDMs_1[i], RDMs_2[j], score=score)
    return rsa_mat
