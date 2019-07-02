import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy


## metric for measuring the disentanglement - Mutual Information Gap (MIG)

def compute_mig(v_array, z_array):  # ground truth, latent variables

    v_array = v_array / np.sum(v_array)  # normalize
    z_array = z_array / np.sum(z_array)  # normalize

    if len(v_array.shape) <= 1:  # pad the array
        v_array = np.expand_dims(v_array, axis=0)
    else:
        v_array = np.reshape(v_array, (v_array.shape[1], v_array.shape[0]))

    if len(z_array.shape) <= 1:  # pad the array
        z_array = np.expand_dims(z_array, axis=0)
    else:
        z_array = np.reshape(z_array, (z_array.shape[1], z_array.shape[0]))

    ## MIG Computation__________________________________________

    mig = -1
    max_mi = -1

    for i, v in enumerate(v_array):  # ground truth
        for j, z in enumerate(z_array):  # latent variables
            if (compute_mi(v, z)) > max_mi:
                max_mi = compute_mi(v, z)
                ## max_mi computed

    for i, v in enumerate(v_array):  # ground truth
        ## computation of MIG
        v_entropy = entropy(v)

        for j, z in enumerate(z_array):  # latent variables

            mig += (1 / float(v_entropy)) * (compute_mi(v, z) - max_mi)

    mig = -((1 / float(z_array.shape[0])) * mig)
    print("Mutual Information Gap:", mig)
    return mig


## metric for measuring the disentanglement - Mutual Information Gap (MIG)

def compute_mig2(v_array, z_array):  # ground truth, latent variables

    ## reshape input data______________________________________

    if len(v_array.shape) <= 1:  # pad the array
        v_array = np.expand_dims(v_array, axis=0)
    else:
        v_array = np.reshape(v_array, (v_array.shape[1], v_array.shape[0]))

    if len(z_array.shape) <= 1:  # pad the array
        z_array = np.expand_dims(z_array, axis=0)
    else:
        z_array = np.reshape(z_array, (z_array.shape[1], z_array.shape[0]))

    ## MIG Computation__________________________________________

    mi_diff = 0
    dim_diff = 1
    mi = np.zeros((v_array.shape[0], z_array.shape[0]))

    for i in range(v_array.shape[0]):  # ground truth
        for j in range(z_array.shape[0]):  # latent variables
            mi[i, j] = compute_mi(v_array[i], z_array[j])

    for i, v in enumerate(v_array):  # ground truth
        ## computation of MIG
        z_order = np.argsort(mi[i,])[::-1]  # descending
        maximum_mi = mi[i, z_order[0]]

        for j in range(z_array.shape[0]):  # latent variables

            mi_diff += mi[i, j] / maximum_mi
            dim_diff += np.abs(z_array.shape[0] - v_array.shape[0])

    mig = 1 + ((mi_diff - (maximum_mi * v_array.shape[0])) / dim_diff)

    print("Mutual Information Gap:", mig)
    return mig


## measure of the mutual dependence between the two variables

def compute_mi(v, z):
    # mi = np.log(np.sum(v) + np.sum(z)) + entropy(z)
    # mi = np.log(np.sum(z@v)) + entropy(z)
    mi = mutual_info_score(v, z, contingency=None)

    # print("mi:", mi)

    return mi

