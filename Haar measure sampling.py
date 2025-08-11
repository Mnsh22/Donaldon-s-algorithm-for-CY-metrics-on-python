'''
def actual_metrics_list():
    metric_list = []
    for i in range(N_t):
        A = sfm[i]
        g = np.zeros((5,5), dtype=complex)
        I = np.eye(5, dtype=complex)
        for m in range(5):
            for n in range(5):
                g[m][n]=(1/np.pi) * ((1/np.einsum("mn,mn,mn", h_new,sfm[i])) * (np.sum(h_new[a][b])))

        metric_list.append(g)
    return metric_list





def error_vol_K(determinant_list, container, h_new):
    I = np.eye(N_k, dtype= complex)
    factor = 0
    for i in range(N_t):

        factor = factor + ((1j /(8*np.pi)) * (np.linalg.det((np.einsum("mn,mn,mn",h_new,I,I))/()))/(25*(abs(container[i]) ** 8))) * (1/(25 * (abs(container[i]) ** 8) * (determinant_list[i])))
'''
