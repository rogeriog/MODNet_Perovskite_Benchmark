import numpy as np
import scipy.sparse as sp

mat = np.random.uniform(size=(20,1))
mat_sp = sp.coo_matrix(mat)

for i in range(1,5):
    mat_new = np.random.uniform(size=(20,1))
    mat_sp_new = sp.coo_matrix(mat_new)
    mat_sp = sp.vstack((mat_sp,mat_sp_new))
