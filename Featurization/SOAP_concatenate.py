import numpy as np
#from dscribe.descriptors import SOAP
import pickle
#import pymatgen
#from pymatgen.io.ase import AseAtomsAdaptor
#from ase.data import atomic_numbers
import os, sys
#import pandas as pd
import glob


## this is very memory intensive
n_soapfiles=len(glob.glob("SOAP_perovsk_featurized_*"))
filename="SOAP_perovsk_featurized_0.pkl"
SOAP_indexes=pickle.load(open(filename,"rb"))
SOAP_indexes=np.array(SOAP_indexes)[:,1]
#SOAP_full=pd.DataFrame.sparse.from_spmatrix(SOAP_full)
for i in range(1, n_soapfiles):
    filename=f"SOAP_perovsk_featurized_{i}.pkl"
    SOAP_tmp=pickle.load(open(filename,"rb"))
    SOAP_tmp=np.array(SOAP_tmp)[:,1]
    SOAP_indexes=np.append(SOAP_indexes, SOAP_tmp,axis = 0)
unique_indexes=np.unique(np.concatenate(SOAP_indexes))
del SOAP_indexes ## free memory
print(f"Total of columns necessary: {len(unique_indexes)}")
# lets create a numpy array ordered according to the unique indexes
SOAP_matrix=np.zeros(len(unique_indexes))
for i in range(31, n_soapfiles):
    filename=f"SOAP_perovsk_featurized_{i}.pkl"
    print(f"Processing {filename}...")
    SOAP_loadedfile=pickle.load(open(filename,"rb"))
    for SOAP_struct in SOAP_loadedfile:
        toinsert=np.nonzero(np.in1d(unique_indexes,SOAP_struct[1]))[0]
        SOAP_tmp=np.zeros(len(unique_indexes))
        SOAP_tmp[toinsert]=SOAP_struct[0]
        SOAP_matrix=np.vstack((SOAP_matrix, SOAP_tmp))
    if i % 30 == 0:
        pickle.dump(SOAP_matrix,open(f"SOAP_matrix_upto{i}.pkl","wb"))
print(SOAP_matrix.shape)
pickle.dump(SOAP_matrix,open(f"SOAP_matrix_final.pkl","wb"))
print('done.')
## approach with pandas not working too heavy
"""
SOAP_DF=pd.DataFrame([],columns=unique_indexes)
del data
del structures
idind=0
for i in range(0,10): # n_soapfiles):
    filename=f"SOAP_perovsk_featurized_{i}.pkl"
    SOAP_loadedfile=pickle.load(open(filename,"rb"))
    for SOAP_struct in SOAP_loadedfile:
        ## there is a shape problem, had to invert columns and index then transpose everything
        SOAP_tmp=pd.DataFrame(SOAP_struct[0],columns=[indexes[idind]],index=SOAP_struct[1]).T 
        SOAP_DF=pd.concat([SOAP_DF, SOAP_tmp],axis=0)
    SOAP_DF.fillna(0)
print('done')
"""
#    print(SOAP_tmp,type(SOAP_tmp))
#    SOAP_full=np.vstack((SOAP_full, SOAP_tmp))
#    print(f"Loaded {filename}.")
#    del SOAP_tmp
#    sys.exit()
#pickle.dump(SOAP_full,open(f"perovsk_SOAP_featurized_full.pkl","wb"))
