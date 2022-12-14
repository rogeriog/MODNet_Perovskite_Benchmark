import numpy as np
from dscribe.descriptors import SOAP
import pymatgen
from pymatgen.io.ase import AseAtomsAdaptor
from ase.data import atomic_numbers
import os
import pandas as pd
import pickle
import glob
structures_ase=pickle.load(open("ase_structures.pkl","rb"))
species=list(atomic_numbers.keys())[1:]
nmax=3 #8
lmax=3 #6
rcut=5
average_soap = SOAP(species=species,
rcut=rcut, nmax=nmax, lmax=lmax,
    average="inner",
    crossover=True,
    periodic=True,
    sparse=False
)

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
ncpus=12
soap_results = average_soap.create(structures_ase[200], n_jobs=ncpus)
results=[soap_results[soap_results.nonzero()[0]],soap_results.nonzero()[0]]
toinsert=np.nonzero(np.in1d(unique_indexes,results[1]))[0]
SOAP_tmp=np.zeros(len(unique_indexes))
SOAP_tmp[toinsert]=results[0]

SOAP_total=pickle.load(open("SOAP_total.pkl","rb"))
print(SOAP_tmp.shape, SOAP_total[200].shape)
#assert(SOAP_tmp == SOAP_total[200])
print(np.alltrue(SOAP_tmp == SOAP_total[200]))
print(np.alltrue(SOAP_tmp == SOAP_total[199]))
print(np.alltrue(SOAP_tmp == SOAP_total[201]))
