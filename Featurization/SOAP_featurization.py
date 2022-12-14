import numpy as np
from dscribe.descriptors import SOAP
import pickle
import pymatgen
from pymatgen.io.ase import AseAtomsAdaptor
from ase.data import atomic_numbers
import os
import pandas as pd
import glob
from modnet.preprocessing import MODData
data=MODData.load('../DATAFILES/matbench_perovskites_moddata.pkl.gz')
structures=data.df_structure['structure']
try:
    structures_ase=pickle.load(open("ase_structures.pkl","rb"))
except:
    structures_ase=list(map(AseAtomsAdaptor.get_atoms,structures))
    pickle.dump(structures_ase,open("ase_structures.pkl","wb"))
    
## declaring the SOAP featurizer
species=list(atomic_numbers.keys())[1:] ## all chemical species
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
print(f"Total number of features: {average_soap.get_number_of_features()}")
ncpus=12 # os.cpu_count()
## this is very memory intensive has to be splitted
slices=[None]+list(range(200,len(structures_ase),200))+[None]

for i, slice1 in list(enumerate(slices))[:-1]:
    #if i < 245 :
    #    continue
    soap_results = average_soap.create(structures_ase[slice1:slices[i+1]], n_jobs=ncpus)
    results=[]
    for j in range(len(soap_results)):
        ## SOAP data is too large and sparse, its better to work with values and index
        results.append([soap_results[j][soap_results[j].nonzero()[0]],
                         soap_results[j].nonzero()[0]])
    pickle.dump(results, open(f"SOAP_perovsk_featurized_{i}.pkl","wb"))
    print(f"Processed SOAP_perovsk_featurized_{i}.pkl, total of {len(slices)-2}")
    del soap_results
    del results
