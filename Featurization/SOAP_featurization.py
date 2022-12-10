import numpy as np
from dscribe.descriptors import SOAP
import pickle
import pymatgen
from pymatgen.io.ase import AseAtomsAdaptor
from ase.data import atomic_numbers
import os
import pandas as pd

#from modnet.preprocessing import MODData
#data=MODData.load('../DATAFILES/matbench_perovskites_moddata.pkl.gz')
#structures=data.df_structure['structure']
#try:
structures_ase=pickle.load(open("ase_structures.pkl","rb"))
#except:
#    structures_ase=list(map(AseAtomsAdaptor.get_atoms,structures))
#    pickle.dump(structures_ase,open("ase_structures.pkl","wb"))
    
## declaring the SOAP featurizer
species=list(atomic_numbers.keys())[1:] ## all chemical species
nmax=8
lmax=6
rcut=5
average_soap = SOAP(species=species,
rcut=rcut, nmax=nmax, lmax=lmax,
    average="inner",
    crossover=True,
    periodic=True,
    sparse=False
)
ncpus=os.cpu_count()
## this is very memory intensive has to be splitted
slices=[None]+list(range(25,len(structures_ase),25))+[None]
for i, slice1 in list(enumerate(slices))[:-1]:
    results = average_soap.create(structures_ase[slice1:slices[i+1]], n_jobs=ncpus)
    pickle.dump(results, open(f"SOAP_perovsk_featurized_{i}.pkl","wb"))
    del results
    print(f"{i} out of {len(slices)-2} subsets to complete SOAP featurization")
