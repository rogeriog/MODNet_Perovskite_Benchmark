import sys
from modnet.preprocessing import MODData
import pandas as pd
# setting path
#sys.path.append('../')
#from MEGNetCustomTraining import MEGNetCustomTraining
#import pickle
#import glob
def AppendToMODData(data_to_concatenate, reference_filename, saved_filename):
    file="result_OFMclustering_perovskites.pkl"
    modfile="/path/to/featurized/matbench_perovskites/precomputed/matbench_perovskites_moddata.pkl.gz"
    dataToConcat=pickle.load(open(data_to_concatenate,"rb"))
    dataToConcat.index="id"+dataToConcat.index.astype(str)
    print(dataToConcat)
    dataReference = MODData.load(reference_filename) # precomputed_moddata)
    print(dataReference.df_featurized)
    concatDF=pd.concat([dataReference.df_featurized,dataToConcat],axis=1)
    dataReference.df_featurized=concatDF
    print(concatDF)
    dataReference.save(saved_filename)
    print(dataReference.df_featurized)

AppendToMODData("result_OFMclustering_perovskites.pkl","/path/to/featurized/matbench_perovskites/precomputed/matbench_perovskites_moddata.pkl.gz","matbench_perovskites_moddataOFM.pkl.gz")
