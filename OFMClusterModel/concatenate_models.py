import sys
# setting path
#sys.path.append('../')
#from MEGNetCustomTraining import MEGNetCustomTraining
#import pickle
#import glob

file="result_OFMclustering_perovskites.pkl"
modfile="/path/to/featurized/matbench_perovskites/precomputed/matbench_perovskites_moddata.pkl.gz"
from modnet.preprocessing import MODData, CompositionContainer
dataOFM=pickle.load(open(file,"rb"))
dataOFM.index="id"+dataOFM.index.astype(str)
print(dataOFM)
data = MODData.load(modfile) # precomputed_moddata)
print(data.df_featurized)
import pandas as pd
concat_df=pd.concat([data.df_featurized,dataOFM],axis=1)
data.df_featurized=concat_df
print(concat_df)
data.save(f"./matbench_perovskites_moddataOFM.pkl.gz")
print(data.df_featurized)
