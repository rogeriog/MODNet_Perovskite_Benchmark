from ProcessFeatureDatasets import getCumulative_PCA, get_PCAdataset, AppendToMODData
import pickle
from modnet.preprocessing import MODData
import pandas as pd
## had to fix index
X = pickle.load(open('./DATAFILES/MEGNetFeats16/MEGNetFeats16_Perovsk.pkl',"rb"))
X=X.set_index(pd.Index(list(range(0,len(X)))))
pickle.dump(X,open('./DATAFILES/MEGNetFeats16/MEGNetFeats16_Perovsk.pkl',"wb"))
print(X)
#getCumulative_PCA(X,datasetname="PerovskOFMPCA",savedir='./DATAFILES/PCA_OFM/')
## 207 over 99.5%
#get_PCAdataset(X,207,datasetname="PerovskOFMPCA",savedir='./DATAFILES/PCA_OFM/',featname="OFM_PCA")
AppendToMODData("./DATAFILES/MEGNetFeats16/MEGNetFeats16_Perovsk.pkl",
                "./DATAFILES/matbench_perovskites_moddata.pkl.gz",
                "./DATAFILES/matbench_perovskites_moddata_MM_MEGNetFeats16.pkl.gz",
                mode = 'concat') #, addidprefix=False) ## because we will substitute whole matminer features
