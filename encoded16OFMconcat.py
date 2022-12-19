from ProcessFeatureDatasets import getCumulative_PCA, get_PCAdataset, AppendToMODData
import pickle
from modnet.preprocessing import MODData
#X = pickle.load(open('./DATAFILES/PerovskitesOFM_0.5compress.pkl',"rb"))
X = pickle.load(open('./DATAFILES/PerovskitesOFM_0.16compress.pkl',"rb"))
#X=data.df_featurized
print(X)
#getCumulative_PCA(X,datasetname="PerovskOFMPCA",savedir='./DATAFILES/PCA_OFM/')
## 207 over 99.5%
#get_PCAdataset(X,207,datasetname="PerovskOFMPCA",savedir='./DATAFILES/PCA_OFM/',featname="OFM_PCA")

#AppendToMODData("./DATAFILES/PerovskitesOFM_0.5compress.pkl",
#                "./DATAFILES/matbench_perovskites_moddata.pkl.gz",
#                "./DATAFILES/matbench_perovskites_moddata_MMOFMencod50.pkl.gz",
#                mode = 'concat', addidprefix=False) ## because we will substitute whole matminer features
AppendToMODData("./DATAFILES/PerovskitesOFM_0.16compress.pkl",
                "./DATAFILES/matbench_perovskites_moddata.pkl.gz",
                "./DATAFILES/matbench_perovskites_moddata_MMOFMencod16.pkl.gz",
                mode = 'concat', addidprefix=False) ## because we will substitute whole matminer features
