from ProcessFeatureDatasets import getCumulative_PCA, get_PCAdataset, AppendToMODData
import pickle
from modnet.preprocessing import MODData
X = pickle.load(open('./DATAFILES/OFM_featurization/OFM_featurizedDF.pkl',"rb"))
#X=data.df_featurized
print(X)
#getCumulative_PCA(X,datasetname="PerovskOFMPCA",savedir='./DATAFILES/PCA_OFM/')
## 207 over 99.5%
#get_PCAdataset(X,207,datasetname="PerovskOFMPCA",savedir='./DATAFILES/PCA_OFM/',featname="OFM_PCA")

AppendToMODData("./DATAFILES/OFM_featurization/OFM_featurizedDF.pkl",
                "./DATAFILES/matbench_perovskites_moddata.pkl.gz",
                "./DATAFILES/matbench_perovskites_moddata_OFMoriginal.pkl.gz",
                mode = 'concat', addidprefix=False ) ## because we will substitute whole matminer features
