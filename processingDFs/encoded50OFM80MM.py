from ProcessFeatureDatasets import getCumulative_PCA, get_PCAdataset, AppendToMODData
import pickle
from modnet.preprocessing import MODData
#X = pickle.load(open('./DATAFILES/matbench_perovskites_moddata_MMOFMencod50.pkl.gz',"rb"))
X = pickle.load(open('./DATAFILES/PerovskitesOFM_0.5compress.pkl',"rb"))
#X=MODData.load('./DATAFILES/matbench_perovskites_moddata_MMOFMencod50.pkl.gz')
# X = X.df_featurized
print(X)
Y=MODData.load('./DATAFILES/matbench_perovskites_moddata_compressed816.pkl.gz')
#Y = pickle.load(open('./DATAFILES/matbench_perovskites_moddata_compressed816.pkl.gz',"rb"))
Y=Y.df_featurized
print(Y)

#getCumulative_PCA(X,datasetname="PerovskOFMPCA",savedir='./DATAFILES/PCA_OFM/')
## 207 over 99.5%
#get_PCAdataset(X,207,datasetname="PerovskOFMPCA",savedir='./DATAFILES/PCA_OFM/',featname="OFM_PCA")

AppendToMODData("./DATAFILES/PerovskitesOFM_0.5compress.pkl",
                "./DATAFILES/matbench_perovskites_moddata_compressed816.pkl.gz",
                "./DATAFILES/matbench_perovskites_moddata_MM80_OFM50.pkl.gz",
                mode = 'concat', addidprefix=False) ## because we will substitute whole matminer features
