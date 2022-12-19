from ProcessFeatureDatasets import getCumulative_PCA, get_PCAdataset, AppendToMODData
from modnet.preprocessing import MODData
data = MODData.load('./DATAFILES/matbench_perovskites_moddata.pkl.gz')
X=data.df_featurized
print(X)
#getCumulative_PCA(X,datasetname="MODNetPerovskData",savedir='./DATAFILES/PCA_MODNetData/')
#get_PCAdataset(X,476,datasetname="MODNetPerovskData",savedir='./DATAFILES/PCA_MODNetData/',featname="MatMinerPCA")

AppendToMODData("./DATAFILES/",
                "./DATAFILES/matbench_perovskites_moddata.pkl.gz",
                "./DATAFILES/PCA_MODNetData/matbench_perovskites_moddata_PCAMM.pkl.gz",
                mode = 'substitute') ## because we will substitute whole matminer features
