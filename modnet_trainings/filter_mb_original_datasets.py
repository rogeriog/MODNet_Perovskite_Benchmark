import sys
sys.path.append('../')
from Featurization.CompressionFunctions import filter_features_dataset
from modnet.preprocessing import MODData
X=MODData.load('./MODNet_elastic/matbench_elastic/precomputed/matbench_elastic_moddata.pkl.gz')
encoded_columns='../DATAFILES/MP_GapFeats_2.2final_custom_n_b/encoded_columns.txt'
Xfiltered=filter_features_dataset(dataset=X.df_featurized , allowed_features_file=encoded_columns)
X.df_featurized=Xfiltered
X.save('MODNet_elastic_filteredfeats/matbench_elastic/precomputed/matbench_elastic_moddata_filter.pkl.gz')
print(Xfiltered)
